import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pipeline.db import copy_labeled_frame, get_conn, read_training_state
from pipeline.settings import FRAUD_SRC

sys.path.insert(0, FRAUD_SRC)
from preprocessing import run_preproc
import scorer

sample = importlib.import_module('pipeline.supply.sample_next_month')
append = importlib.import_module('pipeline.supply.append_labeled_batch')
slide = importlib.import_module('pipeline.retrain.slide_training_window')
publish = importlib.import_module('pipeline.retrain.publish_model')
evaluate = importlib.import_module('pipeline.retrain.evaluate')
quality = importlib.import_module('pipeline.quality')


def fixture_frame():
    rows = []
    for month in range(1, 4):
        for label in (0, 1):
            rows.append(dict(transaction_time=pd.Timestamp(2019, month, 28, 23, 30),
                             merch='shop', cat_id='food', amount=10 + label * 100,
                             name_1='A', name_2='B', gender='F', street='street',
                             one_city='city', us_state='CA', post_code=12345,
                             lat=40.0, lon=-70.0, population_city=1000, jobs='job',
                             merchant_lat=40.1, merchant_lon=-70.1, target=label))
    return pd.DataFrame(rows)


class FeaturesAndGateTests(unittest.TestCase):
    def test_features_are_row_local_and_do_not_use_target(self):
        raw = fixture_frame()
        original = raw.copy(deep=True)
        features = run_preproc(raw)
        one_by_one = pd.concat([run_preproc(raw.iloc[[i]]) for i in range(len(raw))])
        pd.testing.assert_frame_equal(features, one_by_one)
        pd.testing.assert_frame_equal(raw, original)
        changed = raw.copy()
        changed.target = 1 - changed.target
        pd.testing.assert_frame_equal(features.drop(columns='target'), run_preproc(changed).drop(columns='target'))
        self.assertTrue(np.isfinite(features.select_dtypes('number')).all().all())
        self.assertIn('amount_log1p', features)
        self.assertIn('hour_cos', features)

    def test_gate_rejects_any_regression_and_nonfinite_metrics(self):
        baseline = dict(pr_auc=0.8, roc_auc=0.9, f1=0.6)
        self.assertTrue(evaluate.is_acceptable(baseline, baseline))
        for key in baseline:
            with self.subTest(metric=key):
                self.assertFalse(evaluate.is_acceptable({**baseline, key: baseline[key] - 0.01}, baseline))
                self.assertFalse(evaluate.is_acceptable({**baseline, key: float('nan')}, baseline))

    def test_calendar_shift_clamps_day_in_february(self):
        raw = fixture_frame().iloc[[0]].copy()
        raw.transaction_time = pd.Timestamp('2020-01-31 23:30')
        shifted = sample.shift_into_month(raw, pd.Timestamp('2021-02-01'))
        self.assertEqual(shifted.transaction_time.iloc[0], pd.Timestamp('2021-02-28 23:30'))
        self.assertEqual(raw.transaction_time.iloc[0], pd.Timestamp('2020-01-31 23:30'))


class ConnectionProxy:
    def __init__(self, conn):
        self.conn = conn

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def close(self):
        self.conn.rollback()


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.conn = get_conn()
        self.proxy = ConnectionProxy(self.conn)
        with self.conn.cursor() as cur:
            cur.execute('CREATE TEMP TABLE labeled_transactions (LIKE public.labeled_transactions)')
            cur.execute('CREATE TEMP TABLE training_state (LIKE public.training_state INCLUDING ALL)')
            cur.execute("INSERT INTO training_state (id,window_start,window_end,next_batch_month) "
                        "VALUES (1,'2019-01-01','2019-03-01','2019-04-01')")
            cur.execute('CREATE TEMP TABLE pipeline_runs (kind TEXT,run_id TEXT,batch_month DATE,PRIMARY KEY(kind,run_id))')
            cur.execute('CREATE TEMP TABLE supplied_months (batch_month DATE PRIMARY KEY,rows_added INT)')
        copy_labeled_frame(self.conn, fixture_frame())
        self.conn.commit()
        self.patches = [patch.object(module, 'get_conn', return_value=self.proxy)
                        for module in (sample, append, slide, publish, quality)]
        self.patches += [patch.object(sample, 'TRAINING_DIR', self.root),
                         patch('pipeline.state.get_current_context', return_value={'run_id': 'retry-check'}),
                         patch.object(publish, 'MODELS_DIR', self.root / 'models'),
                         patch.object(publish, 'CURRENT_DATASET', self.root / 'current.parquet'),
                         patch.object(evaluate, 'MODELS_DIR', self.root / 'models'),
                         patch.object(quality, 'REPORTS_DIR', self.root / 'reports'),
                         patch.object(quality, 'get_current_context', return_value={'run_id': 'dq-test'})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.conn.close()
        self.directory.cleanup()

    def count_rows(self):
        with self.conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM labeled_transactions')
            return cur.fetchone()[0]

    def test_supply_retries_do_not_duplicate_or_skip_month(self):
        info = sample.sample_next_month()
        append.append_labeled_batch(info)
        count = self.count_rows()
        append.append_labeled_batch(info)
        repeated = sample.sample_next_month()
        self.assertEqual(info, repeated)
        self.assertEqual(info['batch_month'], '2019-04-01')
        self.assertEqual(count, 12)
        self.assertEqual(self.count_rows(), count)

    def test_copy_failure_rolls_back_data_and_journal(self):
        info = sample.sample_next_month()
        def fail_after_copy(conn, frame):
            copy_labeled_frame(conn, frame)
            raise RuntimeError('simulated failure before commit')
        with patch.object(append, 'copy_labeled_frame', side_effect=fail_after_copy):
            with self.assertRaises(RuntimeError):
                append.append_labeled_batch(info)
        self.assertEqual(self.count_rows(), 6)
        append.append_labeled_batch(info)
        self.assertEqual(self.count_rows(), 12)

    def prepare_decision(self):
        info = sample.sample_next_month()
        append.append_labeled_batch(info)
        window = slide.slide_training_window(info)
        frame = run_preproc(pd.read_parquet(window['dataset_path']))
        model = CatBoostClassifier(iterations=3, depth=2, verbose=False, thread_count=1, allow_writing_files=False)
        categories = ['merch', 'cat_id', 'gender', 'one_city', 'us_state', 'jobs']
        model.fit(frame.drop(columns='target'), frame.target, cat_features=categories)
        models = self.root / 'models'
        models.mkdir()
        model.save_model(str(models / 'my_catboost_initial.cbm'))
        (models / 'latest.txt').write_text('initial\n')
        candidate = models / 'candidate.cbm'
        model.save_model(str(candidate))
        test = self.root / 'test.parquet'
        frame.to_parquet(test)
        decision = evaluate.evaluate_model({**window, 'candidate_path': str(candidate),
                                            'version': 'test_candidate', 'test_proc_path': str(test)})
        self.assertTrue(decision['accepted'])
        return decision

    def test_publication_recovers_after_files_saved_but_db_failed(self):
        decision = self.prepare_decision()
        with patch.object(publish, 'advance_training_state', side_effect=RuntimeError('database failure')):
            with self.assertRaises(RuntimeError):
                publish.publish_model(decision)
        self.assertEqual(str(read_training_state(self.conn)['next_batch_month']), '2019-04-01')
        window = slide.slide_training_window(decision)
        frame = pd.read_parquet(window['dataset_path'])
        self.assertEqual(len(frame), 10)
        self.assertEqual(frame.transaction_time.dt.to_period('M').nunique(), 3)
        publish.publish_model(decision)
        publish.publish_model(decision)
        self.assertEqual(str(read_training_state(self.conn)['next_batch_month']), '2019-05-01')
        self.assertEqual((self.root / 'models/latest.txt').read_text().strip(), 'test_candidate')

    def test_rejected_candidate_leaves_production_unchanged(self):
        decision = self.prepare_decision()
        decision['accepted'] = False
        Path(decision['candidate_path']).with_suffix('.decision.json').write_text(json.dumps(decision))
        publish.publish_model(decision)
        self.assertEqual((self.root / 'models/latest.txt').read_text().strip(), 'initial')
        self.assertFalse((self.root / 'models/my_catboost_test_candidate.cbm').exists())
        self.assertEqual(str(read_training_state(self.conn)['next_batch_month']), '2019-05-01')

    def test_scorer_reloads_published_model_in_the_same_process(self):
        decision = self.prepare_decision()
        features = run_preproc(fixture_frame())
        with patch.object(scorer, 'MODELS_DIR', self.root / 'models'), \
             patch.object(scorer, '_model', None), patch.object(scorer, '_version', None):
            before, _, old_version = scorer.make_pred(features)
            previous_model = scorer._model
            publish.publish_model(decision)
            after, _, new_version = scorer.make_pred(features)
            self.assertEqual(old_version, 'initial')
            self.assertEqual(new_version, 'test_candidate')
            self.assertIsNot(scorer._model, previous_model)
            np.testing.assert_allclose(before.score, after.score)

    def test_dq_report_survives_failed_quality_check(self):
        with self.conn.cursor() as cur:
            cur.execute('UPDATE labeled_transactions SET target=3 WHERE target=1')
        self.conn.commit()
        stats = quality.collect_quality_stats()
        report = quality.write_quality_report(stats)
        with self.assertRaises(ValueError):
            quality.enforce_quality(report)
        self.assertIn('FAIL', Path(report['report_path']).read_text())
        self.assertTrue((self.root / 'reports/latest.html').exists())


if __name__ == '__main__':
    unittest.main()
