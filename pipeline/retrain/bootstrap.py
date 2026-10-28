import os
from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostClassifier

from pipeline.db import LABELED_COLUMNS, get_conn
from pipeline.retrain.fit_catboost import prepare_features
from pipeline.settings import CAT_FEATURES, FRAUD_SRC


def train_bootstrap(output=Path('/tmp/bootstrap.cbm')):
    sys.path.insert(0, FRAUD_SRC)
    from preprocessing import run_preproc

    conn = get_conn()
    try:
        raw = pd.read_sql(
            f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
            "WHERE transaction_time >= '2019-01-01' AND transaction_time < '2019-03-01' "
            "ORDER BY transaction_time", conn,
        )
    finally:
        conn.close()
    if set(pd.to_datetime(raw.transaction_time).dt.to_period('M').astype(str)) != {'2019-01', '2019-02'}:
        raise ValueError('Bootstrap requires January and February 2019')
    features, target = prepare_features(run_preproc(raw))
    model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss',
        auto_class_weights='Balanced', random_seed=42, thread_count=4,
        task_type='CPU', allow_writing_files=False, verbose=False,
        metadata={'training_period': '2019-01,2019-02', 'preprocessing': 'v2'},
    )
    model.fit(features, target, cat_features=CAT_FEATURES)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.cbm')
    model.save_model(str(temporary))
    os.replace(temporary, output)
    print(f'Saved {output}: {len(raw)} rows, January-February 2019 only')


if __name__ == '__main__':
    train_bootstrap()
