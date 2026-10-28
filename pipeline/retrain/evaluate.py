import json
import math
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from pipeline.files import write_json
from pipeline.settings import MODELS_DIR

GATE_METRICS = ("pr_auc", "roc_auc", "f1")


def model_metrics(model, frame):
    features = frame.loc[:, model.feature_names_].copy()
    for index in model.get_cat_feature_indices():
        features.iloc[:, index] = features.iloc[:, index].astype(str)
    probabilities = model.predict_proba(features, thread_count=4)[:, 1]
    target = frame["target"]
    if target.nunique() != 2:
        raise ValueError("The control month must contain both target classes")
    return {"pr_auc": float(average_precision_score(target, probabilities)),
            "roc_auc": float(roc_auc_score(target, probabilities)),
            "f1": float(f1_score(target, probabilities > 0.5, zero_division=0))}


def is_acceptable(candidate, incumbent):
    return all(math.isfinite(candidate[key]) and math.isfinite(incumbent[key])
               and candidate[key] + 1e-12 >= incumbent[key] for key in GATE_METRICS)


def evaluate_model(train_info: dict) -> dict:
    """Читает модели и test, пишет решение в JSON; повтор читает готовое решение."""
    path = Path(train_info["candidate_path"]).with_suffix(".decision.json")
    if path.exists():
        return json.loads(path.read_text())
    incumbent_version = (MODELS_DIR / "latest.txt").read_text().strip()
    incumbent = CatBoostClassifier()
    incumbent.load_model(str(MODELS_DIR / f"my_catboost_{incumbent_version}.cbm"))
    candidate = CatBoostClassifier()
    candidate.load_model(train_info["candidate_path"])
    test = pd.read_parquet(train_info["test_proc_path"])
    candidate_metrics = model_metrics(candidate, test)
    incumbent_metrics = model_metrics(incumbent, test)
    accepted = is_acceptable(candidate_metrics, incumbent_metrics)
    result = {**train_info, "incumbent_version": incumbent_version,
              "candidate_metrics": candidate_metrics, "incumbent_metrics": incumbent_metrics,
              "accepted": accepted, "reason": "all metrics non-decreasing" if accepted else "metric regression"}
    write_json(path, result)
    return result
