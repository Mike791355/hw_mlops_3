"""Обучение CatBoost на новом трёхмесячном окне."""

import json
from pathlib import Path
import os

import pandas as pd
from catboost import CatBoostClassifier

from pipeline.files import write_json
from pipeline.settings import CAT_FEATURES, LABEL, MODELS_DIR


def prepare_features(frame):
    """Отделяет target от признаков и приводит категориальные колонки к строкам.

    Категории совпадают с теми, что ожидает онлайн-скоринг после run_preproc.
    Возвращает матрицу признаков и вектор целевой переменной.
    """
    features = frame.drop(columns=[LABEL]).copy()
    for column in CAT_FEATURES:
        features[column] = features[column].astype(str)
    return features, frame[LABEL]


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает CatBoost на train и сохраняет модель-кандидат.

    Читает train parquet из preproc_info. Обучает 100 итераций на CPU,
    не используя test для обучения или ранней остановки.
    Сохраняет бинарник и метаданные в candidates; при повторе читает готовые.
    Возвращает метаданные с версией и путём к кандидату. latest.txt не меняет.
    """
    version = "cpu_v2_" + preproc_info["batch_month"][:7].replace("-", "")
    directory = MODELS_DIR / "candidates"
    metadata = directory / f"{version}.training.json"
    if metadata.exists():
        result = json.loads(metadata.read_text())
        if not Path(result["candidate_path"]).exists():
            raise FileNotFoundError(result["candidate_path"])
        return result
    train = pd.read_parquet(preproc_info["train_proc_path"])
    x_train, y_train = prepare_features(train)
    model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1, loss_function="Logloss",
        auto_class_weights="Balanced", random_seed=42, thread_count=4,
        task_type="CPU", allow_writing_files=False, verbose=False,
    )
    model.fit(x_train, y_train, cat_features=CAT_FEATURES)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.cbm"
    temporary = path.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    os.replace(temporary, path)
    result = {**preproc_info, "version": version, "candidate_path": str(path),
              "train_rows": len(train), "test_rows": len(pd.read_parquet(preproc_info["test_proc_path"]))}
    write_json(metadata, result)
    return result
