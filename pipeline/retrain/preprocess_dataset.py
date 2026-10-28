"""Препроцессинг train и test той же функцией, что и онлайн-скоринг."""

import sys
from pathlib import Path

import pandas as pd

from pipeline.files import write_parquet
from pipeline.settings import FRAUD_SRC


def preprocess_dataset(split_info: dict) -> dict:
    """Применяет run_preproc из fraud_detector/src к сырым train и test.

    Читает пути из split_info. transaction_time, name_1, name_2 и street
    уходят внутри run_preproc, колонка target остаётся для обучения.
    Пишет train_proc.parquet и test_proc.parquet рядом с файлом окна.
    Возвращает пути к ним и метаданные месяца, нужные следующим задачам.
    """
    if FRAUD_SRC not in sys.path:
        sys.path.insert(0, FRAUD_SRC)
    from preprocessing import run_preproc

    directory = Path(split_info["dataset_path"]).parent
    paths = {}
    for part in ("train", "test"):
        path = directory / f"{part}_proc.parquet"
        write_parquet(run_preproc(pd.read_parquet(split_info[f"{part}_raw_path"])), path)
        paths[f"{part}_proc_path"] = str(path)
    return {**split_info, **paths}
