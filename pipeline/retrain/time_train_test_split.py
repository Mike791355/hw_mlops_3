"""Временной train/test split трёхмесячного окна."""

from pathlib import Path

import pandas as pd

from pipeline.files import write_parquet
from pipeline.settings import WINDOW_MONTHS


def time_train_test_split(window_info: dict) -> dict:
    """Делит окно по времени: два более ранних месяца в train, свежий месяц в test.

    Читает parquet окна из window_info, сортирует по transaction_time и
    сохраняет train_raw.parquet и test_raw.parquet рядом с файлом окна.
    Колонка времени остаётся, потому что препроцессинг строит календарные признаки.
    Возвращает пути к частям, месяц батча и подписи месяцев train и test.
    """
    frame = pd.read_parquet(window_info["dataset_path"])
    frame["transaction_time"] = pd.to_datetime(frame.transaction_time)
    frame = frame.sort_values("transaction_time")
    periods = frame.transaction_time.dt.to_period("M")
    if periods.nunique() != WINDOW_MONTHS:
        raise ValueError("Training window must contain exactly three months")
    train, test = frame.loc[periods < periods.max()], frame.loc[periods == periods.max()]
    if train.empty or test.empty or train.target.nunique() != 2 or test.target.nunique() != 2:
        raise ValueError("Both train and test must be nonempty and contain both target classes")
    directory = Path(window_info["dataset_path"]).parent
    train_path, test_path = directory / "train_raw.parquet", directory / "test_raw.parquet"
    write_parquet(train, train_path)
    write_parquet(test, test_path)
    return {**window_info, "train_raw_path": str(train_path), "test_raw_path": str(test_path),
            "train_months": ",".join(sorted(periods[periods < periods.max()].astype(str).unique())),
            "test_month": str(periods.max())}
