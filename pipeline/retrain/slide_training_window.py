"""Сдвиг трёхмесячного окна обучения на один месяц вперёд."""

from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.files import write_parquet
from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive
from pipeline.settings import WINDOW_MONTHS
from pathlib import Path


def slide_training_window(batch_info: dict) -> dict:
    """Собирает окно из WINDOW_MONTHS месяцев, заканчивая месяцем батча.

    Читает строки из labeled_transactions и сохраняет window.parquet
    рядом с файлом батча. При повторе использует готовое окно.
    current.parquet и указатель месяца здесь не меняются.
    Возвращает путь к окну и метаданные батча.
    """
    month = date.fromisoformat(batch_info["batch_month"])
    path = Path(batch_info["batch_path"]).with_name("window.parquet")
    if not path.exists():
        start = month - relativedelta(months=WINDOW_MONTHS - 1)
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(71021)")
            frame = pd.read_sql(
                f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
                "WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time",
                conn, params=(start, month_end_exclusive(month)),
            )
            expected = pd.period_range(start, month, freq="M")
            actual = pd.to_datetime(frame.transaction_time).dt.to_period("M").unique()
            if set(actual) != set(expected):
                raise ValueError(f"Expected {list(expected)}, found {list(actual)}")
            write_parquet(frame, path)
            conn.commit()
        finally:
            conn.close()
    return {**batch_info, "dataset_path": str(path)}
