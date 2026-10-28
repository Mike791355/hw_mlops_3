"""Извлечение очередного месяца из размеченной витрины."""

import pandas as pd
from airflow.exceptions import AirflowSkipException

from pipeline.files import write_parquet
from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive, read_training_state
from pipeline.state import bind_month
from pipeline.settings import BATCHES_DIR


def extract_month_batch() -> dict:
    """Забирает из labeled_transactions месяц, закреплённый за запуском.

    При первом вызове читает next_batch_month из training_state.
    Сохраняет месяц в parquet, при повторе использует готовый файл.
    Указатель месяца не меняет. Возвращает путь к файлу и месяц в формате ISO.
    Если строк нет, помечает задачу как skipped.
    """
    conn = get_conn()
    try:
        month = bind_month(conn, "retrain", lambda c: read_training_state(c)["next_batch_month"])
        path = BATCHES_DIR / f"{month:%Y%m}" / "batch.parquet"
        if not path.exists():
            frame = pd.read_sql(
                f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
                "WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time",
                conn, params=(month, month_end_exclusive(month)),
            )
            if frame.empty:
                raise AirflowSkipException(f"No labeled data for {month}")
            write_parquet(frame, path)
    finally:
        conn.close()
    return {"batch_path": str(path), "batch_month": month.isoformat()}
