"""Случайная выборка 100 тысяч строк и перенос их в следующий месяц."""

import calendar
import logging

import pandas as pd

from pipeline.db import LABELED_COLUMNS, get_conn, read_training_state
from pipeline.settings import SAMPLE_SIZE, TRAINING_DIR, HISTORY_YEARS
from pipeline.files import write_parquet
from pipeline.state import bind_month
from dateutil.relativedelta import relativedelta
from airflow.exceptions import AirflowSkipException

logger = logging.getLogger(__name__)


def next_month_start(max_time: pd.Timestamp) -> pd.Timestamp:
    """Возвращает первое число месяца, следующего за месяцем max_time."""
    return (pd.Timestamp(max_time).to_period("M") + 1).to_timestamp()


def shift_into_month(frame: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    """Переписывает transaction_time в заданный месяц, сохраняя день, час и минуту.

    Если в целевом месяце нет такого дня, день обрезается до последнего числа.
    Секунды обнуляются. Исходный frame не меняется.
    """
    shifted = frame.copy()
    times = pd.to_datetime(shifted["transaction_time"])
    year = int(month_start.year)
    month = int(month_start.month)
    last_day = calendar.monthrange(year, month)[1]
    shifted["transaction_time"] = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": times.dt.day.clip(upper=last_day),
            "hour": times.dt.hour,
            "minute": times.dt.minute,
        }
    )
    return shifted


def choose_supply_month(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT min(r.batch_month) FROM pipeline_runs r
            LEFT JOIN supplied_months s ON s.batch_month=r.batch_month
            WHERE r.kind='supply' AND s.batch_month IS NULL
        """)
        pending = cur.fetchone()[0]
        if pending:
            return pending
        cur.execute("SELECT max(transaction_time) FROM labeled_transactions")
        newest = cur.fetchone()[0]
    if newest is None:
        raise ValueError("Labeled showcase is empty")
    month = next_month_start(newest).date()
    if month - relativedelta(years=HISTORY_YEARS) > read_training_state(conn)["window_start"]:
        raise AirflowSkipException("Supply waits for retraining to protect required history")
    return month


def sample_next_month() -> dict:
    """Сэмплирует строки витрины и готовит батч выбранного месяца.

    Берёт до SAMPLE_SIZE наблюдений из labeled_transactions и переносит даты.
    Месяц закрепляется за запуском: незавершённый или следующий после max времени.
    Пишет supply/YYYYMM.parquet в TRAINING_DIR; при повторе читает готовый файл.
    Витрину не изменяет. Возвращает путь, месяц и число строк.
    """
    conn = get_conn()
    try:
        month = bind_month(conn, "supply", choose_supply_month)
        path = TRAINING_DIR / "supply" / f"{month:%Y%m}.parquet"
        if not path.exists():
            frame = pd.read_sql(
                f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions ORDER BY random() LIMIT %s",
                conn, params=(SAMPLE_SIZE,),
            )
            if frame.empty:
                raise ValueError("No rows to sample")
            write_parquet(shift_into_month(frame, pd.Timestamp(month)), path)
        rows = len(pd.read_parquet(path))
    finally:
        conn.close()
    return {"batch_path": str(path), "batch_month": month.isoformat(), "rows": rows}
