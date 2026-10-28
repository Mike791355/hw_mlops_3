"""Дозапись подготовленного месячного батча в размеченную витрину."""

from datetime import date
import pandas as pd

from pipeline.db import copy_labeled_frame, get_conn


def append_labeled_batch(batch_info: dict) -> dict:
    """Дописывает parquet батча в конец labeled_transactions.

    Читает путь из batch_info. Существующие строки не обновляет и не удаляет.
    Возвращает тот же словарь с числом записанных строк, чтобы следующая
    задача знала, какой месяц только что появился в витрине.
    """
    month = date.fromisoformat(batch_info["batch_month"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(71021)")
        with conn.cursor() as cur:
            cur.execute("SELECT rows_added FROM supplied_months WHERE batch_month=%s", (month,))
            existing = cur.fetchone()
            if existing:
                return {**batch_info, "rows": existing[0]}
            cur.execute("SELECT count(*) FROM labeled_transactions WHERE transaction_time >= %s "
                        "AND transaction_time < %s::date + interval '1 month'", (month, month))
            if cur.fetchone()[0]:
                raise ValueError("Refusing to append over an existing unjournaled month")
        frame = pd.read_parquet(batch_info["batch_path"])
        periods = pd.to_datetime(frame.transaction_time).dt.to_period("M")
        if frame.empty or not (periods == pd.Period(month, freq="M")).all():
            raise ValueError("Batch must contain only its reserved calendar month")
        copy_labeled_frame(conn, frame)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO supplied_months (batch_month, rows_added) VALUES (%s,%s)", (month,len(frame)))
        conn.commit()
        return {**batch_info, "rows": len(frame)}
    finally:
        conn.close()
