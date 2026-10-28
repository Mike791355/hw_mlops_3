"""Публикация бинарника модели и фиксация нового окна обучения."""

from datetime import date
import os
from pathlib import Path
import shutil

import pandas as pd

from pipeline.files import write_json, write_parquet
from pipeline.db import advance_training_state, get_conn, read_training_state
from pipeline.settings import CURRENT_DATASET, MODELS_DIR


def publish_model(decision: dict) -> dict:
    """Публикует принятую модель и фиксирует обработанный месяц.

    Читает сохранённого кандидата и решение проверки метрик.
    Если модель принята, копирует её в my_catboost_<version>.cbm,
    пишет метрики и атомарно обновляет latest.txt.
    Затем заменяет current.parquet и обновляет training_state даже при отказе.
    Проверка указателя под блокировкой не даёт сдвинуть окно повторно.
    Возвращает словарь с версией и результатом допуска модели.
    """
    month = date.fromisoformat(decision["batch_month"])
    candidate = Path(decision["candidate_path"])
    if not candidate.exists() or not candidate.with_suffix(".decision.json").exists():
        raise FileNotFoundError("Candidate and gate decision must be saved before commit")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(71021)")
            cur.execute("SELECT id FROM training_state WHERE id=1 FOR UPDATE")
        state = read_training_state(conn)
        if state["next_batch_month"] > month:
            return {"version": decision["version"], "accepted": decision["accepted"], "already_committed": True}
        if state["next_batch_month"] != month:
            raise ValueError("Cannot skip an unprocessed training month")
        if decision["accepted"]:
            latest = MODELS_DIR / "latest.txt"
            current = latest.read_text().strip()
            if current not in (decision["incumbent_version"], decision["version"]):
                raise ValueError("Production changed after evaluation; refusing stale comparison")
            final = MODELS_DIR / f"my_catboost_{decision['version']}.cbm"
            temporary = final.with_suffix(".tmp.cbm")
            shutil.copyfile(candidate, temporary)
            os.replace(temporary, final)
            write_json(final.with_suffix(".metrics.json"), decision)
            pointer = MODELS_DIR / ".latest.tmp"
            pointer.write_text(decision["version"] + "\n")
            os.replace(pointer, latest)
        write_parquet(pd.read_parquet(decision["dataset_path"]), CURRENT_DATASET)
        advance_training_state(conn, month)
        conn.commit()
    finally:
        conn.close()
    return {"version": decision["version"], "accepted": decision["accepted"]}
