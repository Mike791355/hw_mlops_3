from airflow.operators.python import get_current_context


def bind_month(conn, kind, choose_month):
    run_id = get_current_context()["run_id"]
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(71021)")
        cur.execute("SELECT batch_month FROM pipeline_runs WHERE kind=%s AND run_id=%s", (kind, run_id))
        existing = cur.fetchone()
        if existing:
            month = existing[0]
        else:
            month = choose_month(conn)
            cur.execute("INSERT INTO pipeline_runs VALUES (%s, %s, %s)", (kind, run_id, month))
    conn.commit()
    return month
