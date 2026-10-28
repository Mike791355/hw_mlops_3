import hashlib
import html
import os
from datetime import datetime, timezone

from airflow.operators.python import get_current_context

from pipeline.files import write_json
from pipeline.db import LABELED_COLUMNS, get_conn
from pipeline.settings import REPORTS_DIR


def collect_quality_stats():
    """Читает витрину, возвращает проверки; данных не меняет, повтор безопасен."""
    rules = []

    def rule(name, violations, requirement):
        rules.append({
            "name": name,
            "passed": violations == 0,
            "violations": int(violations),
            "requirement": requirement,
        })

    conn = get_conn()
    try:
        conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='labeled_transactions'")
            columns = dict(cur.fetchall())
            expected_types = {column: "text" for column in LABELED_COLUMNS}
            expected_types.update({column: "double precision" for column in
                                   ("amount", "lat", "lon", "merchant_lat", "merchant_lon")})
            expected_types.update(transaction_time="timestamp without time zone", target="smallint",
                                  post_code="integer", population_city="integer")
            invalid = [column for column, expected in expected_types.items() if columns.get(column) != expected]
            rule("schema", len(invalid), "All 18 required columns have expected SQL types"
                 + (f": invalid {invalid}" if invalid else ""))
            if invalid:
                return {"rules": rules, "rows": None, "checked_at": datetime.now(timezone.utc).isoformat()}
            nulls = " OR ".join(f"{column} IS NULL" for column in LABELED_COLUMNS)
            empty = " OR ".join(f"btrim({column}) = ''" for column in
                                ["merch", "cat_id", "name_1", "name_2", "gender", "street", "one_city", "us_state", "jobs"])
            cur.execute(f"""
                SELECT count(*), min(transaction_time), max(transaction_time),
                    count(*) FILTER (WHERE {nulls}),
                    count(*) FILTER (WHERE {empty}),
                    count(*) FILTER (WHERE target NOT IN (0,1)),
                    count(*) FILTER (WHERE NOT (amount > 0 AND amount < 'Infinity'::float8)),
                    count(*) FILTER (WHERE NOT (lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
                        AND merchant_lat BETWEEN -90 AND 90 AND merchant_lon BETWEEN -180 AND 180))
                FROM labeled_transactions
            """)
            count, first, last, null_count, blank_count, labels, amounts, coordinates = cur.fetchone()
            rule("nonempty", int(count == 0), "Dataset contains at least one transaction")
            rule("nonnull", null_count, "No NULL in any required column")
            rule("nonblank_categories", blank_count, "Text fields are not blank")
            rule("binary_target", labels, "target belongs to {0, 1}")
            rule("positive_finite_amount", amounts, "amount is finite and strictly positive")
            rule("coordinates", coordinates, "Latitude in [-90,90], longitude in [-180,180]")
    finally:
        conn.close()
    return {"rules": rules, "rows": count, "first_time": str(first), "last_time": str(last),
            "checked_at": datetime.now(timezone.utc).isoformat()}


def write_quality_report(stats):
    """Читает проверки, пишет HTML/JSON; повтор заменяет отчёт того же запуска."""
    run_id = get_current_context()["run_id"]
    key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    failed = sum(not rule["passed"] for rule in stats["rules"])
    result = {**stats, "run_id": run_id, "failed_rules": failed}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / f"dq_{key}.json", result)
    rows = []
    for check in stats["rules"]:
        name = html.escape(str(check["name"]))
        requirement = html.escape(str(check["requirement"]))
        status = "PASS" if check["passed"] else "FAIL"
        css_class = "ok" if check["passed"] else "bad"
        rows.append(
            f"<tr><td>{name}</td><td>{requirement}</td>"
            f"<td>{check['violations']}</td><td class='{css_class}'>{status}</td></tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Fraud dataset quality</title>
<style>
body {{ font: 16px system-ui; margin: 40px; max-width: 1200px; }}
table {{ border-collapse: collapse; }}
th, td {{ padding: 10px; border: 1px solid #ccc; text-align: left; }}
.ok {{ color: #176b30; }}
.bad {{ color: #b00020; }}
</style>
<h1>Fraud dataset quality: {'FAIL' if failed else 'PASS'}</h1>
<p>UTC: {html.escape(str(stats['checked_at']))}<br>
Run: {html.escape(run_id)}<br>
Rows: {html.escape(str(stats['rows']))}<br>
Period: {html.escape(str(stats.get('first_time')))} — {html.escape(str(stats.get('last_time')))}</p>
<table><tr><th>Rule</th><th>Requirement</th><th>Violations</th><th>Result</th></tr>
{''.join(rows)}
</table></html>"""
    for name in (f"dq_{key}.html", "latest.html"):
        path = REPORTS_DIR / name
        temporary = path.with_suffix(".html.tmp")
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, path)
    return {"report_path": str(REPORTS_DIR / f"dq_{key}.html"), "failed_rules": failed}


def enforce_quality(report):
    """Читает итог отчёта, падает при нарушениях; ничего не пишет, повтор безопасен."""
    if report["failed_rules"]:
        raise ValueError(f"{report['failed_rules']} DQ rules failed; report: {report['report_path']}")
    return report["report_path"]
