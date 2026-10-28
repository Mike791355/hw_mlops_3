from datetime import datetime, timedelta
from airflow.decorators import dag, task
from pipeline.quality import collect_quality_stats, write_quality_report, enforce_quality


@dag(dag_id="data_quality", schedule="*/5 * * * *", start_date=datetime(2026, 1, 1),
     catchup=False, max_active_runs=1, dagrun_timeout=timedelta(minutes=10),
     default_args={"retries": 1, "retry_delay": timedelta(seconds=30)}, tags=["mlops", "dq"])
def data_quality():
    stats = task(collect_quality_stats)()
    report = task(write_quality_report)(stats)
    task(enforce_quality)(report)


data_quality()
