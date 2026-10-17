from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

def say_time():
    print("Scheduled run at:", datetime.now())

with DAG(
    dag_id="scheduled_dag",
    start_date=datetime(2026, 10, 1),
    schedule="*/2 * * * *",
    catchup=False,
) as dag:
    PythonOperator(task_id="say_time", python_callable=say_time)
