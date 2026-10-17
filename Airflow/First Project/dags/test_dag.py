from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

def extract():
    print("Getting data...")

def transform():
    print("Cleaning data...")

def load():
    print("Saving data...")

with DAG(
    dag_id="test_dag",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3
