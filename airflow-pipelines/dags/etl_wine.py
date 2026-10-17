import os
import sqlite3
from datetime import datetime

import pandas as pd
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
DATA = os.path.join(BASE, "data")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
RAW = os.path.join(DATA, "wine_raw.csv")
CLEAN = os.path.join(DATA, "wine_clean.csv")
DB = os.path.join(DATA, "wine.db")

def extract():
    os.makedirs(DATA, exist_ok=True)
    df = pd.read_csv(URL, sep=";")
    df.to_csv(RAW, index=False)
    print("Raw data saved:", df.shape)

def transform():
    df = pd.read_csv(RAW)
    before = len(df)
    df = df.drop_duplicates().dropna()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.to_csv(CLEAN, index=False)
    print(f"Rows before: {before}, after cleaning: {len(df)}")

def load():
    df = pd.read_csv(CLEAN)
    con = sqlite3.connect(DB)
    df.to_sql("wine", con, if_exists="replace", index=False)
    con.close()
    print("Loaded into SQLite:", DB, df.shape)

with DAG(
    dag_id="etl_wine",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3
