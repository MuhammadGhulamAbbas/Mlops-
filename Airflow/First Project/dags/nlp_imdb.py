"""
nlp_imdb DAG
download -> parse -> preprocess -> train -> evaluate -> deploy

Task: predict the STAR RATING (1-10) of an IMDB movie review from its text.
That is a regression task (the target is a number, not a class).
Model: TF-IDF features + Ridge regression (scikit-learn), saved with joblib.

- Tasks pass data as FILES.
- evaluate fails if the error is too high, so deploy never runs for a bad model.
- pandas / sklearn are imported inside the tasks to keep DAG parsing fast.
"""
import json
import os
import re
import shutil
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---------- paths ----------
BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
DATA = os.path.join(BASE, "data", "imdb")
MODELS = os.path.join(BASE, "models", "imdb")
DEPLOY = os.path.join(BASE, "deploy", "imdb")

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
TAR = os.path.join(DATA, "aclImdb_v1.tar.gz")
REVIEWS = os.path.join(DATA, "reviews.csv")
TRAIN_CSV = os.path.join(DATA, "train.csv")
TEST_CSV = os.path.join(DATA, "test.csv")
MODEL = os.path.join(MODELS, "model.joblib")
METRICS = os.path.join(MODELS, "metrics.json")

# ---------- settings ----------
MAX_RMSE = 2.5   # quality gate: deploy only if test RMSE <= this (ratings are 1-10)


def clean_text(s):
    # Keep identical to clean_text in the Flask app (app_imdb.py)
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ---------- tasks ----------
def download():
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(TAR):
        print("Archive already exists, skipping download:", TAR)
        return
    tmp = TAR + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, TAR)    # only becomes the real file when fully downloaded
    print("Downloaded:", TAR, round(os.path.getsize(TAR) / 1e6, 1), "MB")


def parse():
    import tarfile
    import pandas as pd

    # Files are named like aclImdb/train/pos/123_9.txt  (9 = the star rating)
    pat = re.compile(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt$")
    rows = []
    with tarfile.open(TAR, "r:gz") as tf:    # read straight from the archive, no extraction
        for m in tf:
            if not m.isfile():
                continue
            mt = pat.search(m.name)
            if not mt:
                continue
            text = tf.extractfile(m).read().decode("utf-8", errors="ignore")
            rows.append((mt.group(1), int(mt.group(3)), text))

    if not rows:
        raise ValueError("No reviews found in the archive")
    df = pd.DataFrame(rows, columns=["split", "rating", "text"])
    df.to_csv(REVIEWS, index=False)
    print("Reviews parsed:", len(df))
    print(df.groupby("split")["rating"].agg(["count", "mean"]))


def preprocess():
    import pandas as pd

    df = pd.read_csv(REVIEWS)
    before = len(df)
    df["text"] = df["text"].map(clean_text)
    df = df.dropna().drop_duplicates(subset="text")
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Rows before: {before}, after cleaning: {len(df)}")
    print(f"Train: {len(train)}, Test: {len(test)}")


def train():
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    os.makedirs(MODELS, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)

    pipe = make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True),
        Ridge(alpha=1.0),
    )
    pipe.fit(df["text"], df["rating"])
    joblib.dump(pipe, MODEL)
    print("Model saved:", MODEL)


def evaluate():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pipe = joblib.load(MODEL)
    df = pd.read_csv(TEST_CSV)
    pred = np.clip(pipe.predict(df["text"]), 1, 10)   # ratings are 1-10

    rmse = float(np.sqrt(mean_squared_error(df["rating"], pred)))
    mae = float(mean_absolute_error(df["rating"], pred))
    r2 = float(r2_score(df["rating"], pred))

    metrics = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "test_rows": int(len(df)),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test RMSE: {rmse:.3f}   MAE: {mae:.3f}   R2: {r2:.3f}")

    if rmse > MAX_RMSE:
        raise ValueError(f"RMSE {rmse:.3f} is above {MAX_RMSE}, not deploying")


def deploy():
    os.makedirs(DEPLOY, exist_ok=True)
    shutil.copy(MODEL, os.path.join(DEPLOY, "model.joblib"))
    shutil.copy(METRICS, os.path.join(DEPLOY, "metrics.json"))
    print("Deployed to:", DEPLOY)
    print("Start serve/app_imdb.py (or call POST /reload if it is already running).")


# ---------- DAG ----------
default_args = {"retries": 1, "retry_delay": timedelta(seconds=10)}

with DAG(
    dag_id="nlp_imdb",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["nlp", "imdb", "regression"],
) as dag:
    t_download = PythonOperator(task_id="download", python_callable=download)
    t_parse = PythonOperator(task_id="parse", python_callable=parse)
    t_prep = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_deploy = PythonOperator(task_id="deploy", python_callable=deploy)

    t_download >> t_parse >> t_prep >> t_train >> t_eval >> t_deploy
