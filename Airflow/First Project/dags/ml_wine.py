"""
ml_wine DAG
extract -> transform -> train -> evaluate -> deploy
transform -> load (SQLite, side branch)

Everything between tasks is passed as FILES, not Python objects.
Heavy libraries (pandas, torch, sklearn) are imported INSIDE the tasks,
so the DAG file stays fast to parse.
"""
import json
import os
import shutil
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---------- paths ----------
BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
DEPLOY = os.path.join(BASE, "deploy")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
RAW = os.path.join(DATA, "wine_raw.csv")
CLEAN = os.path.join(DATA, "wine_clean.csv")
TRAIN_NPZ = os.path.join(DATA, "train.npz")
TEST_NPZ = os.path.join(DATA, "test.npz")
SCALER_JSON = os.path.join(DATA, "scaler.json")
DB = os.path.join(DATA, "wine.db")

CKPT = os.path.join(MODELS, "checkpoint.pt")   # resume point during training
MODEL = os.path.join(MODELS, "model.pt")       # final model
METRICS = os.path.join(MODELS, "metrics.json")

# ---------- settings ----------
EPOCHS = 100
BATCH = 64
LR = 1e-3
MIN_ACC = 0.70   # quality gate: deploy only if test accuracy >= this


def build_model(nn, n_in):
    # Keep identical to the model in the Flask app.
    return nn.Sequential(
        nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


# ---------- tasks ----------
def extract():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(RAW):
        print("Raw file already exists, skipping download:", RAW)
        return
    import pandas as pd
    df = pd.read_csv(URL, sep=";")
    df.to_csv(RAW, index=False)
    print("Raw data saved:", df.shape)


def transform():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(RAW)
    before = len(df)
    df = df.drop_duplicates().dropna()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["good"] = (df["quality"] >= 6).astype(int)   # 1 = good wine
    df.to_csv(CLEAN, index=False)

    features = [c for c in df.columns if c not in ("quality", "good")]
    X = df[features].values.astype("float32")
    y = df["good"].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)       # fit on train only
    X_train = scaler.transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    np.savez(TRAIN_NPZ, X=X_train, y=y_train)
    np.savez(TEST_NPZ, X=X_test, y=y_test)
    with open(SCALER_JSON, "w") as f:
        json.dump({
            "features": features,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }, f)
    print(f"Rows before: {before}, after cleaning: {len(df)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, features: {len(features)}")


def load():
    import sqlite3
    import pandas as pd
    df = pd.read_csv(CLEAN)
    con = sqlite3.connect(DB)
    df.to_sql("wine", con, if_exists="replace", index=False)
    con.close()
    print("Loaded into SQLite:", DB, df.shape)


def train():
    import numpy as np
    import torch
    from torch import nn

    os.makedirs(MODELS, exist_ok=True)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    tr = np.load(TRAIN_NPZ)
    X = torch.tensor(tr["X"]).to(device)
    y = torch.tensor(tr["y"]).unsqueeze(1).to(device)
    n_in = X.shape[1]

    model = build_model(nn, n_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    # Resume if a checkpoint exists (e.g. the task crashed earlier)
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"]
        print("Resuming from epoch", start)

    for epoch in range(start, EPOCHS):
        model.train()
        perm = torch.randperm(len(X), device=device)
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(X):.4f}")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch + 1}, CKPT)

    with open(SCALER_JSON) as f:
        sc = json.load(f)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "n_in": n_in,
        "features": sc["features"],
        "mean": sc["mean"],
        "scale": sc["scale"],
    }, MODEL)
    if os.path.exists(CKPT):
        os.remove(CKPT)
    print("Model saved:", MODEL)


def evaluate():
    import numpy as np
    import torch
    from torch import nn

    ck = torch.load(MODEL, map_location="cpu", weights_only=True)
    model = build_model(nn, ck["n_in"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    te = np.load(TEST_NPZ)
    X = torch.tensor(te["X"])
    y = te["y"].astype(int)
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).squeeze(1).numpy()
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y).mean())

    metrics = {
        "accuracy": acc,
        "test_rows": int(len(y)),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test accuracy:", round(acc, 4))

    if acc < MIN_ACC:
        # Task fails -> deploy does not run
        raise ValueError(f"Accuracy {acc:.3f} is below {MIN_ACC}, not deploying")


def deploy():
    os.makedirs(DEPLOY, exist_ok=True)
    shutil.copy(MODEL, os.path.join(DEPLOY, "model.pt"))
    shutil.copy(METRICS, os.path.join(DEPLOY, "metrics.json"))
    print("Deployed to:", DEPLOY)
    print("Now run the Flask app (or call POST /reload if it is already running).")


# ---------- DAG ----------
default_args = {"retries": 1, "retry_delay": timedelta(seconds=10)}

with DAG(
    dag_id="ml_wine",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["ml", "wine", "pytorch"],
) as dag:
    t_extract = PythonOperator(task_id="extract", python_callable=extract)
    t_transform = PythonOperator(task_id="transform", python_callable=transform)
    t_load = PythonOperator(task_id="load", python_callable=load)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_deploy = PythonOperator(task_id="deploy", python_callable=deploy)

    t_extract >> t_transform
    t_transform >> t_load                                  # side branch
    t_transform >> t_train >> t_evaluate >> t_deploy       # main chain
