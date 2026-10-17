"""
dl_mnist DAG  (CPU only)
download -> preprocess -> train -> evaluate -> deploy

Small CNN on MNIST handwritten digits.
- Data between tasks is passed as files (torch tensors saved with torch.save).
- train saves a checkpoint after EVERY epoch and resumes if it crashed.
- evaluate fails if accuracy is below MIN_ACC, so deploy never runs for a bad model.
- torch / torchvision are imported inside the tasks to keep DAG parsing fast.
"""
import json
import os
import shutil
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---------- paths ----------
BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
DATA = os.path.join(BASE, "data", "mnist")
MODELS = os.path.join(BASE, "models", "mnist")
DEPLOY = os.path.join(BASE, "deploy", "mnist")

PREP = os.path.join(DATA, "prep.pt")
CKPT = os.path.join(MODELS, "checkpoint.pt")
MODEL = os.path.join(MODELS, "mnist_cnn.pt")
METRICS = os.path.join(MODELS, "metrics.json")

# ---------- settings (small so it runs fast on CPU) ----------
N_TRAIN = 20000     # use 60000 for the full training set
N_TEST = 5000       # use 10000 for the full test set
EPOCHS = 5
BATCH = 128
LR = 1e-3
MIN_ACC = 0.95      # quality gate

MEAN, STD = 0.1307, 0.3081   # standard MNIST normalization


def build_model(nn):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14 -> 7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


# ---------- tasks ----------
def download():
    from torchvision import datasets
    os.makedirs(DATA, exist_ok=True)
    # download=True skips files that are already there
    datasets.MNIST(DATA, train=True, download=True)
    datasets.MNIST(DATA, train=False, download=True)
    print("MNIST ready in", DATA)


def preprocess():
    import torch
    from torchvision import datasets

    tr = datasets.MNIST(DATA, train=True, download=False)
    te = datasets.MNIST(DATA, train=False, download=False)

    g = torch.Generator().manual_seed(42)
    i_tr = torch.randperm(len(tr), generator=g)[:N_TRAIN]
    i_te = torch.randperm(len(te), generator=g)[:N_TEST]

    def prep(x):
        return ((x.float() / 255.0 - MEAN) / STD).unsqueeze(1)   # (N, 1, 28, 28)

    torch.save({
        "X_train": prep(tr.data[i_tr]), "y_train": tr.targets[i_tr],
        "X_test": prep(te.data[i_te]), "y_test": te.targets[i_te],
    }, PREP)
    print(f"Saved {PREP}: train {N_TRAIN} images, test {N_TEST} images")


def train():
    import torch
    from torch import nn

    os.makedirs(MODELS, exist_ok=True)
    torch.manual_seed(42)
    print("Training on: cpu")

    d = torch.load(PREP, weights_only=True)
    X, y = d["X_train"], d["y_train"]

    model = build_model(nn)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"]
        print("Resuming from epoch", start)

    for epoch in range(start, EPOCHS):
        model.train()
        perm = torch.randperm(len(X))
        total, correct = 0.0, 0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(X[idx])
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (out.argmax(1) == y[idx]).sum().item()
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(X):.4f}  train_acc {correct / len(X):.4f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1}, CKPT)

    torch.save({"state_dict": model.state_dict(), "mean": MEAN, "std": STD}, MODEL)
    if os.path.exists(CKPT):
        os.remove(CKPT)
    print("Model saved:", MODEL)


def evaluate():
    import torch
    from torch import nn

    ck = torch.load(MODEL, weights_only=True)
    model = build_model(nn)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    d = torch.load(PREP, weights_only=True)
    X, y = d["X_test"], d["y_test"]
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), 500):
            out = model(X[i:i + 500])
            correct += (out.argmax(1) == y[i:i + 500]).sum().item()
    acc = correct / len(X)

    metrics = {
        "accuracy": acc,
        "test_images": int(len(X)),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test accuracy:", round(acc, 4))

    if acc < MIN_ACC:
        raise ValueError(f"Accuracy {acc:.3f} is below {MIN_ACC}, not deploying")


def deploy():
    os.makedirs(DEPLOY, exist_ok=True)
    shutil.copy(MODEL, os.path.join(DEPLOY, "mnist_cnn.pt"))
    shutil.copy(METRICS, os.path.join(DEPLOY, "metrics.json"))
    print("Deployed to:", DEPLOY)


# ---------- DAG ----------
default_args = {"retries": 1, "retry_delay": timedelta(seconds=10)}

with DAG(
    dag_id="dl_mnist",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["dl", "mnist", "pytorch", "cpu"],
) as dag:
    t_download = PythonOperator(task_id="download", python_callable=download)
    t_prep = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_deploy = PythonOperator(task_id="deploy", python_callable=deploy)

    t_download >> t_prep >> t_train >> t_eval >> t_deploy
