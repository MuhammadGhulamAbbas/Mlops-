"""
Flask app that serves the model deployed by the ml_wine DAG.
Run:  python app.py      (then open http://localhost:5001/health)
"""
import os

import torch
from torch import nn
from flask import Flask, jsonify, request

BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
MODEL_PATH = os.path.join(BASE, "deploy", "model.pt")

app = Flask(__name__)
state = {}


def build_model(n_in):
    # Keep identical to build_model in ml_wine.py
    return nn.Sequential(
        nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


def load_model():
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = build_model(ck["n_in"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    state["model"] = model
    state["features"] = ck["features"]
    state["mean"] = torch.tensor(ck["mean"], dtype=torch.float32)
    state["scale"] = torch.tensor(ck["scale"], dtype=torch.float32)
    print("Model loaded from", MODEL_PATH)


@app.get("/health")
def health():
    return jsonify(status="ok", features=state["features"])


@app.post("/reload")
def reload_model():
    load_model()   # pick up a newly deployed model without restarting
    return jsonify(status="reloaded")


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    values = data.get("features")
    n = len(state["features"])
    if not isinstance(values, list) or len(values) != n:
        return jsonify(error=f"send 'features' as a list of {n} numbers",
                       order=state["features"]), 400
    x = (torch.tensor(values, dtype=torch.float32) - state["mean"]) / state["scale"]
    with torch.no_grad():
        p = torch.sigmoid(state["model"](x.unsqueeze(0))).item()
    return jsonify(good_wine_probability=round(p, 4),
                   prediction="good" if p >= 0.5 else "not good")


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5001)
