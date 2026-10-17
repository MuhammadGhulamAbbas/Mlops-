"""
Flask app that serves the model deployed by the nlp_imdb DAG.
Run:  python app_imdb.py     (then open http://localhost:5002/health)
"""
import os
import re

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE = os.environ.get("MLOPS_HOME", "/mnt/c/Users/123/Desktop/MLOPS/Airflow/First Project")
MODEL_PATH = os.path.join(BASE, "deploy", "imdb", "model.joblib")

app = Flask(__name__)
state = {}


def clean_text(s):
    # Keep identical to clean_text in nlp_imdb.py
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_model():
    state["model"] = joblib.load(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)


@app.get("/health")
def health():
    return jsonify(status="ok", task="predict IMDB star rating (1-10) from review text")


@app.post("/reload")
def reload_model():
    load_model()    # pick up a newly deployed model without restarting
    return jsonify(status="reloaded")


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="send JSON like {\"text\": \"your review here\"}"), 400
    rating = float(np.clip(state["model"].predict([clean_text(text)])[0], 1, 10))
    return jsonify(predicted_rating=round(rating, 2),
                   sentiment="positive" if rating >= 5.5 else "negative")


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5002)
