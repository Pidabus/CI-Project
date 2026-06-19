"""
GradeMinds Flask API
Exposes the 3-stage pipeline (preprocess → MLP → fuzzy risk)
as a REST endpoint for the frontend dashboard.

Endpoints:
  GET  /health    – liveness probe
  GET  /status    – model training status
  POST /predict   – predict G3 and risk for a single student
"""

import os
import threading
import warnings
import importlib

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

_flask_cors_spec = importlib.util.find_spec("flask_cors")
if _flask_cors_spec is not None:
    _flask_cors = importlib.import_module("flask_cors")
    CORS = _flask_cors.CORS
else:
    def CORS(app, **kwargs):
        return app
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data_mlp import train_and_evaluate_mlp
from data_fuzzy import assess_student_risk

app = Flask(__name__)
CORS(app)

# ── global model state ────────────────────────────────────────────────────────
_state = {
    "status": "not_trained",   # not_trained | training | ready | error
    "model": None,
    "scaler": None,
    "encoders": {},
    "feature_cols": [],
    "best_params": {},
    "metrics": {},
    "error": None,
}
_lock = threading.Lock()

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Sensible defaults for columns the UI does not expose
DEFAULTS = {
    "school": "GP", "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "schoolsup": "no", "paid": "no", "activities": "no",
    "nursery": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 3,
}


def _train():
    with _lock:
        _state["status"] = "training"

    try:
        csv = os.path.join(os.path.dirname(__file__), "student-mat.csv")
        df = pd.read_csv(csv, sep=";").dropna()

        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        y = df["G3"]
        X = df.drop(columns=["G3"])
        feature_cols = list(X.columns)

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model, _ = train_and_evaluate_mlp(X_train, X_test, y_train, y_test)

        with _lock:
            _state.update(
                status="ready", model=model, scaler=scaler,
                encoders=encoders, feature_cols=feature_cols,
                best_params=model.get_params(), error=None,
            )
        print("\n[GradeMinds] Model ready ✓")

    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(exc)
        print(f"\n[GradeMinds] Training failed: {exc}")


threading.Thread(target=_train, daemon=True).start()


def _encode_row(raw: dict) -> pd.DataFrame:
    row = {**DEFAULTS, **raw}
    feature_cols = _state["feature_cols"]
    encoders = _state["encoders"]
    record = {}

    for col in feature_cols:
        val = row.get(col)
        if col in encoders:
            le = encoders[col]
            v = str(val) if val is not None else le.classes_[0]
            v = v if v in le.classes_ else le.classes_[0]
            record[col] = int(le.transform([v])[0])
        else:
            record[col] = float(val) if val is not None else 0.0

    df = pd.DataFrame([record], columns=feature_cols)
    scaled = _state["scaler"].transform(df)
    return pd.DataFrame(scaled, columns=feature_cols)


def _risk_category(r: float) -> str:
    if r < 35:  return "At-Risk"
    if r < 55:  return "Borderline"
    if r < 75:  return "Progressing"
    return "On-Track"


def _intervention(r: float) -> str:
    if r < 35:
        return "High-risk. Recommend immediate academic support, tutoring, and parent contact."
    if r < 55:
        return "Borderline. Monitor attendance, provide supplementary materials, check in weekly."
    if r < 75:
        return "Showing progress. Encourage consistent study habits and optional coursework."
    return "Strong profile. Continue current approach; consider enrichment activities."


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/status")
def status():
    with _lock:
        return jsonify({"status": _state["status"], "error": _state["error"]})


@app.post("/predict")
def predict():
    with _lock:
        s = _state["status"]
    if s == "training":
        return jsonify({"error": "Model is still training — please wait."}), 503
    if s != "ready":
        return jsonify({"error": f"Model not ready (status: {s})"}), 503

    data = request.get_json(force=True) or {}
    required = ["sex", "age", "Medu", "Fedu", "studytime",
                 "failures", "famsup", "higher", "internet", "absences", "G1", "G2"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X_row = _encode_row(data)
        pred = float(_state["model"].predict(X_row)[0])
        pred = round(float(np.clip(pred, 0, 20)), 2)
        risk = round(float(assess_student_risk(pred)), 2)

        return jsonify({
            "predicted_g3":   pred,
            "risk_score":     risk,
            "risk_category":  _risk_category(risk),
            "percentile_est": round((pred / 20) * 100, 1),
            "intervention":   _intervention(risk),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    

@app.get("/")
def serve_frontend():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    print("[GradeMinds] http://localhost:5000  |  /status to check training progress")
    app.run(host="0.0.0.0", port=5000, debug=False)
