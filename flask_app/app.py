from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib, json, re, time

app = Flask(__name__)
CORS(app)

# load models and artifacts
model = joblib.load("../model/xgb_model.pkl")  # keep your path
scaler = joblib.load("artifacts/scaler.pkl")
final_columns = json.load(open("artifacts/final_columns.json"))
allowed_categories = json.load(open("artifacts/allowed_categories.json"))
value_maps = json.load(open("artifacts/value_maps.json"))
experience_map = value_maps["experience_map"]
last_new_job_map = value_maps["last_new_job_map"]

NUM_COLS = ["city_development_index", "experience", "last_new_job", "training_hours"]
CAT_COLS = [
    "relevent_experience", "enrolled_university", "education_level",
    "major_discipline", "company_size", "company_type"
]

def clean_names(cols):
    return [re.sub(r'[<>\[\]\s]', '_', c) for c in cols]

def preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Fill Unknowns for categoricals
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna("Unknown").astype(str)

    # Map buckets to numbers
    def map_experience(x):
        # allow "3", 3, or "<1", ">20"
        s = str(x).strip()
        return experience_map[s] if s in experience_map else (int(s) if s.isdigit() else np.nan)

    def map_last_new_job(x):
        s = str(x).strip()
        return last_new_job_map[s] if s in last_new_job_map else (int(s) if s.isdigit() else np.nan)

    df["experience"] = df.get("experience", np.nan)
    df["experience"] = df["experience"].apply(map_experience)

    df["last_new_job"] = df.get("last_new_job", np.nan)
    df["last_new_job"] = df["last_new_job"].apply(map_last_new_job)

    # Ensure numeric cols exist & cast
    for c in ["city_development_index", "training_hours"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Median fills (mirrors the training approach)
    df["experience"] = df["experience"].fillna(df["experience"].median())
    df["last_new_job"] = df["last_new_job"].fillna(df["last_new_job"].median())
    df["city_development_index"] = df["city_development_index"].fillna(df["city_development_index"].median())
    df["training_hours"] = df["training_hours"].fillna(df["training_hours"].median())

    # One-hot encode categoricals exactly like training
    dummies = pd.get_dummies(df[CAT_COLS], drop_first=True, dtype=int)

    # Combine numeric + dummies
    X = pd.concat([df[NUM_COLS], dummies], axis=1)

    # Clean names and align to training columns
    X.columns = clean_names(X.columns)
    X = X.reindex(columns=final_columns, fill_value=0)

    # Scale numeric columns
    to_scale = [c for c in NUM_COLS if c in X.columns]
    X[to_scale] = scaler.transform(X[to_scale])

    return X

@app.route("/", methods=["GET"])
def home():
    return "Job-switch predictor API is up."

# Existing: model-ready JSON (kept for sample.json)
@app.route("/predict", methods=["POST"])
def predict_model_ready():
    t0 = time.time()
    payload = request.get_json(force=True)
    df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    df.columns = clean_names(df.columns)
    preds = model.predict(df).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()
    out = {
        "prediction": preds if isinstance(payload, list) else preds[0],
        "probability": probs if isinstance(payload, list) else probs[0],
        "model_version": "xgb_v1",
        "latency_ms": int((time.time() - t0) * 1000),
    }
    return jsonify(out)

# New: raw human-friendly JSON
@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    t0 = time.time()
    payload = request.get_json(force=True)
    df_raw = pd.DataFrame(payload if isinstance(payload, list) else [payload])

    # Friendly validation for categoricals
    for col, choices in allowed_categories.items():
        if col in df_raw.columns:
            bad = ~df_raw[col].fillna("Unknown").isin(choices)
            if bad.any():
                return jsonify({
                    "error": "validation_error",
                    "details": {col: f"value(s) must be one of: {choices}"}
                }), 400

    X = preprocess_raw(df_raw)
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)[:, 1].tolist()

    out = {
        "prediction": preds if isinstance(payload, list) else preds[0],
        "probability": probs if isinstance(payload, list) else probs[0],
        "model_version": "xgb_v1",
        "latency_ms": int((time.time() - t0) * 1000),
    }
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
