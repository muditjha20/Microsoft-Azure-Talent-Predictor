from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib, json, re, time, os

app = Flask(__name__)
CORS(app)  # tighten origins later in production

# -------- Paths & Artifacts --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model & artifacts (use robust paths so it works on Render)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
FINAL_COLS_PATH = os.path.join(BASE_DIR, "artifacts", "final_columns.json")
ALLOWED_CATS_PATH = os.path.join(BASE_DIR, "artifacts", "allowed_categories.json")
VALUE_MAPS_PATH = os.path.join(BASE_DIR, "artifacts", "value_maps.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
final_columns = json.load(open(FINAL_COLS_PATH, "r", encoding="utf-8"))
allowed_categories = json.load(open(ALLOWED_CATS_PATH, "r", encoding="utf-8"))
value_maps = json.load(open(VALUE_MAPS_PATH, "r", encoding="utf-8"))
experience_map = value_maps["experience_map"]
last_new_job_map = value_maps["last_new_job_map"]

NUM_COLS = ["city_development_index", "experience", "last_new_job", "training_hours"]
CAT_COLS = [
    "relevent_experience", "enrolled_university", "education_level",
    "major_discipline", "company_size", "company_type"
]

def clean_names(cols):
    # Mirror training name cleaner exactly
    return [re.sub(r'[<>\[\]\s]', '_', c) for c in cols]

def preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Ensure required columns exist
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna("Unknown").astype(str)

    # Map bucket strings to numbers (experience / last_new_job)
    def map_experience(x):
        s = str(x).strip()
        return experience_map[s] if s in experience_map else (int(s) if s.isdigit() else np.nan)

    def map_last_new_job(x):
        s = str(x).strip()
        return last_new_job_map[s] if s in last_new_job_map else (int(s) if s.isdigit() else np.nan)

    df["experience"] = df.get("experience", np.nan)
    df["experience"] = df["experience"].apply(map_experience)

    df["last_new_job"] = df.get("last_new_job", np.nan)
    df["last_new_job"] = df["last_new_job"].apply(map_last_new_job)

    # Numeric casting
    for c in ["city_development_index", "training_hours"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Median fills (same as training)
    df["experience"] = df["experience"].fillna(df["experience"].median())
    df["last_new_job"] = df["last_new_job"].fillna(df["last_new_job"].median())
    df["city_development_index"] = df["city_development_index"].fillna(df["city_development_index"].median())
    df["training_hours"] = df["training_hours"].fillna(df["training_hours"].median())

    # One-hot encode categoricals as trained
    dummies = pd.get_dummies(df[CAT_COLS], drop_first=True, dtype=int)

    # Combine numeric + dummies
    X = pd.concat([df[NUM_COLS], dummies], axis=1)

    # Clean names & align to training columns
    X.columns = clean_names(X.columns)
    X = X.reindex(columns=final_columns, fill_value=0)

    # Scale numeric columns
    to_scale = [c for c in NUM_COLS if c in X.columns]
    X[to_scale] = scaler.transform(X[to_scale])

    return X

# -------- Routes --------
@app.get("/")
def home():
    return "Job-switch predictor API is up."

@app.get("/healthz")
def healthz():
    return jsonify(ok=True)

# Model-ready JSON (kept for sample.json)
@app.post("/predict")
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

# Raw human-friendly JSON (what your frontend calls)
@app.post("/predict_raw")
def predict_raw():
    t0 = time.time()
    payload = request.get_json(force=True)
    df_raw = pd.DataFrame(payload if isinstance(payload, list) else [payload])

    # Friendly categorical validation
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
    
    # For local runs; Render will use Gunicorn
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
