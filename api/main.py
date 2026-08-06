"""FastAPI service that serves the trained churn model."""

import os

import joblib
import pandas as pd
from fastapi import FastAPI

from schema import Customer

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(title="Churn Scoring API", version="1.0")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "will_churn": bool(proba >= 0.5),
        "risk": "high" if proba >= 0.5 else "low",
    }
