"""Prediction API for the churn classification demo."""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.churn_ml.data import FEATURES
from src.churn_ml.train import MODEL_PATH, train_and_evaluate

app = FastAPI(title="Aman Customer Churn Risk API", version="1.0.0",
              description="Educational demo trained on synthetic data.")


class Customer(BaseModel):
    tenure_months: int = Field(ge=1, le=120)
    monthly_fee: float = Field(gt=0, le=1000)
    support_tickets: int = Field(ge=0, le=100)
    contract_type: str = Field(pattern="^(month_to_month|one_year|two_year)$")
    payment_method: str = Field(pattern="^(card|bank_transfer|digital_wallet)$")
    has_streaming: bool


@app.get("/")
def home():
    return {"project": "Aman MLOps Churn Project", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(customer: Customer):
    if not MODEL_PATH.exists():
        try:
            train_and_evaluate()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model unavailable; run training first.") from exc
    try:
        bundle = joblib.load(MODEL_PATH)
        row = pd.DataFrame([customer.model_dump()], columns=FEATURES)
        probability = float(bundle["model"].predict_proba(row)[0, 1])
        return {
            "churn_probability": round(probability, 4),
            "risk_band": "high" if probability >= 0.65 else "medium" if probability >= 0.35 else "low",
            "disclaimer": "Synthetic-data educational demo; not suitable for real customer decisions.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
