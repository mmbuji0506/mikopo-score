"""FastAPI service: score ONE loan applicant in real time.

Run it:      uvicorn mikopo.api:app --reload
Then open:   http://127.0.0.1:8000/docs      (automatic interactive documentation)

An API (Application Programming Interface) lets other systems - a mobile app, a
branch terminal, a loan-origination system - ask the model for a decision over HTTP
without knowing anything about Python or scikit-learn.
"""
import json
from functools import lru_cache
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from mikopo import __version__, config
from mikopo.features import compute_dti
from mikopo.scorecard import decide

app = FastAPI(title="MikopoScore API", version=__version__,
              description="Credit-risk scoring for loan applicants (synthetic-data demo).")


class Applicant(BaseModel):
    """The JSON body the caller must send. Pydantic validates every field automatically:
    a wrong type or an out-of-range value returns a clear HTTP 422 error before our
    model code ever runs (bad inputs must never reach a risk model)."""
    age: int = Field(ge=18, le=100, description="Age in years")
    employment_type: Literal["salaried", "self_employed", "farmer", "informal_trader"]
    monthly_income_tzs: Optional[float] = Field(default=None, ge=0, description="Null if undeclared")
    loan_amount_tzs: float = Field(gt=0)
    loan_term_months: int = Field(gt=0, le=120)
    existing_loans: int = Field(ge=0, le=20)
    months_with_bank: int = Field(ge=0)
    mobile_money_monthly_txn: Optional[float] = Field(default=None, ge=0)
    mobile_money_avg_balance_tzs: Optional[float] = Field(default=None, ge=0)
    past_delinquencies: int = Field(ge=0, le=20)


class Reason(BaseModel):
    feature: str
    value_bin: str
    points_lost: float


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: float
    decision: Literal["approve", "refer", "decline"]
    reasons: list[Reason]          # the 3 factors that cost the applicant the most points
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    """Load the saved scorecard once, then keep it in memory (lru_cache = remember result)."""
    path = config.MODEL_DIR / "scorecard.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m mikopo.pipeline")
    return joblib.load(path)


def model_version() -> str:
    info = config.MODEL_DIR / "metrics.json"
    return json.loads(info.read_text())["version"] if info.exists() else __version__


@app.get("/health")
def health():
    """Used by load balancers / Docker to check the service is alive."""
    return {"status": "ok", "model_loaded": (config.MODEL_DIR / "scorecard.joblib").exists()}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant, model=Depends(get_model)):
    """Return PD, credit score, decision and reason codes for one applicant."""
    row = applicant.model_dump()
    # Compute dti with the SAME function used in training (no training-serving skew).
    income = row["monthly_income_tzs"] if row["monthly_income_tzs"] is not None else np.nan
    row["dti"] = float(compute_dti(row["loan_amount_tzs"], row["loan_term_months"], income))
    # A one-row DataFrame in the exact column layout the model was trained on.
    frame = pd.DataFrame([row])[config.MODEL_FEATURES]
    p = float(model.predict_proba(frame)[0])
    return ScoreResponse(
        probability_of_default=round(p, 4),
        score=round(float(model.score(frame)[0]), 1),
        decision=decide(p),
        reasons=model.reason_codes(frame, n=3)[0],
        model_version=model_version(),
    )
