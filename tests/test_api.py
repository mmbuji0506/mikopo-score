import pytest
from fastapi.testclient import TestClient

from mikopo.api import app, get_model

GOOD = dict(age=45, employment_type="salaried", monthly_income_tzs=3_000_000, loan_amount_tzs=2_000_000,
            loan_term_months=24, existing_loans=0, months_with_bank=100,
            mobile_money_monthly_txn=40, mobile_money_avg_balance_tzs=500_000, past_delinquencies=0)
RISKY = dict(age=20, employment_type="farmer", monthly_income_tzs=250_000, loan_amount_tzs=2_000_000,
             loan_term_months=6, existing_loans=4, months_with_bank=1,
             mobile_money_monthly_txn=2, mobile_money_avg_balance_tzs=5_000, past_delinquencies=3)


@pytest.fixture()
def client(small_model):
    app.dependency_overrides[get_model] = lambda: small_model     # use the small test model
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").status_code == 200


def test_good_applicant_is_safer_than_risky_applicant(client):
    good = client.post("/score", json=GOOD).json()
    risky = client.post("/score", json=RISKY).json()
    assert good["probability_of_default"] < risky["probability_of_default"]
    assert good["score"] > risky["score"]
    assert good["decision"] == "approve"
    assert risky["decision"] == "decline"
    assert len(risky["reasons"]) == 3


def test_missing_income_is_accepted(client):
    body = {**GOOD, "monthly_income_tzs": None}
    assert client.post("/score", json=body).status_code == 200


def test_invalid_age_is_rejected_with_422(client):
    assert client.post("/score", json={**GOOD, "age": 12}).status_code == 422


def test_unknown_employment_type_is_rejected(client):
    assert client.post("/score", json={**GOOD, "employment_type": "pilot"}).status_code == 422
