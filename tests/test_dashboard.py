"""Smoke tests: the dashboard runs and reacts correctly to different applicants.

Skipped automatically if Streamlit is not installed or the pipeline has not been run yet.
"""
import pytest

from mikopo import config

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (config.MODEL_DIR / "scorecard.joblib").exists(),
                                reason="run python -m mikopo.pipeline first")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(config.ROOT / "dashboard" / "app.py"), default_timeout=60).run()


def test_dashboard_runs_without_errors():
    at = _app()
    assert not at.exception
    assert len(at.tabs) == 5
    assert len(at.metric) == 3                      # PD, score, decision


def test_risky_applicant_is_declined_and_safe_applicant_approved():
    at = _app()
    # risky: young farmer, low income, big loan, many loans, 3 past delinquencies
    at.slider[0].set_value(20)                      # age
    at.selectbox[0].set_value("farmer")             # employment type
    at.slider[1].set_value(4)                       # existing loans
    at.slider[2].set_value(3)                       # past delinquencies
    at.number_input[0].set_value(250_000)           # monthly income
    at.number_input[1].set_value(2_000_000)         # loan amount
    at.run()
    assert not at.exception
    assert at.metric[2].value == "DECLINE"

    # safe: older salaried borrower, high income, no debts or delinquencies
    at.slider[0].set_value(45)
    at.selectbox[0].set_value("salaried")
    at.slider[1].set_value(0)
    at.slider[2].set_value(0)
    at.number_input[0].set_value(3_000_000)
    at.number_input[2].set_value(100)               # months with the bank
    at.run()
    assert not at.exception
    assert at.metric[2].value == "APPROVE"
