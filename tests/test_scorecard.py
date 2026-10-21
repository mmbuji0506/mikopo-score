import numpy as np
import pandas as pd

from mikopo import config
from mikopo.scorecard import Scorecard, decide


def test_probabilities_are_valid(small_model, small_data):
    p = small_model.predict_proba(small_data[config.MODEL_FEATURES])
    assert ((p > 0) & (p < 1)).all()


def test_score_equals_sum_of_points_and_matches_odds_formula(small_model, small_data):
    X = small_data[config.MODEL_FEATURES].head(50)
    score = small_model.score(X)
    p = small_model.predict_proba(X)
    # score = offset + factor * ln(good/bad odds); must agree with the probability.
    expected = small_model.offset + small_model.factor * np.log((1 - p) / p)
    assert np.allclose(score, expected, atol=1e-6)


def test_higher_risk_means_lower_score(small_model, small_data):
    X = small_data[config.MODEL_FEATURES]
    p, s = small_model.predict_proba(X), small_model.score(X)
    # The link between PD and score is a curve (log-odds), not a straight line, so we
    # compare RANKS (Spearman correlation). Perfectly inverse ranking = -1.
    assert pd.Series(p).corr(pd.Series(s), method="spearman") < -0.999


def test_reason_codes_returns_three_ranked_reasons(small_model, small_data):
    reasons = small_model.reason_codes(small_data[config.MODEL_FEATURES].head(1), n=3)[0]
    assert len(reasons) == 3
    lost = [r["points_lost"] for r in reasons]
    assert lost == sorted(lost, reverse=True)


def test_missing_values_do_not_crash(small_model, small_data):
    X = small_data[config.MODEL_FEATURES].head(5).copy()
    X["monthly_income_tzs"] = np.nan
    X["dti"] = np.nan
    assert len(small_model.predict_proba(X)) == 5


def test_decision_policy_boundaries():
    assert decide(config.APPROVE_PD - 0.001) == "approve"
    assert decide(config.APPROVE_PD) == "refer"
    assert decide(config.DECLINE_PD) == "decline"
