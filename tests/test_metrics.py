import numpy as np

from mikopo.metrics import gini, ks_statistic, psi_numeric, psi_status


def test_perfect_model_has_ks_and_gini_of_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])    # every bad scores above every good
    assert ks_statistic(y, p) == 1.0
    assert gini(y, p) == 1.0


def test_random_model_has_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    p = rng.random(20000)
    assert abs(gini(y, p)) < 0.03


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(1).normal(size=5000)
    assert psi_numeric(x, x) < 1e-9


def test_psi_detects_a_big_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert psi_numeric(base, shifted) > 0.25
    assert psi_status(psi_numeric(base, shifted)) == "ACTION"


def test_psi_counts_missing_values_as_a_bucket():
    rng = np.random.default_rng(3)
    base = rng.normal(size=4000)
    with_nans = base.copy()
    with_nans[:1500] = np.nan                        # 37% suddenly missing
    assert psi_numeric(base, with_nans) > 0.25
