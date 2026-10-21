"""Shared test fixtures: a small, fast model so tests run in seconds."""
import pytest

from mikopo import config
from mikopo.data_gen import generate_applications
from mikopo.scorecard import Scorecard


@pytest.fixture(scope="session")
def small_data():
    return generate_applications(n=4000, seed=7)


@pytest.fixture(scope="session")
def small_model(small_data):
    return Scorecard().fit(small_data[config.MODEL_FEATURES], small_data[config.TARGET])
