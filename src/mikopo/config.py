"""Central configuration.

WHY a config file? In a bank, a model is a controlled asset. Every number that
changes the model's behaviour (cut-offs, thresholds, seeds, dates) must live in ONE
visible place so an auditor can read it without hunting through code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Path(__file__) is this file; parents[2] climbs: mikopo -> src -> project root.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"        # generated datasets (CSV + SQLite warehouse)
MODEL_DIR = ROOT / "models"     # trained model files + metrics.json
REPORT_DIR = ROOT / "reports"   # validation report, charts, monitoring output
SQL_DIR = ROOT / "sql"          # database schema

# ---------------------------------------------------------------- data generation
RANDOM_SEED = 42                # fixed seed => anyone re-running gets identical results
N_APPLICANTS = 20_000           # number of synthetic loan applications
START_DATE = "2024-07-01"       # first application date
N_DAYS = 730                    # two years of applications
TARGET_DEFAULT_RATE = 0.12      # overall share of borrowers who default (about 12%)
ANNUAL_INTEREST_RATE = 0.18     # simple flat annual interest used to compute instalments

# Out-of-time (OOT) split: banks validate on LATER data than the model was built on,
# because in real life a model is always used on the future, never on the past.
OOT_CUTOFF = "2026-01-01"       # applications on/after this date form the OOT sample

# ---------------------------------------------------------------- features
TARGET = "default"              # 1 = borrower defaulted ("bad"), 0 = repaid ("good")
NUMERIC_FEATURES = [
    "age",
    "monthly_income_tzs",
    "dti",                       # debt-to-income: monthly instalment / monthly income
    "loan_term_months",
    "existing_loans",
    "months_with_bank",
    "mobile_money_monthly_txn",
    "mobile_money_avg_balance_tzs",
    "past_delinquencies",
]
CATEGORICAL_FEATURES = ["employment_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fairness monitoring only. These are NOT inputs to the model. We keep them so the
# validation team can check whether outcomes differ across groups.
MONITOR_ONLY = ["gender", "region"]

# ---------------------------------------------------------------- scorecard scaling
# Industry convention: "600 points = odds of 50:1 good:bad, and every +20 points
# doubles the odds". PDO = Points to Double the Odds.
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# Features with Information Value below this are too weak to keep in the scorecard.
IV_THRESHOLD = 0.02

# ---------------------------------------------------------------- decision policy
APPROVE_PD = 0.08               # PD below 8%  -> approve
DECLINE_PD = 0.20               # PD of 20%+   -> decline; in between -> refer to a human

# ---------------------------------------------------------------- validation thresholds
# These are THIS project's policy, chosen for teaching. Do not present them as CRDB's.
MIN_AUC_OOT = 0.70              # model must still discriminate on future data
MAX_AUC_DROP = 0.05             # train AUC minus OOT AUC must stay below this
MAX_SCORE_PSI = 0.25            # score distribution shift limit
MAX_CALIBRATION_GAP = 0.03      # |average predicted PD - observed default rate| on OOT data
MIN_IMPACT_RATIO = 0.80         # lowest group approval rate / highest group approval rate
PSI_WATCH = 0.10                # PSI below this: stable
PSI_ACTION = 0.25               # PSI above this: significant shift, investigate
