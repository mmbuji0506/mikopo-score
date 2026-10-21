"""Synthetic loan-application generator.

HONESTY NOTE (say this out loud in your presentation): real bank data is
confidential, so this project SIMULATES borrowers. We choose the "true" rules that
decide who defaults, then see whether our model can recover them. That is a valid
way to learn and to test a pipeline, but results (e.g. AUC) describe the simulator,
NOT real Tanzanian borrowers. The pipeline is designed so real data with the same
columns can be dropped in later.
"""
import numpy as np
import pandas as pd

from mikopo import config
from mikopo.features import compute_dti


def _sigmoid(x):
    """Turns any number into a probability between 0 and 1: 1 / (1 + e^-x)."""
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(linear_part, target_rate):
    """Find the intercept so the average default probability equals target_rate.

    We do a binary search ("bisection"): try a middle value, see if the average
    probability is too high or too low, and halve the search range each time.
    """
    low, high = -15.0, 5.0
    for _ in range(60):
        mid = (low + high) / 2
        if _sigmoid(linear_part + mid).mean() > target_rate:
            high = mid           # too many defaults -> lower the intercept
        else:
            low = mid            # too few defaults  -> raise the intercept
    return (low + high) / 2


def generate_applications(n=config.N_APPLICANTS, seed=config.RANDOM_SEED):
    """Return a DataFrame with one row per loan application."""
    rng = np.random.default_rng(seed)    # the random-number generator, seeded

    # ---- 1. WHEN did they apply? Uniformly over two years.
    day_offset = rng.integers(0, config.N_DAYS, size=n)
    app_date = pd.Timestamp(config.START_DATE) + pd.to_timedelta(day_offset, unit="D")
    t = day_offset / config.N_DAYS        # 0.0 at the start -> 1.0 at the end (used for drift)

    # ---- 2. WHO are they? (demographics)
    age = np.clip(rng.normal(34, 9, n), 18, 65).astype(int)
    gender = rng.choice(["F", "M"], size=n, p=[0.48, 0.52])
    region = rng.choice(
        ["Dar es Salaam", "Arusha", "Mwanza", "Dodoma", "Mbeya", "Other"],
        size=n, p=[0.30, 0.12, 0.14, 0.10, 0.10, 0.24],
    )
    employment = rng.choice(
        ["salaried", "self_employed", "farmer", "informal_trader"],
        size=n, p=[0.30, 0.30, 0.15, 0.25],
    )

    # ---- 3. HOW MUCH do they earn? (monthly, in TZS). Typical income differs by job.
    median_income = pd.Series(employment).map(
        {"salaried": 900_000, "self_employed": 700_000, "farmer": 350_000, "informal_trader": 400_000}
    ).to_numpy(dtype=float)
    income = median_income * rng.lognormal(mean=0, sigma=0.45, size=n)
    income = income * (1 + 0.04 * t)      # DRIFT: incomes rise ~4% over the two years (inflation)
    income = np.round(income, -3)         # round to the nearest 1,000 TZS

    # ---- 4. THE LOAN they ask for
    loan_amount = np.round(income * rng.uniform(1.0, 9.0, n), -4)       # 1-9 months of income
    loan_term = rng.choice([6, 12, 24, 36], size=n, p=[0.20, 0.40, 0.25, 0.15])

    # ---- 5. CREDIT HISTORY
    existing_loans = np.clip(rng.poisson(0.8, n), 0, 4)
    months_with_bank = np.clip(rng.exponential(30, n), 0, 180).astype(int)
    # People with more loans are more likely to have been late before.
    past_delinq = np.clip(rng.poisson(0.15 + 0.15 * existing_loans), 0, 3)

    # ---- 6. MOBILE-MONEY BEHAVIOUR (the "alternative data" for informal borrowers)
    txn_rate = pd.Series(employment).map(
        {"salaried": 25, "self_employed": 45, "farmer": 15, "informal_trader": 50}
    ).to_numpy(dtype=float)
    mm_txn = rng.poisson(txn_rate * (1 + 0.5 * t))      # DRIFT: mobile-money use grows ~50%
    mm_balance = np.round(income * rng.beta(2, 5, n), -3)

    # ---- 7. The debt-to-income ratio, computed with the SHARED formula.
    dti_true = compute_dti(loan_amount, loan_term, income)

    # ---- 8. THE HIDDEN TRUTH: what really drives default (log-odds scale).
    # Positive number = riskier. This is the "secret recipe" the model must rediscover.
    employment_effect = pd.Series(employment).map(
        {"salaried": -0.5, "self_employed": 0.0, "farmer": 0.4, "informal_trader": 0.3}
    ).to_numpy(dtype=float)
    linear = (
        1.8 * np.minimum(dti_true, 1.5)              # heavy repayment burden -> riskier
        + 0.55 * past_delinq                         # paid late before -> riskier
        + 0.20 * existing_loans                      # other debts -> riskier
        - 0.12 * np.minimum(months_with_bank / 12, 5)  # long bank relationship -> safer
        - 0.012 * np.minimum(mm_txn, 60)             # active mobile-money user -> safer
        + employment_effect
        - 0.02 * (np.minimum(age, 55) - 34)          # very young borrowers slightly riskier
        + rng.normal(0, 0.5, n)                      # luck / things we cannot observe
    )
    intercept = _calibrate_intercept(linear, config.TARGET_DEFAULT_RATE)
    default = (rng.random(n) < _sigmoid(linear + intercept)).astype(int)

    # ---- 9. Assemble the table
    df = pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        "customer_id": [f"C{i:06d}" for i in range(n)],   # 1 application per customer here
        "application_date": app_date,
        "age": age,
        "gender": gender,
        "region": region,
        "employment_type": employment,
        "monthly_income_tzs": income,
        "loan_amount_tzs": loan_amount,
        "loan_term_months": loan_term,
        "existing_loans": existing_loans,
        "months_with_bank": months_with_bank,
        "mobile_money_monthly_txn": mm_txn.astype(float),
        "mobile_money_avg_balance_tzs": mm_balance,
        "past_delinquencies": past_delinq,
        "dti": dti_true,
        config.TARGET: default,
    })

    # ---- 10. Real data is messy: blank out some values on purpose.
    missing_income = rng.random(n) < 0.03       # 3% did not declare income
    df.loc[missing_income, ["monthly_income_tzs", "dti"]] = np.nan   # dti needs income too
    missing_mm = rng.random(n) < 0.08           # 8% have no mobile-money record
    df.loc[missing_mm, ["mobile_money_monthly_txn", "mobile_money_avg_balance_tzs"]] = np.nan

    return df.sort_values("application_date").reset_index(drop=True)


def main():
    """Generate the dataset and save it to data/applications.csv."""
    config.DATA_DIR.mkdir(exist_ok=True)
    df = generate_applications()
    path = config.DATA_DIR / "applications.csv"
    df.to_csv(path, index=False)
    print(f"[data_gen] {len(df):,} applications saved to {path}")
    print(f"[data_gen] overall default rate: {df[config.TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
