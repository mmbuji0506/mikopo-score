"""Feature formulas shared by training AND the live API.

WHY a separate file? The #1 silent killer of deployed models is "training-serving
skew": the training code computes a feature one way and the production code computes
it slightly differently. Putting the formula in ONE function that both sides import
makes that bug impossible.
"""
import numpy as np

from mikopo.config import ANNUAL_INTEREST_RATE


def monthly_payment(loan_amount, term_months):
    """Monthly instalment under a simple flat-interest loan.

    total repayable = principal * (1 + annual_rate * years)
    instalment      = total repayable / number of months
    Works on single numbers AND on whole pandas columns (vectorised).
    """
    years = np.asarray(term_months, dtype=float) / 12.0
    total = np.asarray(loan_amount, dtype=float) * (1.0 + ANNUAL_INTEREST_RATE * years)
    return total / np.asarray(term_months, dtype=float)


def compute_dti(loan_amount, term_months, monthly_income):
    """Debt-to-income ratio = instalment / income.

    If income is missing (NaN) the result is NaN too - the model treats "unknown
    income" as its own category instead of guessing a number.
    """
    income = np.asarray(monthly_income, dtype=float)
    # np.errstate silences the harmless warning when income is 0 or NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        dti = monthly_payment(loan_amount, term_months) / income
    # Income of zero or below makes no business sense: treat it as missing.
    return np.where(income > 0, dti, np.nan)
