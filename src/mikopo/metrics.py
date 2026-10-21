"""Model-quality metrics used by banks.

Plain-English meaning of each (memorise these for your defence):
  AUC    - if I pick one random defaulter and one random good borrower, how often does
           the model give the defaulter the higher risk? 0.5 = coin flip, 1.0 = perfect.
  Gini   - just a rescaled AUC: 2*AUC - 1. Banks love quoting Gini.
  KS     - the biggest gap between the cumulative share of bads and of goods when
           borrowers are ranked by risk. Higher = better separation.
  PSI    - Population Stability Index: how much a distribution moved between two
           samples. <0.10 stable, 0.10-0.25 watch, >0.25 significant shift.
  Brier  - average squared error of the probabilities (lower is better).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, p_bad):
    """Kolmogorov-Smirnov statistic.

    roc_curve gives, for every threshold, the share of bads caught (tpr) and the
    share of goods wrongly flagged (fpr). KS is the largest vertical gap tpr - fpr.
    """
    fpr, tpr, _ = roc_curve(y_true, p_bad)
    return float(np.max(tpr - fpr))


def gini(y_true, p_bad):
    """Gini coefficient = 2*AUC - 1."""
    return 2 * roc_auc_score(y_true, p_bad) - 1


def summarize(y_true, p_bad):
    """All headline metrics in one dictionary."""
    auc = roc_auc_score(y_true, p_bad)
    return {
        "n": int(len(y_true)),
        "default_rate": float(np.mean(y_true)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, p_bad),
        "brier": float(brier_score_loss(y_true, p_bad)),
    }


def roc_auc_score_safe(y_true, p_bad):
    """AUC, or NaN when the sample contains only one class (AUC is undefined then)."""
    return float(roc_auc_score(y_true, p_bad)) if len(set(y_true)) > 1 else float("nan")


def calibration_table(y_true, p_bad, bins=10):
    """Compare PREDICTED vs OBSERVED default rate in equal-sized risk groups.

    Discrimination (AUC) asks "does the model RANK people correctly?".
    Calibration asks "when it says 12%, do about 12% really default?". A bank needs
    both: calibration drives provisioning and pricing.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_bad)})
    df["group"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    out = df.groupby("group").agg(
        n=("y", "size"), predicted_pd=("p", "mean"), observed_rate=("y", "mean")
    ).reset_index()
    out["gap"] = out["predicted_pd"] - out["observed_rate"]
    return out


def _psi_from_distributions(p_expected, p_actual):
    """Core PSI formula: sum((actual% - expected%) * ln(actual% / expected%))."""
    p_e = np.clip(p_expected, 1e-4, None)    # clip avoids log(0) and division by 0
    p_a = np.clip(p_actual, 1e-4, None)
    return float(np.sum((p_a - p_e) * np.log(p_a / p_e)))


def psi_numeric(expected, actual, bins=10):
    """PSI for a numeric variable. Bin edges come from the EXPECTED (reference) sample.

    Missing values get their own bucket so a jump in missingness is also detected.
    """
    e = pd.Series(expected)
    a = pd.Series(actual)
    # inner quantile edges of the reference sample (10 bins -> 9 edges)
    edges = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, bins + 1)[1:-1]))

    def distribution(s):
        # np.digitize assigns each value to a bin; right=True makes bins (a, b].
        idx = np.digitize(s.dropna().to_numpy(), edges, right=True)
        counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        counts = np.append(counts, s.isna().sum())       # extra bucket: missing
        return counts / max(counts.sum(), 1.0)

    return _psi_from_distributions(distribution(e), distribution(a))


def psi_categorical(expected, actual):
    """PSI for a categorical variable: compare category frequencies."""
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    cats = sorted(set(e.index) | set(a.index))
    p_e = np.array([e.get(c, 0.0) for c in cats])
    p_a = np.array([a.get(c, 0.0) for c in cats])
    return _psi_from_distributions(p_e, p_a)


def psi_status(value, watch=0.10, action=0.25):
    """Traffic-light label for a PSI value."""
    if value < watch:
        return "stable"
    if value < action:
        return "watch"
    return "ACTION"
