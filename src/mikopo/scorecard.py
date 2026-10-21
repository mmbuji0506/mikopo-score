"""The champion model: a WoE + logistic-regression SCORECARD.

Why do banks still use scorecards in the age of deep learning?
  1. Explainable: every applicant's score is a sum of points per feature, so you can
     tell a customer exactly why they were declined (regulators require this).
  2. Stable and easy to validate, monitor and document.
  3. Competitive: on tabular credit data it is usually close to gradient boosting.

Pipeline inside this file:
  raw columns --(WoEBinner)--> WoE numbers --(LogisticRegression)--> P(default)
  --(scaling)--> credit score in points, with per-feature points for reason codes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mikopo import config


class WoEBinner:
    """Turns each feature into bins and each bin into a Weight of Evidence (WoE).

    WoE of a bin = ln( share of all GOOD borrowers in the bin / share of all BAD
    borrowers in the bin ).
        WoE > 0  -> bin has proportionally more goods  -> safer
        WoE < 0  -> bin has proportionally more bads   -> riskier
    Replacing raw values with WoE makes every feature linear in the log-odds, which
    is exactly what logistic regression needs, and handles missing values and
    outliers automatically.

    Information Value (IV) = sum over bins of (good% - bad%) * WoE measures how
    predictive a whole feature is. Rule of thumb: <0.02 useless, 0.02-0.1 weak,
    0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspiciously strong (check for leakage).
    """

    def __init__(self, numeric, categorical, n_bins=5, max_discrete=6):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.n_bins = n_bins
        # A numeric column with <= max_discrete distinct values (e.g. 0,1,2,3 loans)
        # is treated like a category: one bin per value.
        self.max_discrete = max_discrete

    # ---------------------------------------------------------------- binning
    def _labels(self, col, series):
        """Assign each value of `series` to a readable bin label (a string)."""
        if self.kind_[col] == "continuous":
            edges = self.edges_[col]
            values = series.to_numpy(dtype=float)
            # np.digitize(right=True): bin i holds edges[i-1] < x <= edges[i]
            idx = np.digitize(np.nan_to_num(values), edges, right=True)
            names = [f"<= {edges[0]:g}"]
            names += [f"({edges[i-1]:g}, {edges[i]:g}]" for i in range(1, len(edges))]
            names += [f"> {edges[-1]:g}"]
            labels = np.array(names, dtype=object)[idx]
            labels[np.isnan(values)] = "MISSING"          # NaN gets its own bin
            return pd.Series(labels, index=series.index, dtype=object)
        # discrete numeric or categorical: the bin label is the value itself
        def to_label(v):
            if pd.isna(v):
                return "MISSING"
            if self.kind_[col] == "discrete":
                return str(int(v))
            return str(v)
        return series.map(to_label).astype(object)

    # ---------------------------------------------------------------- fit
    def fit(self, X, y):
        """Learn bin edges and the WoE of every bin from TRAINING data only."""
        y = pd.Series(np.asarray(y), index=X.index)
        self.kind_, self.edges_, self.woe_, self.tables_, self.iv_ = {}, {}, {}, {}, {}
        total_bad = float((y == 1).sum())
        total_good = float((y == 0).sum())

        for col in self.numeric + self.categorical:
            s = X[col]
            if col in self.categorical:
                self.kind_[col] = "categorical"
            elif s.nunique(dropna=True) <= self.max_discrete:
                self.kind_[col] = "discrete"
            else:
                self.kind_[col] = "continuous"
                # quantile edges give bins of roughly equal size
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                self.edges_[col] = np.unique(np.quantile(s.dropna(), qs))

            labels = self._labels(col, s)
            grp = pd.DataFrame({"bin": labels, "bad": (y == 1).astype(int)}).groupby("bin")["bad"]
            table = pd.DataFrame({"count": grp.size(), "bad": grp.sum()})
            table["good"] = table["count"] - table["bad"]
            k = len(table)
            # +0.5 smoothing: avoids ln(0) when a bin has zero goods or zero bads
            dist_good = (table["good"] + 0.5) / (total_good + 0.5 * k)
            dist_bad = (table["bad"] + 0.5) / (total_bad + 0.5 * k)
            table["bad_rate"] = table["bad"] / table["count"]
            table["woe"] = np.log(dist_good / dist_bad)
            table["iv_part"] = (dist_good - dist_bad) * table["woe"]
            self.tables_[col] = table.reset_index()
            self.woe_[col] = table["woe"].to_dict()
            self.iv_[col] = float(table["iv_part"].sum())
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, X, columns=None):
        """Replace raw values by WoE. A bin never seen in training gets WoE 0 (neutral)."""
        out = {}
        for col in (columns or self.numeric + self.categorical):
            out[col] = self._labels(col, X[col]).map(self.woe_[col]).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)

    def bin_labels(self, X, columns):
        """Return the bin label of every row, used for reason codes."""
        return pd.DataFrame({c: self._labels(c, X[c]) for c in columns}, index=X.index)


class Scorecard:
    """WoE binning + logistic regression + points scaling."""

    def __init__(self, numeric=None, categorical=None, iv_threshold=config.IV_THRESHOLD):
        self.numeric = list(numeric if numeric is not None else config.NUMERIC_FEATURES)
        self.categorical = list(categorical if categorical is not None else config.CATEGORICAL_FEATURES)
        self.iv_threshold = iv_threshold
        # Scaling constants derived from the "600 points at 50:1 odds, PDO 20" convention.
        self.factor = config.PDO / np.log(2)
        self.offset = config.BASE_SCORE - self.factor * np.log(config.BASE_ODDS)

    def fit(self, X, y):
        self.binner_ = WoEBinner(self.numeric, self.categorical).fit(X, y)
        # Keep only features that carry enough information.
        self.features_ = [c for c, iv in self.binner_.iv_.items() if iv >= self.iv_threshold]
        woe = self.binner_.transform(X, self.features_)
        # L2-regularised logistic regression on the WoE inputs (C=1.0 is the default).
        self.model_ = LogisticRegression(C=1.0, max_iter=1000)
        self.model_.fit(woe, y)
        self.coef_ = dict(zip(self.features_, self.model_.coef_[0]))
        self._build_points()
        return self

    def predict_proba(self, X):
        """Probability of default (PD) for each row."""
        woe = self.binner_.transform(X, self.features_)
        return self.model_.predict_proba(woe)[:, 1]

    # ---------------------------------------------------------------- points
    def _build_points(self):
        """Convert each (feature, bin) into score points.

        ln(odds_good) = -(b0 + sum(b_j * woe_j))          (odds = P(good)/P(bad))
        score         = offset + factor * ln(odds_good)
        so a bin contributes  -factor * b_j * woe   points, and we spread the
        constant part (offset - factor*b0) evenly over the features.
        """
        b0 = float(self.model_.intercept_[0])
        n_feat = len(self.features_)
        self.base_points_ = (self.offset - self.factor * b0) / n_feat
        rows = []
        for col in self.features_:
            for _, r in self.binner_.tables_[col].iterrows():
                pts = -self.factor * self.coef_[col] * r["woe"] + self.base_points_
                rows.append({"feature": col, "bin": r["bin"], "bad_rate": r["bad_rate"],
                             "woe": r["woe"], "points": pts})
        self.points_table_ = pd.DataFrame(rows)
        # best possible points per feature, used to measure "how many points were lost"
        self.max_points_ = self.points_table_.groupby("feature")["points"].max().to_dict()

    def points_breakdown(self, X):
        """Points earned by each row on each feature (one column per feature)."""
        woe = self.binner_.transform(X, self.features_)
        pts = pd.DataFrame(index=X.index)
        for col in self.features_:
            pts[col] = -self.factor * self.coef_[col] * woe[col] + self.base_points_
        return pts

    def score(self, X):
        """Total credit score = sum of points over features."""
        return self.points_breakdown(X).sum(axis=1).to_numpy()

    def reason_codes(self, X, n=3):
        """For each row: the n features where it lost the most points vs the best bin.

        These are the "principal reasons" a bank must give a declined customer.
        """
        pts = self.points_breakdown(X)
        labels = self.binner_.bin_labels(X, self.features_)
        reasons = []
        for i in range(len(X)):
            lost = {c: self.max_points_[c] - pts.iloc[i][c] for c in self.features_}
            top = sorted(lost, key=lost.get, reverse=True)[:n]
            reasons.append([
                {"feature": c, "value_bin": labels.iloc[i][c], "points_lost": round(float(lost[c]), 1)}
                for c in top
            ])
        return reasons


def decide(pd_value):
    """Business rule that turns a probability of default into an action."""
    if pd_value < config.APPROVE_PD:
        return "approve"
    if pd_value >= config.DECLINE_PD:
        return "decline"
    return "refer"      # borderline: a human credit officer reviews it
