"""Independent model validation: produces reports/validation_report.md + charts.

In a bank, the team that BUILDS a model (first line) must not be the team that
APPROVES it. A separate Model Risk team re-tests it and writes a report that says
"fit for use", "fit with conditions" or "not fit". This script plays that role: it
only loads the saved model and the saved data splits, and re-computes everything.

The Model Risk Analyst job at CRDB is, in essence, writing documents like this one.
"""
import json
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                 # draw charts to files without needing a screen
import matplotlib.pyplot as plt       # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from mikopo import config  # noqa: E402
from mikopo.metrics import (  # noqa: E402
    calibration_table, psi_categorical, psi_numeric, psi_status, summarize,
)
from mikopo.scorecard import decide  # noqa: E402


def md_table(df, floatfmt=3):
    """Render a DataFrame as a Markdown table (tiny helper, no extra dependency)."""
    def fmt(v):
        return f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.astype(object).iterrows():   # astype(object) keeps ints as ints
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def fairness_table(oot, group_col, p_bad):
    """Approval rate and default-rate-of-approved for each group (e.g. each gender).

    Impact ratio = lowest approval rate / highest approval rate. Values well below
    1.0 mean one group is approved much less often and need an explanation.
    """
    tmp = oot.assign(decision=[decide(p) for p in p_bad])
    rows = []
    for group, part in tmp.groupby(group_col):
        approved = part[part["decision"] == "approve"]
        rows.append({
            group_col: group,
            "applications": len(part),
            "approval_rate": (part["decision"] == "approve").mean(),
            "default_rate_of_approved": approved[config.TARGET].mean() if len(approved) else np.nan,
        })
    out = pd.DataFrame(rows)
    ratio = out["approval_rate"].min() / out["approval_rate"].max()
    return out, float(ratio)


def make_charts(oot, p_champ, p_chall, score_oot, feat_psi, calib):
    """Save four PNG charts used in the report."""
    out = config.REPORT_DIR
    y = oot[config.TARGET].to_numpy()

    # 1) ROC curves: the closer to the top-left corner, the better.
    plt.figure(figsize=(5, 4.5))
    for name, p in (("Scorecard (champion)", p_champ), ("Gradient boosting (challenger)", p_chall)):
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False positive rate (good customers flagged)")
    plt.ylabel("True positive rate (defaulters caught)")
    plt.title("ROC curve - out-of-time sample")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "roc.png", dpi=130); plt.close()

    # 2) Calibration: points should sit on the diagonal.
    plt.figure(figsize=(5, 4.5))
    plt.plot(calib["predicted_pd"], calib["observed_rate"], "o-", label="Scorecard")
    top = max(calib["predicted_pd"].max(), calib["observed_rate"].max()) * 1.05
    plt.plot([0, top], [0, top], "k--", label="Perfect calibration")
    plt.xlabel("Predicted default rate"); plt.ylabel("Observed default rate")
    plt.title("Calibration by risk decile - out-of-time")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "calibration.png", dpi=130); plt.close()

    # 3) Score distributions of goods vs bads: less overlap = better separation.
    plt.figure(figsize=(5.5, 4))
    plt.hist(score_oot[y == 0], bins=30, alpha=0.6, density=True, label="Repaid (good)")
    plt.hist(score_oot[y == 1], bins=30, alpha=0.6, density=True, label="Defaulted (bad)")
    plt.xlabel("Credit score (points)"); plt.ylabel("Density")
    plt.title("Score distribution - out-of-time"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "score_distribution.png", dpi=130); plt.close()

    # 4) Feature PSI with the traffic-light thresholds.
    plt.figure(figsize=(6, 4))
    ordered = feat_psi.sort_values("psi")
    plt.barh(ordered["feature"], ordered["psi"])
    plt.axvline(config.PSI_WATCH, color="orange", linestyle="--", label=f"watch ({config.PSI_WATCH})")
    plt.axvline(config.PSI_ACTION, color="red", linestyle="--", label=f"action ({config.PSI_ACTION})")
    plt.xlabel("PSI (train vs out-of-time)"); plt.title("Feature stability")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "feature_psi.png", dpi=130); plt.close()


def main():
    config.REPORT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(config.DATA_DIR / "model_data.csv", parse_dates=["application_date"])
    champion = joblib.load(config.MODEL_DIR / "scorecard.joblib")
    challenger = joblib.load(config.MODEL_DIR / "challenger.joblib")
    info = json.loads((config.MODEL_DIR / "metrics.json").read_text())

    parts = {s: data[data["split"] == s] for s in ("train", "test", "oot")}
    train, oot = parts["train"], parts["oot"]
    X_oot = oot[config.MODEL_FEATURES]

    # ---- recompute predictions ourselves (independent re-performance)
    p_champ = champion.predict_proba(X_oot)
    p_chall = challenger.predict_proba(X_oot)[:, 1]
    score_train = champion.score(train[config.MODEL_FEATURES])
    score_oot = champion.score(X_oot)

    # ---- 1. data summary
    data_summary = pd.DataFrame([{
        "sample": name, "rows": len(part), "default_rate": part[config.TARGET].mean(),
        "from": part["application_date"].min().date(), "to": part["application_date"].max().date(),
    } for name, part in parts.items()])

    # ---- 2. discriminatory power, recomputed for every model and sample
    perf_rows = []
    for model_name, model, is_sc in (("scorecard", champion, True), ("challenger", challenger, False)):
        for split_name, part in parts.items():
            p = model.predict_proba(part[config.MODEL_FEATURES])
            p = p if is_sc else p[:, 1]
            m = summarize(part[config.TARGET], p)
            perf_rows.append({"model": model_name, "sample": split_name, "AUC": m["auc"],
                              "Gini": m["gini"], "KS": m["ks"], "Brier": m["brier"]})
    perf = pd.DataFrame(perf_rows)
    auc = lambda model, split: float(perf[(perf.model == model) & (perf["sample"] == split)]["AUC"].iloc[0])  # noqa: E731

    # ---- 3. calibration on OOT
    calib = calibration_table(oot[config.TARGET], p_champ)
    mean_pred, mean_obs = float(p_champ.mean()), float(oot[config.TARGET].mean())
    calib_gap = abs(mean_pred - mean_obs)

    # ---- 4. stability
    score_psi = psi_numeric(score_train, score_oot)
    feat_rows = []
    for col in config.NUMERIC_FEATURES:
        feat_rows.append({"feature": col, "psi": psi_numeric(train[col], oot[col])})
    for col in config.CATEGORICAL_FEATURES:
        feat_rows.append({"feature": col, "psi": psi_categorical(train[col], oot[col])})
    feat_psi = pd.DataFrame(feat_rows)
    feat_psi["status"] = feat_psi["psi"].map(lambda v: psi_status(v, config.PSI_WATCH, config.PSI_ACTION))

    # ---- 5. fairness (gender and region are NOT model inputs - monitoring only)
    fair_gender, ratio_gender = fairness_table(oot, "gender", p_champ)
    fair_region, ratio_region = fairness_table(oot, "region", p_champ)

    # ---- 6. policy checks and verdict
    checks = [
        (f"OOT AUC >= {config.MIN_AUC_OOT}", auc("scorecard", "oot"), auc("scorecard", "oot") >= config.MIN_AUC_OOT),
        (f"AUC drop train->OOT <= {config.MAX_AUC_DROP}", auc("scorecard", "train") - auc("scorecard", "oot"),
         auc("scorecard", "train") - auc("scorecard", "oot") <= config.MAX_AUC_DROP),
        (f"Score PSI < {config.MAX_SCORE_PSI}", score_psi, score_psi < config.MAX_SCORE_PSI),
        (f"abs(mean predicted PD - observed rate) <= {config.MAX_CALIBRATION_GAP}", calib_gap, calib_gap <= config.MAX_CALIBRATION_GAP),
        (f"Gender impact ratio >= {config.MIN_IMPACT_RATIO}", ratio_gender, ratio_gender >= config.MIN_IMPACT_RATIO),
        (f"Region impact ratio >= {config.MIN_IMPACT_RATIO}", ratio_region, ratio_region >= config.MIN_IMPACT_RATIO),
    ]
    check_df = pd.DataFrame([{"check": c, "value": v, "result": "PASS" if ok else "FAIL"} for c, v, ok in checks])
    all_pass = all(ok for _, _, ok in checks)
    # A feature can drift a lot while the final score still looks stable. Validators
    # flag that as a CONDITION: "approved, but fix/monitor this".
    drifted = feat_psi.loc[feat_psi["status"] == "ACTION", "feature"].tolist()
    if not all_pass:
        verdict = "NOT APPROVED - failed checks must be resolved"
    elif drifted:
        verdict = "PASSED WITH CONDITIONS (synthetic data - not for real lending)"
    else:
        verdict = "ALL CHECKS PASSED (synthetic data - not for real lending)"
    conditions = ""
    if drifted:
        conditions = (f"**Condition:** significant input drift in `{', '.join(drifted)}`. The overall score is "
                      "stable today, but this input has shifted a lot. Review it, monitor it monthly, and "
                      "refresh the model if the drift persists or spreads to the score.\n")

    # Which is better on future data? Simplicity wins ties.
    champ_oot, chall_oot = auc("scorecard", "oot"), auc("challenger", "oot")
    chall_gap = auc("challenger", "train") - chall_oot
    if chall_oot > champ_oot + 0.01:
        challenger_note = (f"The challenger beats the scorecard on OOT AUC ({chall_oot:.3f} vs {champ_oot:.3f}). "
                           "Consider promoting it, but only with an explainability layer such as SHAP.")
    else:
        challenger_note = (f"The challenger does not beat the scorecard on OOT AUC ({chall_oot:.3f} vs {champ_oot:.3f}) "
                           f"and its AUC falls {chall_gap:.3f} from train to OOT (the scorecard's AUC moves by "
                           f"{champ_oot - auc('scorecard', 'train'):+.3f}). The extra complexity is not justified, "
                           "so the explainable scorecard stays champion.")

    make_charts(oot, p_champ, p_chall, score_oot, feat_psi, calib)

    # ---- write the Markdown report
    report = f"""# Independent Model Validation Report - MikopoScore

*Generated {datetime.now():%Y-%m-%d %H:%M} | model version {info['version']} | trained {info['trained_at']}*

## Verdict: **{verdict}**

{md_table(check_df)}

{conditions}
> Thresholds are this project's own teaching policy (see `config.py`), not any bank's.

## 1. Purpose and scope
The model estimates the **probability of default (PD)** of a loan applicant at application
time. Output is a credit score (600 points = 50:1 good:bad odds, +{config.PDO} points doubles the odds)
and a decision: approve if PD < {config.APPROVE_PD:.0%}, decline if PD >= {config.DECLINE_PD:.0%}, otherwise refer to a human.
Model type: Weight-of-Evidence logistic-regression scorecard. Challenger: gradient boosting.

## 2. Data
{md_table(data_summary)}

**Important:** the data is SYNTHETIC. All performance numbers describe a simulator, not real borrowers.

## 3. Discriminatory power
{md_table(perf)}

![ROC](roc.png)
![Score distribution](score_distribution.png)

**Champion vs challenger:** {challenger_note}

## 4. Calibration (out-of-time)
Average predicted PD **{mean_pred:.3f}** vs observed default rate **{mean_obs:.3f}** (gap {calib_gap:.3f}).

{md_table(calib)}

![Calibration](calibration.png)

## 5. Stability (train vs out-of-time)
Score PSI = **{score_psi:.3f}** ({psi_status(score_psi, config.PSI_WATCH, config.PSI_ACTION)}).

{md_table(feat_psi)}

![Feature PSI](feature_psi.png)

## 6. Fairness screening (gender and region are NOT model inputs)
**Gender** (impact ratio {ratio_gender:.2f})

{md_table(fair_gender)}

**Region** (impact ratio {ratio_region:.2f})

{md_table(fair_region)}

*Caveat:* in this synthetic data gender and region are independent of risk by construction, so
passing is expected. It proves the CHECK works; it does not prove real data would be fair.
Removing a protected attribute from the model does not guarantee fairness, because other
variables can act as proxies. That is why outcomes are tested.

## 7. Limitations and recommendations
1. **Synthetic data.** Re-validate on real data before any real use.
2. **Short history.** The out-of-time sample covers only six months; defaults often take longer to appear.
3. **Reject inference.** Real bank data only contains customers who were approved. This ignores that bias.
4. **Drift.** Income and mobile-money activity grow over time in this dataset; see section 5. Monitor PSI monthly.
5. **Recalibration.** If the calibration gap widens, recalibrate the intercept before rebuilding.
6. **Governance.** Real use needs model documentation, ownership, a change log and an annual review.
"""
    (config.REPORT_DIR / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"[validate] verdict: {verdict}")
    print(check_df.to_string(index=False))
    print(f"[validate] report written to {config.REPORT_DIR / 'validation_report.md'}")


if __name__ == "__main__":
    main()
