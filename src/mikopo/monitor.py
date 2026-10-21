"""Ongoing monitoring: is the live model still behaving like the one we validated?

Models DECAY. Customers change, the economy changes, products change. A bank
therefore re-checks every deployed model on a schedule (monthly here). We compare
each new month against the TRAIN sample (the "reference") and raise traffic-light
alerts using PSI, plus compare predicted vs actually observed defaults.

Here the "live data" is the out-of-time sample, replayed month by month. In
production the same code would read the prediction log of the API instead.
"""
import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mikopo import config  # noqa: E402
from mikopo.metrics import psi_categorical, psi_numeric, psi_status, roc_auc_score_safe  # noqa: E402


def monthly_monitoring(model, reference, live):
    """One row per calendar month of live data."""
    ref_scores = model.score(reference[config.MODEL_FEATURES])
    live = live.assign(
        month=live["application_date"].dt.to_period("M").astype(str),
        pd_hat=model.predict_proba(live[config.MODEL_FEATURES]),
        score=model.score(live[config.MODEL_FEATURES]),
    )
    rows = []
    for month, part in live.groupby("month"):
        score_psi = psi_numeric(ref_scores, part["score"])
        rows.append({
            "month": month,
            "applications": len(part),
            "mean_predicted_pd": part["pd_hat"].mean(),
            "observed_default_rate": part[config.TARGET].mean(),
            "auc": roc_auc_score_safe(part[config.TARGET], part["pd_hat"]),
            "score_psi": score_psi,
            "score_status": psi_status(score_psi, config.PSI_WATCH, config.PSI_ACTION),
        })
    return pd.DataFrame(rows)


def feature_monitoring(reference, live):
    """PSI of every model input: reference vs all live data."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        rows.append({"feature": col, "psi": psi_numeric(reference[col], live[col])})
    for col in config.CATEGORICAL_FEATURES:
        rows.append({"feature": col, "psi": psi_categorical(reference[col], live[col])})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(lambda v: psi_status(v, config.PSI_WATCH, config.PSI_ACTION))
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def make_chart(monthly, features):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(monthly["month"], monthly["mean_predicted_pd"], "o-", label="Predicted PD")
    axes[0].plot(monthly["month"], monthly["observed_default_rate"], "s-", label="Observed default rate")
    axes[0].set_title("Predicted vs observed default rate by month")
    axes[0].set_ylabel("Default rate"); axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    colors = features["status"].map({"stable": "tab:green", "watch": "orange", "ACTION": "tab:red"})
    axes[1].barh(features["feature"], features["psi"], color=colors)
    axes[1].axvline(config.PSI_WATCH, color="orange", linestyle="--")
    axes[1].axvline(config.PSI_ACTION, color="red", linestyle="--")
    axes[1].set_title("Input drift (PSI, live vs training)")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.REPORT_DIR / "monitoring.png", dpi=130)
    plt.close()


def main():
    config.REPORT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(config.DATA_DIR / "model_data.csv", parse_dates=["application_date"])
    model = joblib.load(config.MODEL_DIR / "scorecard.joblib")
    reference, live = data[data["split"] == "train"], data[data["split"] == "oot"]

    monthly = monthly_monitoring(model, reference, live)
    features = feature_monitoring(reference, live)
    monthly.to_csv(config.REPORT_DIR / "monitoring_monthly.csv", index=False)
    features.to_csv(config.REPORT_DIR / "monitoring_features.csv", index=False)
    make_chart(monthly, features)

    print(monthly.round(3).to_string(index=False))
    print()
    print(features.round(3).to_string(index=False))
    alerts = features[features["status"] == "ACTION"]
    for _, r in alerts.iterrows():
        print(f"[monitor] ALERT: input '{r['feature']}' has drifted (PSI {r['psi']:.2f}) - investigate.")
    if alerts.empty:
        print("[monitor] no ACTION-level drift detected.")


if __name__ == "__main__":
    main()
