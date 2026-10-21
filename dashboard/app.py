"""MikopoScore dashboard (Streamlit): a visual front end for the whole project.

Run:  streamlit run dashboard/app.py
Needs the artifacts created by:  python -m mikopo.pipeline

Five tabs: score one applicant, model performance, drift monitoring,
scorecard points (explainability) and the data warehouse.
Streamlit re-runs this whole script from top to bottom every time you touch a widget.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from mikopo import config
from mikopo.features import compute_dti
from mikopo.scorecard import decide

st.set_page_config(page_title="MikopoScore", page_icon=":bank:", layout="wide")
st.title("MikopoScore - credit-risk scoring dashboard")
st.caption("Synthetic data demo. Scores describe a simulator, not real borrowers.")


@st.cache_resource            # load the model once, not on every widget click
def load_model():
    return joblib.load(config.MODEL_DIR / "scorecard.joblib")


def read_csv(name):
    path = config.REPORT_DIR / name
    return pd.read_csv(path) if path.exists() else None


if not (config.MODEL_DIR / "scorecard.joblib").exists():
    st.error("No trained model found. Run `python -m mikopo.pipeline` first, then reload this page.")
    st.stop()

model = load_model()
tab_score, tab_perf, tab_monitor, tab_points, tab_wh = st.tabs(
    ["Score an applicant", "Model performance", "Monitoring", "Scorecard points", "Data warehouse"])

# ------------------------------------------------------------------ 1. score one applicant
with tab_score:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Applicant")
        age = st.slider("Age", 18, 65, 30)
        employment = st.selectbox("Employment type", ["salaried", "self_employed", "farmer", "informal_trader"])
        income_known = st.checkbox("Income declared", value=True)
        income = st.number_input("Monthly income (TZS)", 0, 20_000_000, 600_000, step=50_000, disabled=not income_known)
        loan = st.number_input("Loan amount (TZS)", 100_000, 100_000_000, 2_000_000, step=100_000)
        term = st.selectbox("Loan term (months)", [6, 12, 24, 36], index=1)
        existing = st.slider("Existing loans", 0, 4, 1)
        tenure = st.number_input("Months with the bank", 0, 240, 12)
        mm_known = st.checkbox("Has mobile-money record", value=True)
        mm_txn = st.number_input("Mobile-money transactions per month", 0, 500, 30, disabled=not mm_known)
        mm_bal = st.number_input("Average mobile-money balance (TZS)", 0, 20_000_000, 100_000, step=10_000, disabled=not mm_known)
        delinq = st.slider("Past delinquencies", 0, 3, 0)

    income_val = float(income) if income_known else np.nan
    row = {
        "age": age, "employment_type": employment, "monthly_income_tzs": income_val,
        "loan_term_months": term, "existing_loans": existing, "months_with_bank": tenure,
        "mobile_money_monthly_txn": float(mm_txn) if mm_known else np.nan,
        "mobile_money_avg_balance_tzs": float(mm_bal) if mm_known else np.nan,
        "past_delinquencies": delinq,
        # SAME formula as training and the API (no training-serving skew)
        "dti": float(compute_dti(loan, term, income_val)),
    }
    frame = pd.DataFrame([row])[config.MODEL_FEATURES]
    p = float(model.predict_proba(frame)[0])
    score = float(model.score(frame)[0])
    decision = decide(p)

    with right:
        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability of default", f"{p:.1%}")
        c2.metric("Credit score", f"{score:.0f}")
        c3.metric("Decision", decision.upper())
        message = {"approve": st.success, "refer": st.warning, "decline": st.error}[decision]
        message(f"Policy: approve if PD < {config.APPROVE_PD:.0%}, decline if PD >= {config.DECLINE_PD:.0%}, "
                "otherwise refer to a credit officer.")
        if not np.isnan(row["dti"]):
            st.write(f"Monthly instalment is **{row['dti']:.0%}** of income (debt-to-income).")
        st.markdown("**Top reasons points were lost**")
        reasons = pd.DataFrame(model.reason_codes(frame, n=3)[0])
        st.dataframe(reasons, hide_index=True, width="stretch")
        st.markdown("**Points earned per feature** (higher = safer)")
        st.bar_chart(model.points_breakdown(frame).iloc[0].sort_values())

# ------------------------------------------------------------------ 2. model performance
with tab_perf:
    info_path = config.MODEL_DIR / "metrics.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        rows = [{"model": m, "sample": s, **{k: round(v, 3) for k, v in vals.items() if k in ("auc", "gini", "ks", "brier")}}
                for m, d in info["metrics"].items() for s, vals in d.items()]
        st.subheader("Champion (scorecard) vs challenger (gradient boosting)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Out-of-time (oot) = the newest months, the honest test of how the model behaves on the future.")
    cols = st.columns(3)
    for col, (img, caption) in zip(cols, [("roc.png", "ROC curve"), ("calibration.png", "Calibration"),
                                          ("score_distribution.png", "Score distribution")]):
        if (config.REPORT_DIR / img).exists():
            col.image(str(config.REPORT_DIR / img), caption=caption)
    report = config.REPORT_DIR / "validation_report.md"
    if report.exists():
        with st.expander("Read the full validation report"):
            st.markdown(report.read_text(encoding="utf-8").replace("![", "<!-- ").replace(".png)", ".png -->"),
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ 3. monitoring
with tab_monitor:
    monthly, feats = read_csv("monitoring_monthly.csv"), read_csv("monitoring_features.csv")
    if monthly is None or feats is None:
        st.info("Monitoring output not found. Run `python -m mikopo.monitor`.")
    else:
        st.subheader("Predicted vs observed default rate by month")
        st.line_chart(monthly.set_index("month")[["mean_predicted_pd", "observed_default_rate"]])
        st.subheader("Input drift (PSI, live vs training)")
        st.bar_chart(feats.set_index("feature")["psi"])
        st.caption(f"PSI below {config.PSI_WATCH} = stable, {config.PSI_WATCH}-{config.PSI_ACTION} = watch, "
                   f"above {config.PSI_ACTION} = act.")
        for _, r in feats[feats["status"] == "ACTION"].iterrows():
            st.error(f"ALERT: `{r['feature']}` has drifted (PSI {r['psi']:.2f}). Investigate and consider refitting.")
        st.dataframe(feats.round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4. scorecard points
with tab_points:
    st.subheader("How the scorecard awards points")
    pts = model.points_table_
    iv = pd.Series(model.binner_.iv_).sort_values(ascending=False)
    chosen = st.selectbox("Feature", list(iv.index.intersection(pts["feature"].unique())))
    sub = pts[pts["feature"] == chosen][["bin", "bad_rate", "woe", "points"]].round(3)
    st.bar_chart(sub.set_index("bin")["points"])
    st.dataframe(sub, hide_index=True, width="stretch")
    st.caption("Bad rate = share of borrowers in the bin who defaulted. WoE > 0 means safer than average.")
    st.subheader("Information value (predictive strength) per feature")
    st.bar_chart(iv)

# ------------------------------------------------------------------ 5. data warehouse
with tab_wh:
    st.subheader("Data-quality checks")
    dq = read_csv("data_quality.csv")
    if dq is not None:
        st.dataframe(dq, hide_index=True, width="stretch")
    st.subheader("Business question answered with SQL joins on the star schema")
    q = read_csv("example_query.csv")
    if q is not None:
        st.dataframe(q, hide_index=True, width="stretch")
        st.caption("Defaults by employment type and loan-term band: fact table joined to two dimensions.")
