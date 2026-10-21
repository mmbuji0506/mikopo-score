"""Train the champion (scorecard) and the challenger (gradient boosting).

CHAMPION / CHALLENGER is a standard bank practice: the model in production is the
"champion"; a more complex model is trained as a "challenger" to prove (or
disprove) that the extra complexity is worth it.

DATA SPLITS (very important to explain):
  train : 75% of the older data      -> the model learns from this
  test  : 25% of the older data      -> honest check on unseen but similar data
  oot   : the newest 6 months        -> out-of-time check: the real-world test,
                                        because the future never looks exactly like the past
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from mikopo import config, __version__
from mikopo.metrics import summarize
from mikopo.scorecard import Scorecard


def load_and_split(path=None):
    """Read the CSV and attach a `split` column: train / test / oot."""
    path = path or config.DATA_DIR / "applications.csv"
    df = pd.read_csv(path, parse_dates=["application_date"])
    is_oot = df["application_date"] >= pd.Timestamp(config.OOT_CUTOFF)
    dev, oot = df[~is_oot], df[is_oot]
    # stratify keeps the default rate the same in train and test
    train, test = train_test_split(
        dev, test_size=0.25, stratify=dev[config.TARGET], random_state=config.RANDOM_SEED
    )
    out = pd.concat([
        train.assign(split="train"), test.assign(split="test"), oot.assign(split="oot")
    ]).sort_values("application_date").reset_index(drop=True)
    return out


def build_challenger():
    """Gradient boosting: many small decision trees, each correcting the previous ones.

    HistGradientBoosting is scikit-learn's fast implementation of the same idea as
    XGBoost/LightGBM. It handles missing values natively (no imputation needed).
    """
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan), config.CATEGORICAL_FEATURES)],
        remainder="passthrough",          # numeric columns pass through untouched
    )
    gbm = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        categorical_features=[0],         # column 0 is the encoded employment_type
        random_state=config.RANDOM_SEED,
    )
    return Pipeline([("encode", encode), ("gbm", gbm)])


def log_to_mlflow(metrics, params):
    """Record the run in MLflow so experiments are reproducible and comparable.

    Wrapped in try/except: if MLflow is missing or misconfigured, training still works.
    """
    try:
        import os
        # Newer MLflow versions warn that the simple local-folder store is in
        # "maintenance mode". It is perfect for a student project, so we opt in
        # explicitly. A production team would use a database backend instead.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
        mlflow.set_experiment("mikopo-score")
        with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}"):
            mlflow.log_params(params)
            for model_name, splits in metrics.items():
                for split_name, m in splits.items():
                    for k in ("auc", "gini", "ks", "brier"):
                        mlflow.log_metric(f"{model_name}_{split_name}_{k}", m[k])
            mlflow.log_artifact(str(config.MODEL_DIR / "metrics.json"))
        print("[train] run logged to MLflow (view with: mlflow ui --backend-store-uri mlruns)")
    except Exception as exc:                         # noqa: BLE001
        print(f"[train] MLflow logging skipped: {exc}")


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    data = load_and_split()
    data.to_csv(config.DATA_DIR / "model_data.csv", index=False)   # validate/monitor reuse this
    parts = {s: data[data["split"] == s] for s in ("train", "test", "oot")}
    train = parts["train"]

    # ---- fit both models on TRAIN only
    champion = Scorecard().fit(train[config.MODEL_FEATURES], train[config.TARGET])
    challenger = build_challenger().fit(train[config.MODEL_FEATURES], train[config.TARGET])

    # ---- evaluate both on all three samples
    metrics = {"scorecard": {}, "challenger": {}}
    for name, part in parts.items():
        X, y = part[config.MODEL_FEATURES], part[config.TARGET]
        metrics["scorecard"][name] = summarize(y, champion.predict_proba(X))
        metrics["challenger"][name] = summarize(y, challenger.predict_proba(X)[:, 1])

    # ---- save everything
    joblib.dump(champion, config.MODEL_DIR / "scorecard.joblib")
    joblib.dump(challenger, config.MODEL_DIR / "challenger.joblib")
    payload = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features_in_scorecard": champion.features_,
        "information_value": champion.binner_.iv_,
        "metrics": metrics,
    }
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    champion.points_table_.to_csv(config.REPORT_DIR / "scorecard_points.csv", index=False)
    pd.DataFrame({"feature": list(champion.binner_.iv_), "iv": list(champion.binner_.iv_.values())}) \
        .sort_values("iv", ascending=False).to_csv(config.REPORT_DIR / "information_value.csv", index=False)

    # ---- print a short summary
    print("[train] rows -> " + ", ".join(f"{k}: {len(v):,}" for k, v in parts.items()))
    for model_name in metrics:
        for split_name, m in metrics[model_name].items():
            print(f"[train] {model_name:<10} {split_name:<5} AUC={m['auc']:.3f} "
                  f"Gini={m['gini']:.3f} KS={m['ks']:.3f}")
    log_to_mlflow(metrics, {"iv_threshold": config.IV_THRESHOLD, "pdo": config.PDO,
                            "base_score": config.BASE_SCORE, "n_train": len(train)})


if __name__ == "__main__":
    main()
