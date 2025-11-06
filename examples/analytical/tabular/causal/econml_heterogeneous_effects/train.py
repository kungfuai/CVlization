import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from econml.dr import LinearDRLearner
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = int(os.environ.get("ECONML_RANDOM_SEED", "42"))
N_SAMPLES = int(os.environ.get("ECONML_SAMPLES", "8000"))
N_FEATURES = int(os.environ.get("ECONML_FEATURES", "12"))

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
MODEL_PATH = MODEL_DIR / "econml_linear_dr.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SAMPLE_OUTPUT_PATH = ARTIFACTS_DIR / "sample_predictions.csv"


def generate_synthetic_data(
    n_samples: int, n_features: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, n_features))
    feature_names = [f"feature_{i}" for i in range(n_features)]

    tau = (
        1.0
        + 0.5 * np.sin(X[:, 0])
        + 0.25 * X[:, 1]
        - 0.2 * X[:, 2] * X[:, 3]
        + 0.1 * X[:, 0] * X[:, 1]
    )

    propensity_linear = 0.4 * X[:, 0] - 0.5 * X[:, 2] + 0.3 * X[:, 4] - 0.25 * X[:, 5]
    propensity = 1 / (1 + np.exp(-propensity_linear))
    T = rng.binomial(1, propensity)

    beta = rng.normal(scale=0.5, size=n_features)
    y0 = X @ beta + rng.normal(scale=0.8, size=n_samples)
    y1 = y0 + tau + rng.normal(scale=0.8, size=n_samples)
    y = np.where(T == 1, y1, y0)

    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, T, tau, y0, y1


def policy_value(tau: np.ndarray, y0: np.ndarray, y1: np.ndarray, treat_mask: np.ndarray) -> float:
    outcomes = np.where(treat_mask, y1, y0)
    return float(outcomes.mean())


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_df, y, T, tau_true, y0, y1 = generate_synthetic_data(
        n_samples=N_SAMPLES, n_features=N_FEATURES, seed=RANDOM_SEED
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        T_train,
        T_test,
        tau_train,
        tau_test,
        y0_train,
        y0_test,
        y1_train,
        y1_test,
    ) = train_test_split(
        X_df,
        y,
        T,
        tau_true,
        y0,
        y1,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    outcome_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LassoCV(cv=3, random_state=RANDOM_SEED, n_jobs=-1)),
        ]
    )
    propensity_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegressionCV(
                    cv=3,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    learner = LinearDRLearner(
        model_propensity=propensity_model,
        model_regression=outcome_model,
        random_state=RANDOM_SEED,
    )

    learner.fit(y_train, T_train, X=X_train)

    tau_pred = learner.effect(X_test)
    tau_lower, tau_upper = learner.effect_interval(X_test, alpha=0.1)

    cate_rmse = float(np.sqrt(np.mean((tau_pred - tau_test) ** 2)))
    ate_pred = float(np.mean(tau_pred))
    ate_true = float(np.mean(tau_test))
    ate_error = ate_pred - ate_true

    policy_rec = tau_pred > 0
    policy_val = policy_value(tau_test, y0_test, y1_test, policy_rec)
    oracle_policy_val = policy_value(tau_test, y0_test, y1_test, tau_test > 0)
    rng = np.random.default_rng(RANDOM_SEED)
    random_policy_val = policy_value(tau_test, y0_test, y1_test, rng.random(len(tau_test)) > 0.5)

    metrics: Dict[str, float] = {
        "ate_pred": ate_pred,
        "ate_true": ate_true,
        "ate_error": ate_error,
        "cate_rmse": cate_rmse,
        "policy_value": policy_val,
        "oracle_policy_value": oracle_policy_val,
        "random_policy_value": random_policy_val,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(learner, MODEL_PATH)

    metadata = {
        "dataset": "synthetic_heterogeneous_treatment",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "feature_columns": X_df.columns.tolist(),
        "random_seed": RANDOM_SEED,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_inputs = X_test.reset_index(drop=True).head(5).copy()
    sample_inputs["treatment"] = T_test[:5]
    sample_inputs["outcome"] = y_test[:5]
    sample_inputs["true_tau"] = tau_test[:5]
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)

    sample_outputs = pd.DataFrame(
        {
            "predicted_tau": tau_pred[:5],
            "tau_lower_p05": tau_lower[:5],
            "tau_upper_p95": tau_upper[:5],
            "recommend_treatment": policy_rec[:5].astype(int),
            "true_tau": tau_test[:5],
        }
    )
    sample_outputs.to_csv(SAMPLE_OUTPUT_PATH, index=False)

    print("ATE (pred, true, error):", ate_pred, ate_true, ate_error)
    print("CATE RMSE:", cate_rmse)
    print("Policy value vs random/oracle:", policy_val, random_policy_val, oracle_policy_val)


if __name__ == "__main__":
    main()
