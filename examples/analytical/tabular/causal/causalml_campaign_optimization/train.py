import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

RANDOM_SEED = int(os.environ.get("CAMPAIGN_RANDOM_SEED", "42"))
N_SAMPLES = int(os.environ.get("CAMPAIGN_SAMPLES", "10000"))
N_FEATURES = int(os.environ.get("CAMPAIGN_FEATURES", "12"))
TREATMENTS = ["email", "display", "social"]

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SAMPLE_PREDICTIONS_PATH = ARTIFACTS_DIR / "sample_predictions.csv"
MODEL_PATH = MODEL_DIR / "uplift_models.joblib"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_campaign_data(
    n_samples: int, n_features: int, treatments: List[str], seed: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n_samples, n_features))
    beta = rng.normal(scale=0.5, size=n_features)
    baseline_logit = X @ beta
    control_prob = sigmoid(baseline_logit)

    uplift_thetas = {
        treat: rng.normal(scale=0.7, size=n_features) for treat in treatments
    }
    uplift_bias = {
        treat: rng.normal(scale=0.1) for treat in treatments
    }
    true_uplift = {}
    for treat in treatments:
        logit = baseline_logit + uplift_bias[treat] + X @ uplift_thetas[treat]
        treatment_prob = sigmoid(logit)
        true_uplift[treat] = treatment_prob - control_prob

    assign_scores = np.column_stack(
        [baseline_logit] + [baseline_logit + 0.5 * (X @ uplift_thetas[treat]) for treat in treatments]
    )
    assign_softmax = np.exp(assign_scores - assign_scores.max(axis=1, keepdims=True))
    assign_prob = assign_softmax / assign_softmax.sum(axis=1, keepdims=True)
    cumulative = assign_prob.cumsum(axis=1)
    random_draws = rng.uniform(size=n_samples)
    # Convert the row-wise cumulative probabilities into sampled assignments without loops.
    assignments = (random_draws[:, None] > cumulative).sum(axis=1)

    treatment_labels = np.array(["control"] + treatments)[assignments]

    outcome_prob = control_prob.copy()
    for idx, treat in enumerate(treatments, start=1):
        mask = assignments == idx
        outcome_prob[mask] = control_prob[mask] + true_uplift[treat][mask]
    outcome_prob = np.clip(outcome_prob, 1e-4, 1 - 1e-4)
    outcomes = rng.binomial(1, outcome_prob)

    feature_columns = {f"feature_{i}": X[:, i] for i in range(n_features)}
    df = pd.DataFrame(feature_columns)
    df["treatment"] = treatment_labels
    df["outcome"] = outcomes

    for treat in treatments:
        df[f"true_uplift_{treat}"] = true_uplift[treat]

    return df, outcome_prob, control_prob, true_uplift


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    treat_train: np.ndarray,
    treatments: List[str],
    control_name: str = "control",
) -> Dict[str, BaseXRegressor]:
    models: Dict[str, BaseXRegressor] = {}
    for treat in treatments:
        mask = (treat_train == control_name) | (treat_train == treat)
        base_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
        )
        learner = BaseXRegressor(learner=base_model, control_name=0)
        subset_treat = treat_train[mask]
        unique_levels = set(subset_treat.tolist())
        if control_name not in unique_levels:
            print(f"Skipping treatment {treat} due to missing control observations")
            continue
        print(f"Fitting uplift model for {treat} with groups: {unique_levels}, dtype: {subset_treat.dtype}")
        X_subset = X_train.loc[mask]
        y_subset = y_train[mask]
        treat_subset = pd.Series((subset_treat == treat).astype(int), index=X_subset.index)
        if treat_subset.nunique() < 2:
            print(f"Skipping treatment {treat} due to insufficient positive samples")
            continue
        learner.fit(
            X_subset,
            y_subset,
            treat_subset,
            p=None,
        )
        models[treat] = learner
    return models


def predict_uplift(
    models: Dict[str, BaseXRegressor], X: pd.DataFrame, treatments: List[str]
) -> Dict[str, np.ndarray]:
    uplift = {}
    for treat, model in models.items():
        uplift[treat] = np.asarray(model.predict(X)).reshape(-1)
    return uplift


def evaluate_models(
    uplift_pred: Dict[str, np.ndarray],
    true_uplift: Dict[str, np.ndarray],
    treatments: List[str],
) -> Dict[str, float]:
    metrics = {}
    for treat in treatments:
        rmse = np.sqrt(np.mean((true_uplift[treat] - uplift_pred[treat]) ** 2))
        metrics[f"rmse_{treat}"] = float(rmse)
    return metrics


def evaluate_policy(
    uplift_pred: Dict[str, np.ndarray],
    true_uplift: Dict[str, np.ndarray],
) -> Dict[str, float]:
    treatments = list(uplift_pred.keys())
    stacked = np.column_stack([uplift_pred[treat] for treat in treatments])
    best_idx = stacked.argmax(axis=1)

    stacked_true = np.column_stack([true_uplift[treat] for treat in treatments])
    optimal_true = stacked_true[np.arange(len(best_idx)), best_idx]
    random_gain = stacked_true.mean(axis=1)
    oracle_gain = stacked_true.max(axis=1)

    policy_metrics = {
        "policy_gain_mean": float(optimal_true.mean()),
        "policy_gain_std": float(optimal_true.std()),
        "random_policy_gain": float(random_gain.mean()),
        "oracle_policy_gain": float(oracle_gain.mean()),
    }
    return policy_metrics


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, outcome_prob, control_prob, true_uplift = generate_campaign_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        treatments=TREATMENTS,
        seed=RANDOM_SEED,
    )

    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
        df[feature_cols],
        df["outcome"].values,
        df["treatment"].values,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["treatment"].values,
    )

    true_uplift_test = {t: df.loc[X_test.index, f"true_uplift_{t}"].values for t in TREATMENTS}

    models = train_models(X_train, y_train, treat_train, TREATMENTS)
    joblib.dump(models, MODEL_PATH)

    uplift_pred = predict_uplift(models, X_test, TREATMENTS)
    metrics = evaluate_models(uplift_pred, true_uplift_test, TREATMENTS)
    metrics.update(evaluate_policy(uplift_pred, true_uplift_test))
    metrics["samples_train"] = int(X_train.shape[0])
    metrics["samples_test"] = int(X_test.shape[0])

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "treatments": ["control"] + TREATMENTS,
        "feature_columns": feature_cols,
        "random_seed": RANDOM_SEED,
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    df[["treatment", "outcome"] + feature_cols].head(5).to_csv(
        SAMPLE_INPUT_PATH, index=False
    )

    sample_predictions = pd.DataFrame(
        {
            "treatment_recommended": [
                TREATMENTS[idx]
                for idx in np.column_stack([uplift_pred[t] for t in TREATMENTS]).argmax(axis=1)[:5]
            ]
        }
    )
    for treat in TREATMENTS:
        sample_predictions[f"pred_uplift_{treat}"] = uplift_pred[treat][:5]
        sample_predictions[f"true_uplift_{treat}"] = true_uplift_test[treat][:5]
    sample_predictions.to_csv(SAMPLE_PREDICTIONS_PATH, index=False)

    print("Saved metrics to", METRICS_PATH)
    print("Policy gain mean:", metrics["policy_gain_mean"])


if __name__ == "__main__":
    main()
