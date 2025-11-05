import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import networkx as nx

try:  # dowhy expects this helper on newer networkx versions
    from networkx.algorithms.d_separation import d_separation

    if not hasattr(nx.algorithms, "d_separated"):
        nx.algorithms.d_separated = lambda g, x, y, z=None: d_separation(g, x, y, z)
except ImportError:  # pragma: no cover
    if not hasattr(nx.algorithms, "d_separated"):
        nx.algorithms.d_separated = lambda g, x, y, z=None: False

from dowhy import CausalModel
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "lalonde.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
CONFIG_PATH = MODEL_DIR / "config.json"
DR_MODEL_PATH = MODEL_DIR / "dr_learner.pkl"
PROPENSITY_MODEL_PATH = MODEL_DIR / "propensity_model.pkl"

TARGET_COLUMN = "re78"
TREATMENT_COLUMN = "treat"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 0.2  # Top 20% uplift bucket


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Dataset already present at {RAW_DATA_PATH}")
        return

    print(f"Downloading Lalonde dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Saved dataset to {RAW_DATA_PATH}")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    # The first column is just an index
    if df.columns[0] == "Unnamed: 0":
        df = df.drop(columns=["Unnamed: 0"])
    df[TREATMENT_COLUMN] = df[TREATMENT_COLUMN].astype(int)
    print(f"Loaded Lalonde dataset with {len(df):,} rows")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    base_features = [
        "age",
        "educ",
        "married",
        "nodegree",
        "re74",
        "re75",
    ]
    categorical_cols = ["race"]

    continuous = df[base_features].astype(float)
    categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
    X = pd.concat([continuous.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    y = df[TARGET_COLUMN].astype(float)
    feature_cols = X.columns.tolist()
    return X, y, feature_cols, categorical.columns.tolist()


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        t,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=t,
    )


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    treat_train: pd.Series,
) -> Tuple[DRLearner, LogisticRegression]:
    propensity_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    propensity_model.fit(X_train, treat_train)

    outcome_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    dr_learner = DRLearner(
        model_regression=outcome_model,
        model_propensity=propensity_model,
    )
    dr_learner.fit(y_train.values, treat_train.values, X=X_train.values)
    return dr_learner, propensity_model


def run_dowhy_ate(X: pd.DataFrame, y: pd.Series, t: pd.Series) -> float:
    try:
        data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True), t.reset_index(drop=True)], axis=1)
        data[TREATMENT_COLUMN] = t.values
        data[TARGET_COLUMN] = y.values

        model = CausalModel(
            data=data,
            treatment=TREATMENT_COLUMN,
            outcome=TARGET_COLUMN,
            common_causes=X.columns.tolist(),
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.propensity_score_stratification",
            method_params={
                "propensity_score_model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            },
        )
        return float(getattr(estimate, "value", estimate))
    except Exception as exc:  # pragma: no cover - fallback when DoWhy not supported
        print(f"DoWhy estimation fallback due to: {exc}")
        return float(y[t == 1].mean() - y[t == 0].mean())


def evaluate(
    dr_learner: DRLearner,
    propensity_model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    treat_test: pd.Series,
    ate_dowhy: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    uplift_scores = dr_learner.effect(X_test.values)
    ate_dr = float(np.mean(uplift_scores))

    treated_mask = treat_test == 1
    control_mask = treat_test == 0
    outcome_treated = float(y_test[treated_mask].mean())
    outcome_control = float(y_test[control_mask].mean())
    empirical_uplift = outcome_treated - outcome_control

    top_k = max(1, int(len(uplift_scores) * TOP_K))
    ranked_indices = np.argsort(uplift_scores)[::-1]
    top_mask = np.zeros_like(uplift_scores, dtype=bool)
    top_mask[ranked_indices[:top_k]] = True
    uplift_top = float(
        y_test[top_mask & treated_mask].mean() - y_test[top_mask & control_mask].mean()
    ) if (top_mask & control_mask).any() else float("nan")

    propensities = propensity_model.predict_proba(X_test)[:, 1]
    auc_propensity = float(roc_auc_score(treat_test, propensities))

    metrics = {
        "ate_dowhy": ate_dowhy,
        "ate_dr": ate_dr,
        "empirical_uplift": empirical_uplift,
        "uplift_top20": uplift_top,
        "outcome_treated_mean": outcome_treated,
        "outcome_control_mean": outcome_control,
        "propensity_auc": auc_propensity,
    }
    return metrics, uplift_scores


def save_artifacts(
    dr_learner: DRLearner,
    propensity_model: LogisticRegression,
    feature_names: List[str],
    categorical_dummy_cols: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    treat_test: pd.Series,
    uplift_scores: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(dr_learner, DR_MODEL_PATH)
    joblib.dump(propensity_model, PROPENSITY_MODEL_PATH)

    with CONFIG_PATH.open("w") as f:
        json.dump(
            {
                "feature_columns": feature_names,
                "treatment_column": TREATMENT_COLUMN,
                "outcome_column": TARGET_COLUMN,
                "top_k_fraction": TOP_K,
                "categorical_dummy_columns": categorical_dummy_cols,
                "original_categorical_columns": ["race"],
            },
            f,
            indent=2,
        )

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    X_test.head(5).to_csv(SAMPLE_INPUT_PATH, index=False)

    predictions_df = X_test.copy()
    predictions_df[TARGET_COLUMN] = y_test.values
    predictions_df[TREATMENT_COLUMN] = treat_test.values
    predictions_df["uplift_score"] = uplift_scores
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    download_dataset()
    df = load_dataset()
    X, y, feature_names, categorical_dummies = prepare_features(df)
    treatment = df[TREATMENT_COLUMN].astype(int)
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split_data(X, y, treatment)
    dr_learner, propensity_model = fit_models(X_train, y_train, treat_train)
    ate_dowhy = run_dowhy_ate(X_train, y_train, treat_train)
    metrics, uplift_scores = evaluate(dr_learner, propensity_model, X_test, y_test, treat_test, ate_dowhy)
    save_artifacts(
        dr_learner,
        propensity_model,
        feature_names,
        categorical_dummies,
        X_test,
        y_test,
        treat_test,
        uplift_scores,
        metrics,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
