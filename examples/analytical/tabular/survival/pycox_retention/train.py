import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.datasets import support
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import integrate

if not hasattr(pd.Series, "is_monotonic"):  # Compatibility for pandas >=2.2
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

if not hasattr(integrate, "simps") and hasattr(integrate, "simpson"):
    integrate.simps = integrate.simpson

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CONFIG_PATH = MODEL_DIR / "config.json"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
WEIGHTS_PATH = MODEL_DIR / "coxph_weights.pt"
BASELINE_HAZARDS_PATH = MODEL_DIR / "baseline_hazards.csv"
BASELINE_CUM_HAZARDS_PATH = MODEL_DIR / "baseline_cum_hazards.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SURVIVAL_CURVES_PATH = ARTIFACTS_DIR / "sample_survival_curves.csv"
SURVIVAL_SUMMARY_PATH = ARTIFACTS_DIR / "survival_horizons.csv"
RISK_PATH = ARTIFACTS_DIR / "risk_predictions.csv"

RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 120
PATIENCE = 15
HORIZON_DAYS = [180, 365, 730]


def load_dataset() -> pd.DataFrame:
    df = support.read_df().reset_index(drop=True)
    print(f"Loaded SUPPORT dataset with {len(df):,} rows")
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    features = df.drop(columns=["duration", "event"])
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    return encoded, features, categorical_cols


def split_indices(events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(events))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=events,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=events[train_idx],
    )
    return train_idx, val_idx, test_idx


def prepare_data(
    encoded: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray], StandardScaler, List[str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    feature_cols = encoded.columns.tolist()
    train_idx, val_idx, test_idx = split_indices(events)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(encoded.iloc[train_idx])
    X_val = scaler.transform(encoded.iloc[val_idx])
    X_test = scaler.transform(encoded.iloc[test_idx])

    tensors = {
        "train": torch.tensor(X_train, dtype=torch.float32),
        "val": torch.tensor(X_val, dtype=torch.float32),
        "test": torch.tensor(X_test, dtype=torch.float32),
    }

    arrays = {
        "train_duration": durations[train_idx],
        "train_event": events[train_idx],
        "val_duration": durations[val_idx],
        "val_event": events[val_idx],
        "test_duration": durations[test_idx],
        "test_event": events[test_idx],
    }

    return tensors, arrays, scaler, feature_cols, (train_idx, val_idx, test_idx)


def build_model(input_dim: int) -> CoxPH:
    net = tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=[128, 64],
        out_features=1,
        batch_norm=True,
        dropout=0.2,
    )
    optimizer = tt.optim.Adam(lr=1e-3)
    model = CoxPH(net, optimizer)
    return model


def train_model(
    tensors: Dict[str, torch.Tensor],
    arrays: Dict[str, np.ndarray],
) -> Tuple[CoxPH, Dict[str, float], pd.DataFrame]:
    y_train = (
        torch.tensor(arrays["train_duration"], dtype=torch.float32),
        torch.tensor(arrays["train_event"], dtype=torch.float32),
    )
    y_val = (
        torch.tensor(arrays["val_duration"], dtype=torch.float32),
        torch.tensor(arrays["val_event"], dtype=torch.float32),
    )

    model = build_model(tensors["train"].shape[1])
    callbacks = [tt.callbacks.EarlyStopping(patience=PATIENCE)]

    history = model.fit(
        tensors["train"],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        val_data=(tensors["val"], y_val),
        verbose=True,
    )

    model.compute_baseline_hazards(tensors["train"], y_train)

    surv_df = model.predict_surv_df(tensors["test"])
    surv_df.index = surv_df.index.astype(float)
    surv_df = surv_df[~surv_df.index.duplicated()].sort_index()
    ev = EvalSurv(
        surv_df,
        arrays["test_duration"],
        arrays["test_event"],
        censor_surv="km",
    )
    time_grid = np.linspace(surv_df.index.min(), surv_df.index.max(), 100)
    history_df = history.to_pandas()

    metrics = {
        "concordance_td": float(ev.concordance_td()),
        "integrated_brier_score": float(ev.integrated_brier_score(time_grid)),
        "integrated_nbll": float(ev.integrated_nbll(time_grid)),
        "epochs_trained": int(len(history_df)),
    }
    return model, metrics, surv_df


def summarize_survival(surv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, column in enumerate(surv_df.columns[:5]):
        for horizon in HORIZON_DAYS:
            horizon_idx = np.searchsorted(surv_df.index.values, horizon, side="left")
            if horizon_idx >= len(surv_df.index):
                horizon_prob = float(surv_df.iloc[-1][column])
            else:
                horizon_prob = float(surv_df.iloc[horizon_idx][column])
            rows.append({
                "sample_column": int(column) if isinstance(column, (int, np.integer)) else idx,
                "horizon_days": horizon,
                "survival_prob": horizon_prob,
            })
    return pd.DataFrame(rows)


def save_artifacts(
    model: CoxPH,
    scaler: StandardScaler,
    feature_cols: List[str],
    categorical_cols: List[str],
    raw_features: pd.DataFrame,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    surv_df: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    model.save_model_weights(str(WEIGHTS_PATH))
    baseline_hazards = model.baseline_hazards_.copy()
    baseline_hazards.index = baseline_hazards.index.astype(float)
    baseline_hazards = baseline_hazards.sort_index()
    baseline_hazards.to_csv(BASELINE_HAZARDS_PATH)

    baseline_cum_hazards = model.baseline_cumulative_hazards_.copy()
    baseline_cum_hazards.index = baseline_cum_hazards.index.astype(float)
    baseline_cum_hazards = baseline_cum_hazards.sort_index()
    baseline_cum_hazards.to_csv(BASELINE_CUM_HAZARDS_PATH)

    with CONFIG_PATH.open("w") as f:
        json.dump(
            {
                "feature_columns": feature_cols,
                "categorical_columns": categorical_cols,
                "network": {
                    "layers": [128, 64],
                    "dropout": 0.2,
                },
                "horizons_days": HORIZON_DAYS,
            },
            f,
            indent=2,
        )

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    _, _, test_idx = indices
    raw_features.iloc[test_idx[:5]].to_csv(SAMPLE_INPUT_PATH, index=False)

    surv_df.iloc[:, :5].to_csv(SURVIVAL_CURVES_PATH)
    summarize_survival(surv_df).to_csv(SURVIVAL_SUMMARY_PATH, index=False)

    risk_scores = model.predict(tensors["test"]).flatten()
    risk_df = pd.DataFrame(
        {
            "duration": arrays["test_duration"],
            "event": arrays["test_event"],
            "predicted_risk": risk_scores,
        }
    )
    risk_df.to_csv(RISK_PATH, index=False)


def main() -> None:
    df = load_dataset()
    encoded, raw_features, categorical_cols = encode_features(df)
    durations = df["duration"].to_numpy(dtype=float)
    events = df["event"].astype(int).to_numpy()

    tensors, arrays, scaler, feature_cols, indices = prepare_data(encoded, durations, events)
    model, metrics, surv_df = train_model(tensors, arrays)
    save_artifacts(
        model,
        scaler,
        feature_cols,
        categorical_cols,
        raw_features,
        indices,
        arrays,
        tensors,
        metrics,
        surv_df,
    )


if __name__ == "__main__":
    main()
