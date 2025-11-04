import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gbt import train as gbt_train

DATA_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/house.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "ames_housing.csv"
ARTIFACTS_DIR = Path("artifacts")
TRAIN_ARTIFACT_DIR = ARTIFACTS_DIR / "training"
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
SCENARIO_PATH = ARTIFACTS_DIR / "scenario_analysis.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
CALIBRATOR_PATH = MODEL_DIR / "isotonic_calibrator.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Dataset already present at {RAW_DATA_PATH}")
        return

    print(f"Downloading housing dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Saved dataset to {RAW_DATA_PATH}")


def load_and_preprocess() -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Standardize missing indicators
    df = df.replace("NA", np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    return df, categorical_cols, numeric_cols


def compute_metrics(y_true: np.ndarray, y_raw: np.ndarray, y_cal: np.ndarray) -> Dict[str, float]:
    rmse_raw = float(np.sqrt(mean_squared_error(y_true, y_raw)))
    rmse_cal = float(np.sqrt(mean_squared_error(y_true, y_cal)))
    mae_cal = float(mean_absolute_error(y_true, y_cal))
    r2_cal = float(r2_score(y_true, y_cal))
    return {
        "rmse_raw": rmse_raw,
        "rmse_calibrated": rmse_cal,
        "mae": mae_cal,
        "r2": r2_cal,
    }


def scenario_analysis(
    base_row: pd.Series,
    model,
    calibrator: IsotonicRegression,
) -> pd.DataFrame:
    scenarios = []

    def predict(row: pd.Series) -> Tuple[float, float]:
        raw = float(model.predict(row.to_frame().T)[0])
        calibrated = float(calibrator.predict([raw])[0]) if calibrator else raw
        return raw, calibrated

    base_raw, base_cal = predict(base_row)
    scenarios.append({
        "scenario": "Base listing",
        "price_raw": base_raw,
        "price_calibrated": base_cal,
    })

    # Scenario 1: Upgrade overall quality by one level (max 10)
    upgraded = base_row.copy()
    if "OverallQual" in upgraded:
        upgraded["OverallQual"] = min(10, upgraded["OverallQual"] + 1)
    raw, cal = predict(upgraded)
    scenarios.append({
        "scenario": "Upgrade finishing quality",
        "price_raw": raw,
        "price_calibrated": cal,
    })

    # Scenario 2: Add 200 sq ft of living area
    larger = base_row.copy()
    if "GrLivArea" in larger:
        larger["GrLivArea"] = larger["GrLivArea"] + 200
    raw, cal = predict(larger)
    scenarios.append({
        "scenario": "Add 200 sq ft living area",
        "price_raw": raw,
        "price_calibrated": cal,
    })

    # Scenario 3: Finish 300 sq ft of basement space
    finished = base_row.copy()
    if "TotalBsmtSF" in finished:
        finished["TotalBsmtSF"] = finished["TotalBsmtSF"] + 300
    if "BsmtFinSF1" in finished:
        finished["BsmtFinSF1"] = finished["BsmtFinSF1"] + 300
    raw, cal = predict(finished)
    scenarios.append({
        "scenario": "Finish 300 sq ft basement",
        "price_raw": raw,
        "price_calibrated": cal,
    })

    # Scenario 4: Add an extra full bathroom if possible
    bath = base_row.copy()
    if "FullBath" in bath:
        bath["FullBath"] = bath["FullBath"] + 1
    raw, cal = predict(bath)
    scenarios.append({
        "scenario": "Add full bathroom",
        "price_raw": raw,
        "price_calibrated": cal,
    })

    return pd.DataFrame(scenarios)


def main() -> None:
    download_dataset()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, categorical_cols, numeric_cols = load_and_preprocess()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=1337,
    )
    base_train_df, calib_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=1337,
    )

    model = gbt_train(
        df=base_train_df,
        df_test=calib_df,
        model_lib="l2",
        label_column=TARGET_COLUMN,
        categorical_feature_columns=categorical_cols,
        numerical_feature_columns=numeric_cols,
        val_size=0.15,
        log_dir=str(TRAIN_ARTIFACT_DIR),
        num_boost_round=500,
        early_stopping_rounds=40,
    )

    model.save(str(MODEL_DIR))
    print(f"Saved model artifacts to {MODEL_DIR}")

    # Fit isotonic calibrator on hold-out calibration split
    y_calib = calib_df[TARGET_COLUMN].to_numpy()
    y_calib_pred = model.predict(calib_df)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_calib_pred, y_calib)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Saved calibration model to {CALIBRATOR_PATH}")

    # Evaluate on test set
    y_test = test_df[TARGET_COLUMN].to_numpy()
    y_test_raw = model.predict(test_df)
    y_test_cal = calibrator.predict(y_test_raw)

    metrics = compute_metrics(y_test, y_test_raw, y_test_cal)
    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    # Save predictions for inspection
    predictions_df = test_df.copy()
    predictions_df["prediction_raw"] = y_test_raw
    predictions_df["prediction_calibrated"] = y_test_cal
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Test predictions saved to {PREDICTIONS_PATH}")

    # Feature importances
    booster = model.booster
    gains = booster.feature_importance(importance_type="gain")
    features = booster.feature_name()
    if len(gains) and len(features):
        fi_df = (
            pd.DataFrame({"feature": features, "importance_gain": gains})
            .sort_values("importance_gain", ascending=False)
        )
        fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        print(f"Feature importances saved to {FEATURE_IMPORTANCE_PATH}")

    # Scenario analysis on a representative listing
    feature_cols = [c for c in test_df.columns if c != TARGET_COLUMN]
    sample_inputs = test_df[feature_cols].head(5).copy()
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    base_row = sample_inputs.iloc[0].copy()
    scenario_df = scenario_analysis(base_row, model, calibrator)
    scenario_df.to_csv(SCENARIO_PATH, index=False)
    print(f"Scenario analysis saved to {SCENARIO_PATH}")

    # Persist config for inference scripts
    config = {
        "categorical_features": categorical_cols,
        "numerical_features": numeric_cols,
        "target": TARGET_COLUMN,
        "calibration": "isotonic",
        "scenario_rows": len(scenario_df),
    }
    with CONFIG_PATH.open("w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
