"""
evaluate_unseen.py

Evaluates already-trained greenhouse forecasting models on
unseen-compartment test data.

Current configuration:
- lookback = 12 observations
- horizon = 1 observation
- interval = 5 minutes
- forecast = 5 minutes ahead

This script does NOT train, tune, or modify models.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change these paths to match your project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = Path("unseen_results")

DATASETS = [
    "first_edition",
    "second_edition",
]

OUTER_SPLITS = [
    "outer_01",
    "outer_02",
    "outer_03",
    "outer_04",
    "outer_05",
    "outer_06",
]

INNER_SPLITS = [
    "inner_01",
    "inner_02",
    "inner_03",
]

MODELS = [
    "gru",
    "lstm",
    "xgboost",
    "tinyml",
]

TARGETS = [
    "Temperature",
    "Humidity",
    "CO2"
]

LOOKBACK = 12
HORIZON = 1
INTERVAL_MINUTES = 5

# Column names in your CSV
# Change these if your actual column names are different
TIMESTAMP_COLUMN = "timestamp"
COMPARTMENT_COLUMN = "compartment"
FEATURE_COLUMNS = [
    "Temperature",
    "Humidity",
    "CO2"
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_file(folder: Path, extensions):
    """Find the first existing file with one of the given extensions."""

    for extension in extensions:
        matches = list(folder.glob(f"*{extension}"))

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No file with extensions {extensions} found in {folder}"
    )


def load_model_and_artifacts(model_dir: Path, model_name: str):
    """
    Load a saved forecasting model and preprocessing bundle.

    Expected filenames:
        GRU:    gru_forecast.keras + gru_preprocessing.joblib
        LSTM:   lstm_forecast.keras + lstm_preprocessing.joblib
        XGBoost: xgboost_models.joblib
        TinyML: tinyml_mlp_forecast.keras + tinyml_preprocessing.joblib
    """

    if model_name == "gru":
        model_path = model_dir / "gru_forecast.keras"
        preprocessing_path = model_dir / "gru_preprocessing.joblib"

        model = load_model(model_path)
        preprocessing = joblib.load(preprocessing_path)

    elif model_name == "lstm":
        model_path = model_dir / "lstm_forecast.keras"
        preprocessing_path = model_dir / "lstm_preprocessing.joblib"

        model = load_model(model_path)
        preprocessing = joblib.load(preprocessing_path)

    elif model_name == "xgboost":
        model_path = model_dir / "xgboost_models.joblib"

        model = joblib.load(model_path)
        preprocessing = None

    elif model_name == "tinyml":
        model_path = model_dir / "tinyml_mlp_forecast.keras"
        preprocessing_path = model_dir / "tinyml_preprocessing.joblib"

        model = load_model(model_path)
        preprocessing = joblib.load(preprocessing_path)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, preprocessing

def create_windows(df: pd.DataFrame, lookback: int, horizon: int):
    """
    Create sliding windows, skipping any window with NaN values.

    X shape:
        GRU / LSTM / TinyML: (samples, lookback, features)
        XGBoost: flattened before prediction

    y shape:
        (samples, features)
    """
    values = df[FEATURE_COLUMNS].values

    X = []
    y = []

    for i in range(lookback, len(df) - horizon + 1):
        x_window = values[i - lookback:i]
        y_target = values[i + horizon - 1]

        if np.isnan(x_window).any() or np.isnan(y_target).any():
            continue

        X.append(x_window)
        y.append(y_target)

    return np.array(X), np.array(y)

def predict(model, model_name: str, X: np.ndarray):
    """Generate predictions using the correct input format."""

    if model_name in ["gru", "lstm", "tinyml"]:
        predictions = model.predict(X, verbose=0)

    elif model_name == "xgboost":
        X_flat = X.reshape(X.shape[0], -1)

        if isinstance(model, dict):
            # Unwrap if saved as a bundle dict like {"models": {...}, ...}
            xgb_models = model["models"] if "models" in model and isinstance(model["models"], dict) else model

            if all(target in xgb_models for target in TARGETS):
                predictions = np.column_stack([
                    xgb_models[target].predict(X_flat) for target in TARGETS
                ])
            else:
                raise KeyError(
                    f"Unexpected xgboost_models.joblib structure. Available keys: {list(model.keys())}"
                )
        else:
            predictions = model.predict(X_flat)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return np.array(predictions)


def calculate_metrics(y_true, y_pred, persistence):
    """Calculate MAE, RMSE, R2 and persistence improvement."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    persistence_mae = mean_absolute_error(y_true, persistence)
    persistence_rmse = np.sqrt(mean_squared_error(y_true, persistence))

    improvement_mae = persistence_mae - mae
    improvement_rmse = persistence_rmse - rmse

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Persistence_MAE": persistence_mae,
        "Persistence_RMSE": persistence_rmse,
        "Improvement_MAE": improvement_mae,
        "Improvement_RMSE": improvement_rmse,
    }

def extract_scalers(preprocessing):
    """Extract x_scaler and y_scaler from a scaler object or dict bundle."""
    if preprocessing is None:
        return None, None

    if not isinstance(preprocessing, dict):
        return preprocessing, preprocessing

    # Check common key names first
    x_scaler = (
        preprocessing.get("x_scaler")
        or preprocessing.get("feature_scaler")
        or preprocessing.get("scalers")
    )
    y_scaler = (
        preprocessing.get("y_scaler")
        or preprocessing.get("target_scaler")
        or x_scaler
    )

    # Fallback: pick the first value in the dict that has a .transform method
    if x_scaler is None:
        for val in preprocessing.values():
            if hasattr(val, "transform") and hasattr(val, "inverse_transform"):
                x_scaler = val
                y_scaler = val
                break

    if x_scaler is None:
        raise KeyError(
            f"Could not find a scaler in preprocessing dict. Available keys: {list(preprocessing.keys())}"
        )

    return x_scaler, y_scaler

def scale_inputs(X: np.ndarray, preprocessing) -> np.ndarray:
    """Standardize 3D input array (samples, lookback, features) using saved mean and std."""
    scalers = preprocessing["scalers"] if isinstance(preprocessing, dict) and "scalers" in preprocessing else preprocessing

    X_mean = np.asarray(scalers["X_mean"], dtype=float)
    X_std = np.asarray(scalers["X_std"], dtype=float)

    return (X - X_mean) / X_std


def inverse_scale_predictions(predictions_scaled: np.ndarray, preprocessing) -> np.ndarray:
    """Convert standardized predictions back to original units using saved mean and std."""
    scalers = preprocessing["scalers"] if isinstance(preprocessing, dict) and "scalers" in preprocessing else preprocessing

    y_mean = np.asarray(scalers["y_mean"], dtype=float)
    y_std = np.asarray(scalers["y_std"], dtype=float)

    return predictions_scaled * y_std + y_mean

# ============================================================
# 3. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    dataset: str,
    outer_split: str,
    inner_split: str,
    model_name: str,
    test_unseen: pd.DataFrame,
):
    """Evaluate one saved model on one unseen-compartment test set."""

    model_dir = (
        RESULTS_ROOT
        / dataset
        / outer_split
        / inner_split
        / model_name
    )

    if not model_dir.exists():
        print(f"Skipping missing model folder: {model_dir}")
        return []

    print(f"Evaluating: {model_dir}")

    model, preprocessing = load_model_and_artifacts(
        model_dir,
        model_name
    )

    x_scaler, y_scaler = extract_scalers(preprocessing)

    X, y = create_windows(
        test_unseen,
        LOOKBACK,
        HORIZON
    )

    # Scale inputs using the scaler fitted during training
    if model_name == "xgboost":
        X_scaled = X
    else:
        X_scaled = scale_inputs(X, preprocessing)

    predictions_scaled = predict(
        model,
        model_name,
        X_scaled
    )

    # Convert predictions back to original units
    if model_name == "xgboost":
        predictions = predictions_scaled
    else:
        predictions = inverse_scale_predictions(predictions_scaled, preprocessing)

    # Persistence baseline: previous observed value
    # Fixed: last observed value from the lookback window
    persistence = X[:, -1, :]

    for target_index, target_name in enumerate(TARGETS):
        metrics = calculate_metrics(
            y_true=y[:, target_index],
            y_pred=predictions[:, target_index],
            persistence=persistence[:, target_index],
        )

        result = {
            "Dataset": dataset,
            "Outer_Split": outer_split,
            "Inner_Split": inner_split,
            "Model": model_name.upper(),
            "Target": target_name,
            "Lookback": LOOKBACK,
            "Horizon": HORIZON,
            "Interval_Minutes": INTERVAL_MINUTES,
            **metrics,
        }

        yield result


# ============================================================
# 4. MAIN SCRIPT
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in DATASETS:
        for outer_split in OUTER_SPLITS:
            test_unseen_path = (
                DATA_ROOT
                / dataset
                /"rotating_splits"
                / outer_split
                / "test_unseen.csv"
            )

            if not test_unseen_path.exists():
                print(f"Skipping missing test file: {test_unseen_path}")
                continue

            print(f"\nLoading unseen test data: {test_unseen_path}")
            test_unseen = pd.read_csv(test_unseen_path)

            for inner_split in INNER_SPLITS:
                for model_name in MODELS:
                    results = evaluate_model(
                        dataset=dataset,
                        outer_split=outer_split,
                        inner_split=inner_split,
                        model_name=model_name,
                        test_unseen=test_unseen,
                    )

                    for result in results:
                        all_results.append(result)

    if not all_results:
        print("No results were generated. Check paths and model folders.")
        return

    detailed_df = pd.DataFrame(all_results)

    detailed_path = OUTPUT_ROOT / "unseen_results_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    summary_df = (
        detailed_df
        .groupby(["Dataset", "Model", "Target"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            Persistence_MAE_mean=("Persistence_MAE", "mean"),
            Improvement_MAE_mean=("Improvement_MAE", "mean"),
            Improvement_RMSE_mean=("Improvement_RMSE", "mean"),
        )
        .reset_index()
    )

    summary_path = OUTPUT_ROOT / "unseen_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nEvaluation complete.")
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()