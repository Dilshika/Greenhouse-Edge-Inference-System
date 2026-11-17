import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from training_data import (
    build_windows,
    load_split,
    assert_disjoint_split_rows,
)


TARGETS = ["Temperature", "Humidity", "CO2"]


def flatten_history(X):
    """
    Convert (n_windows, lookback, n_features)
    to (n_windows, lookback * n_features).
    """
    return X.reshape(
        X.shape[0],
        -1,
    )


def make_model(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )


def evaluate(
    y_true,
    y_pred,
    persistence,
):
    """
    Evaluate one target at a time.

    y_true, y_pred and persistence must each be one-dimensional
    arrays for the same target.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    persistence = np.asarray(persistence).reshape(-1)

    if not (
        y_true.shape[0]
        == y_pred.shape[0]
        == persistence.shape[0]
    ):
        raise ValueError(
            "Actual, predicted and persistence arrays "
            "must have the same length."
        )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    persistence_mse = mean_squared_error(
        y_true,
        persistence,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(mse)
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "persistence_mae": float(
            mean_absolute_error(
                y_true,
                persistence,
            )
        ),
        "persistence_rmse": float(
            np.sqrt(persistence_mse)
        ),
        "improvement_over_persistence_mae": float(
            mean_absolute_error(
                y_true,
                persistence,
            )
            - mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "improvement_over_persistence_rmse": float(
            np.sqrt(persistence_mse)
            - np.sqrt(mse)
        ),
    }


def train_one_target(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    persistence_test,
    target_index,
    target_name,
    hyperparameters,
):
    model = make_model(**hyperparameters)

    model.fit(
        X_train,
        y_train[:, target_index],
        eval_set=[
            (
                X_validation,
                y_validation[:, target_index],
            )
        ],
        verbose=False,
    )

    # The sklearn interface uses the best iteration
    # for prediction when early stopping is enabled.
    prediction = model.predict(X_test)

    metrics = evaluate(
        y_test,
        np.stack(
            [
                prediction
                for _ in range(1)
            ],
            axis=1,
        ),
        persistence_test,
    )

    return model, prediction, metrics[target_name]


def train_xgboost(
    train_path,
    validation_path,
    test_path,
    output_dir,
    lookback=12,
    horizon=1,
    interval_minutes=5,
    hyperparameters=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hyperparameters is None:
        hyperparameters = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    train_df = load_split(train_path)
    validation_df = load_split(validation_path)
    test_df = load_split(test_path)

    assert_disjoint_split_rows(
        train_df,
        validation_df,
        test_df,
    )

    train_windows = build_windows(
        train_df,
        lookback=lookback,
        horizon=horizon,
        interval_minutes=interval_minutes,
    )

    validation_windows = build_windows(
        validation_df,
        lookback=lookback,
        horizon=horizon,
        interval_minutes=interval_minutes,
    )

    test_windows = build_windows(
        test_df,
        lookback=lookback,
        horizon=horizon,
        interval_minutes=interval_minutes,
    )

    X_train = flatten_history(
        train_windows.X
    )
    y_train = train_windows.y

    X_validation = flatten_history(
        validation_windows.X
    )
    y_validation = validation_windows.y

    X_test = flatten_history(
        test_windows.X
    )
    y_test = test_windows.y
    persistence_test = test_windows.persistence

    print(
        "Window counts:",
        {
            "train": X_train.shape[0],
            "validation": X_validation.shape[0],
            "test": X_test.shape[0],
        },
    )

    print(
        "Input shape:",
        X_train.shape,
    )

    models = {}
    predictions = []
    metrics = {}

    for target_index, target in enumerate(TARGETS):
        print(f"\nTraining target: {target}")

        model = make_model(**hyperparameters)

        model.fit(
            X_train,
            y_train[:, target_index],
            eval_set=[
                (
                    X_validation,
                    y_validation[:, target_index],
                )
            ],
            verbose=False,
        )

        prediction = model.predict(X_test)

        models[target] = model

        predictions.append(prediction)

        metrics[target] = evaluate(
            y_test[:, target_index],
            prediction,
            persistence_test[:, target_index],
        )

        print(
            json.dumps(
                metrics[target],
                indent=2,
            )
        )

    predictions = np.stack(
        predictions,
        axis=1,
    )

    # Save the three independently trained models.
    model_path = output_dir / "xgboost_models.joblib"

    joblib.dump(
        {
            "models": models,
            "lookback": lookback,
            "horizon": horizon,
            "interval_minutes": interval_minutes,
            "hyperparameters": hyperparameters,
        },
        model_path,
    )

    prediction_path = (
        output_dir / "xgboost_test_predictions.parquet"
    )

    prediction_frame = test_windows.metadata.copy()

    for target_index, target in enumerate(TARGETS):
        prediction_frame[
            f"{target}_actual"
        ] = y_test[:, target_index]

        prediction_frame[
            f"{target}_predicted"
        ] = predictions[:, target_index]

        prediction_frame[
            f"{target}_persistence"
        ] = persistence_test[:, target_index]

    prediction_frame.to_parquet(
        prediction_path,
        index=False,
    )

    metrics_path = (
        output_dir / "xgboost_test_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "lookback": lookback,
                "horizon": horizon,
                "interval_minutes": interval_minutes,
                "hyperparameters": hyperparameters,
                "window_counts": {
                    "train": int(
                        X_train.shape[0]
                    ),
                    "validation": int(
                        X_validation.shape[0]
                    ),
                    "test": int(
                        X_test.shape[0]
                    ),
                },
                "window_audit": {
                    "train": train_windows.audit,
                    "validation": validation_windows.audit,
                    "test": test_windows.audit,
                },
                "test_metrics": metrics,
            },
            indent=2,
        )
    )

    print(
        f"\nSaved models: {model_path}"
    )

    print(
        f"Saved predictions: {prediction_path}"
    )

    print(
        f"Saved metrics: {metrics_path}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--validation",
        required=True,
    )

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=5,
    )

    args = parser.parse_args()

    train_xgboost(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        interval_minutes=args.interval_minutes,
    )


if __name__ == "__main__":
    main()