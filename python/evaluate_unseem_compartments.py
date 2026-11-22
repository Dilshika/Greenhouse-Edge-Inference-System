import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from training_data import (
    build_windows,
    load_split,
)


TARGETS = ["Temperature", "Humidity", "CO2"]


def evaluate_target(
    y_true,
    y_pred,
    persistence,
):
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

    mse = mean_squared_error(y_true, y_pred)
    persistence_mse = mean_squared_error(
        y_true,
        persistence,
    )

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
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


def evaluate_model(
    model_type,
    model_path,
    preprocessing_path,
    unseen_test_path,
    output_dir,
    lookback=12,
    horizon=1,
    interval_minutes=5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessing = joblib.load(
        preprocessing_path
    )

    scalers = preprocessing["scalers"]

    saved_lookback = preprocessing["lookback"]
    saved_horizon = preprocessing["horizon"]
    saved_interval = preprocessing[
        "interval_minutes"
    ]

    if (
        saved_lookback != lookback
        or saved_horizon != horizon
        or saved_interval != interval_minutes
    ):
        raise ValueError(
            "Evaluation settings do not match the "
            "settings used during training."
        )

    unseen_df = load_split(unseen_test_path)

    unseen_windows = build_windows(
        unseen_df,
        lookback=lookback,
        horizon=horizon,
        interval_minutes=interval_minutes,
    )

    X = unseen_windows.X
    y = unseen_windows.y
    persistence = unseen_windows.persistence

    X_scaled = (
        X - scalers["X_mean"]
    ) / scalers["X_std"]

    X_scaled = X_scaled.astype(np.float32)

    if model_type == "xgboost":
        bundle = joblib.load(model_path)

        models = bundle["models"]

        predictions = []

        for target_index, target in enumerate(
            TARGETS
        ):
            X_flat = X.reshape(
                X.shape[0],
                -1,
            )

            prediction = models[
                target
            ].predict(X_flat)

            predictions.append(prediction)

        y_pred = np.stack(
            predictions,
            axis=1,
        )

    elif model_type in {
        "lstm",
        "gru",
        "tinyml",
    }:
        model = tf.keras.models.load_model(
            model_path
        )

        y_pred_scaled = model.predict(
            X_scaled,
            verbose=0,
        )

        y_pred = (
            y_pred_scaled
            * scalers["y_std"]
            + scalers["y_mean"]
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    metrics = {}

    for index, target in enumerate(TARGETS):
        metrics[target] = evaluate_target(
            y[:, index],
            y_pred[:, index],
            persistence[:, index],
        )

    prediction_frame = unseen_windows.metadata.copy()

    for index, target in enumerate(TARGETS):
        prediction_frame[
            f"{target}_actual"
        ] = y[:, index]

        prediction_frame[
            f"{target}_predicted"
        ] = y_pred[:, index]

        prediction_frame[
            f"{target}_persistence"
        ] = persistence[:, index]

    prediction_path = (
        output_dir
        / f"{model_type}_unseen_compartments_predictions.csv"
    )

    prediction_frame.to_csv(
        prediction_path,
        index=False,
    )

    metrics_path = (
        output_dir
        / f"{model_type}_unseen_compartments_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "evaluation_type": (
                    "unseen_compartments"
                ),
                "unseen_test_path": str(
                    unseen_test_path
                ),
                "lookback": lookback,
                "horizon": horizon,
                "interval_minutes": interval_minutes,
                "window_audit": unseen_windows.audit,
                "window_counts": {
                    "unseen_test": int(
                        X.shape[0]
                    ),
                },
                "metrics": metrics,
            },
            indent=2,
        )
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"\nSaved predictions: {prediction_path}"
    )

    print(
        f"Saved metrics: {metrics_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        required=True,
        choices=[
            "xgboost",
            "lstm",
            "gru",
            "tinyml",
        ],
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--preprocessing-path",
        required=True,
    )

    parser.add_argument(
        "--unseen-test",
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

    arguments = parser.parse_args()

    evaluate_model(
        model_type=arguments.model_type,
        model_path=arguments.model_path,
        preprocessing_path=(
            arguments.preprocessing_path
        ),
        unseen_test_path=arguments.unseen_test,
        output_dir=arguments.output_dir,
        lookback=arguments.lookback,
        horizon=arguments.horizon,
        interval_minutes=(
            arguments.interval_minutes
        ),
    )


if __name__ == "__main__":
    main()