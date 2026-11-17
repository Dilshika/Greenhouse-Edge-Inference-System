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
    assert_disjoint_split_rows,
)


TARGETS = ["Temperature", "Humidity", "CO2"]


def build_tinyml_model(
    lookback,
    n_features,
    n_targets,
    hidden_units_1=32,
    hidden_units_2=16,
    learning_rate=0.001,
):
    inputs = tf.keras.Input(
        shape=(lookback, n_features),
        name="sensor_history",
    )

    x = tf.keras.layers.Flatten(
        name="flatten_history"
    )(inputs)

    x = tf.keras.layers.Dense(
        hidden_units_1,
        activation="relu",
        name="dense_1",
    )(x)

    x = tf.keras.layers.Dense(
        hidden_units_2,
        activation="relu",
        name="dense_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        n_targets,
        name="forecast",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="tinyml_mlp",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
    )

    return model


def fit_scalers(
    X_train,
    y_train,
):
    X_mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    X_std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    X_std[X_std < 1e-8] = 1.0

    y_mean = y_train.mean(
        axis=0,
        keepdims=True,
    )

    y_std = y_train.std(
        axis=0,
        keepdims=True,
    )

    y_std[y_std < 1e-8] = 1.0

    return {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def apply_scalers(
    X,
    y,
    scalers,
):
    X_scaled = (
        X - scalers["X_mean"]
    ) / scalers["X_std"]

    y_scaled = (
        y - scalers["y_mean"]
    ) / scalers["y_std"]

    return (
        X_scaled.astype(np.float32),
        y_scaled.astype(np.float32),
    )


def evaluate(
    y_true,
    y_pred,
    persistence,
):
    metrics = {}

    for index, target in enumerate(TARGETS):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        baseline = persistence[:, index]

        model_mse = mean_squared_error(
            actual,
            predicted,
        )

        persistence_mse = mean_squared_error(
            actual,
            baseline,
        )

        metrics[target] = {
            "mae": float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "rmse": float(
                np.sqrt(model_mse)
            ),
            "r2": float(
                r2_score(
                    actual,
                    predicted,
                )
            ),
            "persistence_mae": float(
                mean_absolute_error(
                    actual,
                    baseline,
                )
            ),
            "persistence_rmse": float(
                np.sqrt(persistence_mse)
            ),
            "improvement_over_persistence_mae": float(
                mean_absolute_error(
                    actual,
                    baseline,
                )
                - mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "improvement_over_persistence_rmse": float(
                np.sqrt(persistence_mse)
                - np.sqrt(model_mse)
            ),
        }

    return metrics


def convert_to_tflite(
    model,
    X_train_scaled,
    output_path,
    calibration_samples=200,
):
    """
    Initial deployment export uses dynamic-range
    quantisation.

    This is not full integer quantisation.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    calibration_count = min(
        calibration_samples,
        X_train_scaled.shape[0],
    )

    calibration_data = X_train_scaled[
        :calibration_count
    ].astype(np.float32)

    def representative_dataset():
        for sample in calibration_data:
            yield [
                sample.reshape(
                    1,
                    *sample.shape,
                )
            ]

    # Keep this disabled for the first run.
    # Full integer conversion requires additional
    # compatibility checks and separate evaluation.
    converter.representative_dataset = None
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)

    return {
        "quantisation": "dynamic_range",
        "calibration_samples_used": 0,
        "model_size_bytes": len(tflite_model),
    }


def train_tinyml(
    train_path,
    validation_path,
    test_path,
    output_dir,
    lookback=12,
    horizon=1,
    interval_minutes=5,
    hidden_units_1=32,
    hidden_units_2=16,
    learning_rate=0.001,
    batch_size=64,
    max_epochs=100,
    patience=10,
    random_seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.utils.set_random_seed(
        random_seed
    )

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

    X_train = train_windows.X
    y_train = train_windows.y

    X_validation = validation_windows.X
    y_validation = validation_windows.y

    X_test = test_windows.X
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

    scalers = fit_scalers(
        X_train,
        y_train,
    )

    X_train_scaled, y_train_scaled = apply_scalers(
        X_train,
        y_train,
        scalers,
    )

    X_validation_scaled, y_validation_scaled = apply_scalers(
        X_validation,
        y_validation,
        scalers,
    )

    X_test_scaled, _ = apply_scalers(
        X_test,
        y_test,
        scalers,
    )

    model = build_tinyml_model(
        lookback=lookback,
        n_features=X_train.shape[2],
        n_targets=y_train.shape[1],
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        learning_rate=learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(
            X_validation_scaled,
            y_validation_scaled,
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Predict using the floating-point Keras model.
    y_test_prediction_scaled = model.predict(
        X_test_scaled,
        verbose=0,
    )

    y_test_prediction = (
        y_test_prediction_scaled
        * scalers["y_std"]
        + scalers["y_mean"]
    )

    metrics = evaluate(
        y_test,
        y_test_prediction,
        persistence_test,
    )

    print(
        "\nFloating-point TinyML candidate test metrics:"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    # Save the trained Keras model.
    keras_path = (
        output_dir / "tinyml_mlp_forecast.keras"
    )

    model.save(keras_path)

    # Export an initial TensorFlow Lite artifact.
    tflite_path = (
        output_dir / "tinyml_mlp_forecast.tflite"
    )

    conversion_info = convert_to_tflite(
        model=model,
        X_train_scaled=X_train_scaled,
        output_path=tflite_path,
    )

    # Save preprocessing and run configuration.
    preprocessing_path = (
        output_dir / "tinyml_preprocessing.joblib"
    )

    joblib.dump(
        {
            "model_name": "tinyml_mlp",
            "lookback": lookback,
            "horizon": horizon,
            "interval_minutes": interval_minutes,
            "hidden_units_1": hidden_units_1,
            "hidden_units_2": hidden_units_2,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "random_seed": random_seed,
            "scalers": scalers,
        },
        preprocessing_path,
    )

    # Save test predictions for later comparison.
    prediction_path = (
        output_dir
        / "tinyml_test_predictions.parquet"
    )

    prediction_frame = test_windows.metadata.copy()

    for index, target in enumerate(TARGETS):
        prediction_frame[
            f"{target}_actual"
        ] = y_test[:, index]

        prediction_frame[
            f"{target}_predicted"
        ] = y_test_prediction[:, index]

        prediction_frame[
            f"{target}_persistence"
        ] = persistence_test[:, index]

    prediction_frame.to_parquet(
        prediction_path,
        index=False,
    )

    # Save training history and evaluation metrics.
    history_frame = pd.DataFrame(
        history.history
    )

    history_path = (
        output_dir / "tinyml_training_history.csv"
    )

    history_frame.to_csv(
        history_path,
        index=False,
    )

    metrics_path = (
        output_dir / "tinyml_test_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "tinyml_mlp",
                "lookback": lookback,
                "horizon": horizon,
                "interval_minutes": interval_minutes,
                "hyperparameters": {
                    "hidden_units_1": hidden_units_1,
                    "hidden_units_2": hidden_units_2,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "random_seed": random_seed,
                },
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
                "epochs_completed": int(
                    len(history.history["loss"])
                ),
                "best_validation_loss": float(
                    min(
                        history.history["val_loss"]
                    )
                ),
                "tflite_conversion": conversion_info,
                "floating_point_test_metrics": metrics,
            },
            indent=2,
        )
    )

    print(
        f"\nSaved Keras model: {keras_path}"
    )

    print(
        f"Saved TensorFlow Lite model: {tflite_path}"
    )

    print(
        f"Saved preprocessing: {preprocessing_path}"
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

    parser.add_argument(
        "--hidden-units-1",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-units-2",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    train_tinyml(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        interval_minutes=args.interval_minutes,
        hidden_units_1=args.hidden_units_1,
        hidden_units_2=args.hidden_units_2,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()