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


def build_recurrent_model(
    model_name,
    lookback,
    n_features,
    n_targets,
    recurrent_units=32,
    dense_units=16,
    learning_rate=0.001,
):
    inputs = tf.keras.Input(
        shape=(lookback, n_features),
        name="sensor_history",
    )

    if model_name == "lstm":
        x = tf.keras.layers.LSTM(
            recurrent_units,
            name="lstm_encoder",
        )(inputs)

    elif model_name == "gru":
        x = tf.keras.layers.GRU(
            recurrent_units,
            name="gru_encoder",
        )(inputs)

    else:
        raise ValueError(
            f"Unsupported recurrent model: {model_name}"
        )

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
    )(x)

    outputs = tf.keras.layers.Dense(
        n_targets,
        name="forecast",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
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
    """
    Fit scaling statistics using training windows only.
    """
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


def train_recurrent(
    model_name,
    train_path,
    validation_path,
    test_path,
    output_dir,
    lookback=12,
    horizon=1,
    interval_minutes=5,
    recurrent_units=32,
    dense_units=16,
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

    model = build_recurrent_model(
        model_name=model_name,
        lookback=lookback,
        n_features=X_train.shape[2],
        n_targets=y_train.shape[1],
        recurrent_units=recurrent_units,
        dense_units=dense_units,
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

    # Predictions are scaled; convert them back
    # to original physical units.
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
        f"\n{model_name} test metrics:"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    # Save the trained Keras model.
    model_path = (
        output_dir / f"{model_name}_forecast.keras"
    )

    model.save(model_path)

    # Save preprocessing and run configuration.
    preprocessing_path = (
        output_dir / f"{model_name}_preprocessing.joblib"
    )

    joblib.dump(
        {
            "model_name": model_name,
            "lookback": lookback,
            "horizon": horizon,
            "interval_minutes": interval_minutes,
            "recurrent_units": recurrent_units,
            "dense_units": dense_units,
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
        / f"{model_name}_test_predictions.parquet"
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
        output_dir / f"{model_name}_training_history.csv"
    )

    history_frame.to_csv(
        history_path,
        index=False,
    )

    metrics_path = (
        output_dir / f"{model_name}_test_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "lookback": lookback,
                "horizon": horizon,
                "interval_minutes": interval_minutes,
                "hyperparameters": {
                    "recurrent_units": recurrent_units,
                    "dense_units": dense_units,
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
                "test_metrics": metrics,
            },
            indent=2,
        )
    )

    print(
        f"\nSaved model: {model_path}"
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
        "--model",
        required=True,
        choices=["lstm", "gru"],
    )

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
        "--recurrent-units",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dense-units",
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

    train_recurrent(
        model_name=args.model,
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        interval_minutes=args.interval_minutes,
        recurrent_units=args.recurrent_units,
        dense_units=args.dense_units,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()