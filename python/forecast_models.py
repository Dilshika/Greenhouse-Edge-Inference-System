import tensorflow as tf


def build_neural_model(
    model_name,
    lookback,
    n_features,
    n_targets,
    learning_rate=0.001,
):
    inputs = tf.keras.Input(
        shape=(lookback, n_features),
        name="sensor_history",
    )

    if model_name == "lstm":
        x = tf.keras.layers.LSTM(
            32,
            name="lstm_encoder",
        )(inputs)

        x = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(x)

    elif model_name == "gru":
        x = tf.keras.layers.GRU(
            32,
            name="gru_encoder",
        )(inputs)

        x = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(x)

    elif model_name == "tiny_mlp":
        x = tf.keras.layers.Flatten()(inputs)

        x = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(x)

        x = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(x)

    else:
        raise ValueError(
            f"Unsupported neural model: {model_name}"
        )

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