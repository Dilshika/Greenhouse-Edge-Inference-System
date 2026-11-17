from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURES = ["Temperature", "Humidity", "CO2"]
TARGETS = ["Temperature", "Humidity", "CO2"]


@dataclass
class WindowedData:
    X: np.ndarray
    y: np.ndarray
    persistence: np.ndarray
    metadata: pd.DataFrame
    audit: dict


def load_split(csv_path):
    """
    Load one prepared split CSV and validate its structure.

    This function does not impute missing measurements,
    round timestamps, or remove quality-flagged rows.
    """
    df = pd.read_csv(
        csv_path,
        skipinitialspace=True,
        na_values=["", "NaN", "nan", "NAN"],
        keep_default_na=True,
    )

    df.columns = df.columns.str.strip()

    required_columns = {
        "Timestamp",
        "Compartment",
        *FEATURES,
        *TARGETS,
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{csv_path}: missing required columns: "
            f"{missing_columns}"
        )

    if df["Compartment"].isna().any():
        raise ValueError(
            f"{csv_path}: missing compartment labels."
        )

    df["Compartment"] = (
        df["Compartment"]
        .astype(str)
        .str.strip()
    )

    if df["Compartment"].eq("").any():
        raise ValueError(
            f"{csv_path}: empty compartment labels."
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
    )

    if df["Timestamp"].isna().any():
        raise ValueError(
            f"{csv_path}: missing timestamps."
        )

    for column in dict.fromkeys(
        FEATURES + TARGETS
    ):
        values = df[column]

        if values.dtype == object:
            values = values.map(
                lambda value: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
            )

            missing_text = values.map(
                lambda value: (
                    isinstance(value, str)
                    and value.lower() in {"", "nan"}
                )
            )

            values = values.mask(missing_text)

        # Convert to numeric without silently accepting
        # unexpected text.
        numeric_values = pd.to_numeric(
            values,
            errors="raise",
        )

        if np.isinf(
            numeric_values.to_numpy(dtype=float)
        ).any():
            raise ValueError(
                f"{csv_path}: infinity found in {column}."
            )

        df[column] = numeric_values

    duplicate_keys = df.duplicated(
        ["Compartment", "Timestamp"],
        keep=False,
    )

    if duplicate_keys.any():
        examples = df.loc[
            duplicate_keys,
            ["Compartment", "Timestamp"],
        ].head(10)

        raise ValueError(
            f"{csv_path}: duplicate compartment/timestamp "
            "keys:\n"
            + examples.to_string(index=False)
        )

    return df.sort_values(
        ["Compartment", "Timestamp"]
    ).reset_index(drop=True)


def build_windows(
    df,
    lookback=12,
    horizon=1,
    interval_minutes=5,
):
    """
    Build forecasting windows independently within each
    compartment and within the supplied split.

    Input:
        The previous `lookback` observations.

    Target:
        The observation `horizon` steps after the final
        input observation.

    A window is accepted only when:
        1. Every timestamp from input start through target
           is separated by exactly `interval_minutes`.
        2. All input measurements are finite.
        3. All target measurements are finite.

    Missing measurements are retained in the source data.
    Windows containing them are excluded rather than filled
    with zero or interpolated.
    """
    if lookback < 1:
        raise ValueError(
            "lookback must be at least 1."
        )

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive."
        )

    step = pd.Timedelta(
        minutes=interval_minutes
    )

    X_values = []
    y_values = []
    persistence_values = []
    metadata_rows = []

    audit = {
        "candidate_windows": 0,
        "rejected_time_gaps": 0,
        "rejected_missing_measurements": 0,
        "accepted_windows": 0,
    }

    target_feature_positions = [
        FEATURES.index(target)
        for target in TARGETS
    ]

    for compartment, group in df.groupby(
        "Compartment",
        sort=False,
    ):
        group = group.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        feature_values = group[
            FEATURES
        ].to_numpy(dtype=np.float32)

        target_values = group[
            TARGETS
        ].to_numpy(dtype=np.float32)

        timestamps = group["Timestamp"]

        for input_end in range(
            lookback - 1,
            len(group) - horizon,
        ):
            input_start = (
                input_end - lookback + 1
            )

            target_index = (
                input_end + horizon
            )

            audit["candidate_windows"] += 1

            # Validate the complete temporal path:
            # input start through target timestamp.
            time_slice = timestamps.iloc[
                input_start:target_index + 1
            ]

            time_differences = (
                time_slice.diff().iloc[1:]
            )

            if not time_differences.eq(
                step
            ).all():
                audit[
                    "rejected_time_gaps"
                ] += 1
                continue

            X_window = feature_values[
                input_start:input_end + 1
            ]

            y_target = target_values[
                target_index
            ]

            if (
                not np.isfinite(X_window).all()
                or not np.isfinite(
                    y_target
                ).all()
            ):
                audit[
                    "rejected_missing_measurements"
                ] += 1
                continue

            X_values.append(
                X_window.copy()
            )

            y_values.append(
                y_target.copy()
            )

            # Persistence baseline:
            # use the latest observed value for each target.
            persistence_values.append(
                X_window[
                    -1,
                    target_feature_positions,
                ].copy()
            )

            metadata_rows.append({
                "Compartment": compartment,
                "input_start": (
                    timestamps.iloc[input_start]
                ),
                "forecast_origin": (
                    timestamps.iloc[input_end]
                ),
                "target_timestamp": (
                    timestamps.iloc[
                        target_index
                    ]
                ),
            })

            audit["accepted_windows"] += 1

    if not X_values:
        raise ValueError(
            "No eligible windows were produced. "
            f"Audit: {audit}"
        )

    return WindowedData(
        X=np.stack(X_values),
        y=np.stack(y_values),
        persistence=np.stack(
            persistence_values
        ),
        metadata=pd.DataFrame(
            metadata_rows
        ),
        audit=audit,
    )


def assert_disjoint_split_rows(
    train_df,
    validation_df,
    test_df,
):
    """
    Reject exact overlapping compartment/timestamp keys
    between supplied splits.

    This is a basic integrity check. It does not replace
    checking that the split design satisfies the intended
    temporal and held-out-compartment protocol.
    """
    def extract_keys(frame):
        return set(
            frame[
                ["Compartment", "Timestamp"]
            ].itertuples(
                index=False,
                name=None,
            )
        )

    split_keys = {
        "train": extract_keys(train_df),
        "validation": extract_keys(
            validation_df
        ),
        "test": extract_keys(test_df),
    }

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left, right in comparisons:
        overlap = (
            split_keys[left]
            & split_keys[right]
        )

        if overlap:
            raise ValueError(
                f"{left}/{right}: "
                f"{len(overlap)} overlapping "
                "compartment/timestamp keys."
            )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to one prepared training, validation "
            "or test CSV."
        ),
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
        help=(
            "Number of historical observations used as "
            "input."
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help=(
            "Number of five-minute steps ahead to forecast."
        ),
    )

    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=5,
        help=(
            "Expected sampling interval in minutes."
        ),
    )

    arguments = parser.parse_args()

    split_frame = load_split(
        arguments.input
    )

    windows = build_windows(
        split_frame,
        lookback=arguments.lookback,
        horizon=arguments.horizon,
        interval_minutes=(
            arguments.interval_minutes
        ),
    )

    print(
        json.dumps(
            windows.audit,
            indent=2,
        )
    )

    print(
        "X shape:",
        windows.X.shape,
    )

    print(
        "y shape:",
        windows.y.shape,
    )

    print(
        "Persistence shape:",
        windows.persistence.shape,
    )

    print(
        "\nExample windows:"
    )

    print(
        windows.metadata.head().to_string(
            index=False
        )
    )