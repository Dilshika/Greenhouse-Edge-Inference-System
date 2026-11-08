"""
Chronological dataset splitting for greenhouse forecasting.

Outputs:
    development.csv  - first 70% of unique timestamps
    fit.csv          - earlier portion of development
    validation.csv   - later portion of development
    test.csv         - final 30% of unique timestamps
    split_manifest.json

Important:
    - Preserve timestamps when formatting the original dataset.
    - All columns are retained.
    - Missing sensor values are NOT removed or imputed here.
    - Multiple compartments use common timestamp boundaries.
    - Forecast windows must subsequently be assigned by target time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_timestamps(
    series: pd.Series,
    mode: str,
    date_format: str | None = None,
) -> pd.Series:
    """Parse timestamps using an explicitly selected representation."""

    if series.isna().any():
        raise ValueError(
            f"Timestamp column contains {series.isna().sum()} missing values."
        )

    if mode == "excel":
        numeric = pd.to_numeric(series, errors="raise")

        parsed = pd.to_datetime(
            numeric,
            unit="D",
            origin="1899-12-30",
            errors="raise",
        )

        # Remove sub-microsecond floating-point conversion artifacts.
        parsed = parsed.dt.round("us")

    elif mode == "text":
        if pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                "Timestamp column is numeric. Do not parse it as text "
                "without checking its representation. If it contains "
                "Excel serial-day values, use --timestamp-mode excel."
            )

        parsed = pd.to_datetime(
            series,
            format=date_format,
            errors="raise",
        )

    else:
        raise ValueError("Timestamp mode must be 'text' or 'excel'.")

    if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
        raise ValueError(
            "Timestamps could not be represented as one consistent "
            "datetime series. Check mixed time zones or mixed formats."
        )

    return parsed


def describe_partition(
    frame: pd.DataFrame,
    timestamp_col: str,
    compartment_col: str | None,
) -> dict:
    """Summarise a partition for reproducibility."""

    summary = {
        "rows": int(len(frame)),
        "unique_timestamps": int(frame[timestamp_col].nunique()),
        "first_timestamp": frame[timestamp_col].min().isoformat(),
        "last_timestamp": frame[timestamp_col].max().isoformat(),
    }

    if compartment_col:
        counts = (
            frame.groupby(compartment_col, dropna=False)
            .size()
        )

        summary["rows_per_compartment"] = {
            str(compartment): int(count)
            for compartment, count in counts.items()
        }

    return summary


def chronological_split(
    input_path: Path,
    output_dir: Path,
    timestamp_col: str,
    compartment_col: str | None = None,
    development_ratio: float = 0.70,
    validation_fraction: float = 0.20,
    timestamp_mode: str = "text",
    date_format: str | None = None,
    expected_interval_minutes: float | None = None,
) -> dict:
    """
    Split by ordered unique timestamps.

    validation_fraction is the fraction of DEVELOPMENT reserved for
    validation, not the fraction of the whole dataset.

    When multiple compartments exist, the same calendar boundaries are
    applied to all compartments.
    """

    if not 0 < development_ratio < 1:
        raise ValueError("development_ratio must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if (
        expected_interval_minutes is not None
        and expected_interval_minutes <= 0
    ):
        raise ValueError("Expected interval must be positive.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Avoid accidentally treating a multi-compartment file as one series.
    if compartment_col is None and "Compartment" in df.columns:
        compartment_col = "Compartment"

    if compartment_col:
        if compartment_col not in df.columns:
            raise KeyError(
                f"Compartment column '{compartment_col}' not found."
            )

        if df[compartment_col].isna().any():
            raise ValueError(
                "Compartment identifiers contain missing values."
            )

    df[timestamp_col] = parse_timestamps(
        df[timestamp_col],
        mode=timestamp_mode,
        date_format=date_format,
    )

    identity_cols = (
        [compartment_col, timestamp_col]
        if compartment_col
        else [timestamp_col]
    )

    duplicate_mask = df.duplicated(
        subset=identity_cols,
        keep=False,
    )

    if duplicate_mask.any():
        example = df.loc[duplicate_mask, identity_cols].head(10)

        raise ValueError(
            "Duplicate timestamps were found within a series.\n"
            "Resolve them explicitly before splitting.\n"
            f"{example.to_string(index=False)}"
        )

    sort_cols = (
        [timestamp_col, compartment_col]
        if compartment_col
        else [timestamp_col]
    )

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Split unique timestamps so simultaneous observations are never
    # assigned to different partitions.
    ordered_times = pd.Index(
        df[timestamp_col].drop_duplicates().sort_values()
    )

    n_times = len(ordered_times)
    n_development = int(np.floor(n_times * development_ratio))

    n_fit = int(
        np.floor(n_development * (1 - validation_fraction))
    )

    if not 0 < n_fit < n_development < n_times:
        raise ValueError(
            "Too few unique timestamps for non-empty fit, validation "
            "and test partitions."
        )

    validation_start = ordered_times[n_fit]
    test_start = ordered_times[n_development]

    fit = df.loc[
        df[timestamp_col] < validation_start
    ].copy()

    validation = df.loc[
        (df[timestamp_col] >= validation_start)
        & (df[timestamp_col] < test_start)
    ].copy()

    test = df.loc[
        df[timestamp_col] >= test_start
    ].copy()

    development = df.loc[
        df[timestamp_col] < test_start
    ].copy()

    partitions = {
        "development": development,
        "fit": fit,
        "validation": validation,
        "test": test,
    }

    assert fit[timestamp_col].max() < validation[timestamp_col].min()
    assert validation[timestamp_col].max() < test[timestamp_col].min()

    assert len(fit) + len(validation) + len(test) == len(df)
    assert len(development) == len(fit) + len(validation)

    # Audit spacing without dropping rows or filling gaps.
    interval_audit = {}

    if compartment_col:
        groups = df.groupby(compartment_col, sort=False)
    else:
        groups = [("single_series", df)]

    for compartment, group in groups:
        times = group[timestamp_col].sort_values()
        differences = times.diff().dropna()

        audit = {
            "rows": int(len(group)),
            "median_interval_minutes": (
                float(differences.dt.total_seconds().median() / 60)
                if not differences.empty
                else None
            ),
        }

        if expected_interval_minutes is not None:
            expected = pd.Timedelta(
                minutes=expected_interval_minutes
            )

            mismatch_count = int(
                (differences != expected).sum()
            )

            audit["intervals_not_matching_expected"] = mismatch_count

            if mismatch_count:
                print(
                    f"WARNING: Series '{compartment}' contains "
                    f"{mismatch_count} intervals not matching "
                    f"{expected_interval_minutes} minutes. "
                    "Do not construct forecast windows across these "
                    "intervals without an explicit gap-handling policy."
                )

        interval_audit[str(compartment)] = audit

    manifest = {
        "input_file": str(input_path.resolve()),
        "timestamp_column": timestamp_col,
        "compartment_column": compartment_col,
        "timestamp_mode": timestamp_mode,
        "date_format": date_format,
        "split_basis": (
            "Ordered unique timestamps; common calendar boundaries "
            "across compartments"
        ),
        "development_ratio_requested": development_ratio,
        "validation_fraction_of_development": validation_fraction,
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "expected_interval_minutes": expected_interval_minutes,
        "interval_audit": interval_audit,
        "partitions": {
            name: describe_partition(
                frame,
                timestamp_col,
                compartment_col,
            )
            for name, frame in partitions.items()
        },
        "notes": [
            "All original columns are retained.",
            "Missing sensor values are not dropped or imputed.",
            "No scaling or synthetic noise is applied.",
            "Window partitioning must respect future target timestamps.",
            "File creation does not establish regular time spacing.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in partitions.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    manifest_path = output_dir / "split_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nChronological split completed")
    print(f"Input: {input_path}")
    print(f"First timestamp: {ordered_times[0]}")
    print(f"Last timestamp:  {ordered_times[-1]}")
    print(f"Validation starts: {validation_start}")
    print(f"Test starts:       {test_start}")

    for name in ["fit", "validation", "test"]:
        frame = partitions[name]

        timestamp_percentage = (
            100 * frame[timestamp_col].nunique() / n_times
        )

        print(
            f"{name:12s}: "
            f"{len(frame):,} rows | "
            f"{timestamp_percentage:.2f}% of unique timestamps"
        )

    if compartment_col:
        print("\nRows per compartment:")
        print(
            pd.concat(
                {
                    name: frame.groupby(compartment_col).size()
                    for name, frame in partitions.items()
                    if name != "development"
                },
                axis=1,
            )
            .fillna(0)
            .astype(int)
        )

    print(f"\nSaved files to: {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Chronological greenhouse dataset splitting."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-col", required=True)
    parser.add_argument("--compartment-col", default=None)

    parser.add_argument(
        "--timestamp-mode",
        choices=["text", "excel"],
        default="text",
    )

    parser.add_argument(
        "--date-format",
        default=None,
        help="Optional explicit text format, e.g. '%%Y-%%m-%%d %%H:%%M:%%S'.",
    )

    parser.add_argument(
        "--development-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of the development block used for validation.",
    )

    parser.add_argument(
        "--expected-interval-minutes",
        type=float,
        default=None,
        help="Optional verified sampling interval for gap auditing.",
    )

    args = parser.parse_args()

    chronological_split(
        input_path=args.input,
        output_dir=args.output_dir,
        timestamp_col=args.timestamp_col,
        compartment_col=args.compartment_col,
        development_ratio=args.development_ratio,
        validation_fraction=args.validation_fraction,
        timestamp_mode=args.timestamp_mode,
        date_format=args.date_format,
        expected_interval_minutes=args.expected_interval_minutes,
    )


if __name__ == "__main__":
    main()