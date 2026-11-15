"""Rotating compartment holdouts with inner chronological validation.
Python >=3.10. Dependencies: pandas, numpy, scikit-learn.
Input: one long-format CSV per edition containing timestamp and compartment.
All measurement/quality columns are preserved; no windows, imputation or scaling.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import TimeSeriesSplit


def build_splits(input_csv, output_dir, timestamp_col="Timestamp",
                 compartment_col="Compartment", interval_minutes=5,
                 development_ratio=0.7, inner_splits=3, gap_steps=0,
                 timestamp_mode="text", date_format=None,
                 expected_compartments=6):
    source, out = Path(input_csv), Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty. Choose a fresh directory.")
    if not 0 < development_ratio < 1:
        raise ValueError("development_ratio must be between zero and one.")
    if interval_minutes <= 0 or gap_steps < 0 or inner_splits < 2:
        raise ValueError("Require positive interval, nonnegative gap and >=2 inner splits.")
    df = pd.read_csv(source, dtype={compartment_col: "string"})
    for col in (timestamp_col, compartment_col):
        if col not in df:
            raise KeyError(f"Missing {col!r}; available columns: {list(df.columns)}")
        if df[col].isna().any():
            raise ValueError(f"Missing values in {col!r}.")
    df[compartment_col] = df[compartment_col].str.strip()
    if df[compartment_col].eq("").any():
        raise ValueError("Blank compartment identifiers.")
    if timestamp_mode == "excel":
        df[timestamp_col] = pd.to_datetime(pd.to_numeric(df[timestamp_col], errors="raise"),
            unit="D", origin="1899-12-30", errors="raise").dt.round("us")
    else:
        if pd.api.types.is_numeric_dtype(df[timestamp_col]):
            raise ValueError("Numeric timestamps: verify representation; use excel mode only if appropriate.")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format, errors="raise")
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise ValueError("Mixed/unsupported timestamp formats or time zones.")
    if df.duplicated([compartment_col, timestamp_col]).any():
        raise ValueError("Duplicate timestamps within a compartment; resolve before splitting.")
    df = df.sort_values([timestamp_col, compartment_col]).reset_index(drop=True)
    compartments = sorted(df[compartment_col].unique().tolist())
    if len(compartments) < 2:
        raise ValueError("At least two compartments required.")
    if expected_compartments and len(compartments) != expected_compartments:
        raise ValueError(f"Expected {expected_compartments} compartments, found {len(compartments)}: {compartments}")
    step = pd.Timedelta(minutes=interval_minutes)
    start = df[timestamp_col].min()
    end = df[timestamp_col].max()

    # Work directly with timedeltas to avoid integer-unit mismatches.
    offsets = df[timestamp_col] - start
    remainders = offsets % step

    off_grid = remainders.ne(pd.Timedelta(0))

    if off_grid.any():
        diagnostic = df.loc[
            off_grid,
            [compartment_col, timestamp_col],
        ].copy()

        # Distance to the nearest timestamp on the proposed grid.
        remainder_seconds = remainders.loc[
            off_grid
        ].dt.total_seconds()

        step_seconds = step.total_seconds()

        diagnostic["distance_to_grid_seconds"] = np.minimum(
            remainder_seconds,
            step_seconds - remainder_seconds,
        ).to_numpy()

        print("\nTimestamp grid validation failed.")
        print(f"Requested interval: {interval_minutes} minutes")
        print(f"Grid origin: {start}")
        print(f"Off-grid rows: {int(off_grid.sum()):,}")

        print("\nExamples:")
        print(diagnostic.head(20).to_string(index=False))

        print("\nMost common observed intervals per compartment:")

        for compartment, group in df.groupby(
            compartment_col,
            sort=False,
        ):
            differences = (
                group[timestamp_col]
                .sort_values()
                .diff()
                .dropna()
                .dt.total_seconds()
            )

            print(f"\nCompartment: {compartment}")
            print(
                differences
                .value_counts()
                .head(8)
                .rename_axis("interval_seconds")
                .to_string()
            )

        raise ValueError(
            "Timestamps do not match the requested grid. "
            "Inspect the diagnostics before changing the interval "
            "or applying timestamp corrections."
        )
    grid = pd.date_range(start=start, end=end, freq=step)
    n_dev = int(np.floor(len(grid) * development_ratio))
    if n_dev < 3 or n_dev >= len(grid):
        raise ValueError("Insufficient timestamps for development/test.")
    test_start = grid[n_dev]
    dev_grid = grid[:n_dev]
    cv = TimeSeriesSplit(n_splits=inner_splits, gap=gap_steps)
    inner_indices = list(cv.split(dev_grid))
    dev = df[df[timestamp_col] < test_start].copy()
    test = df[df[timestamp_col] >= test_start].copy()
    summary = []
    audit = []
    for c in compartments:
        rows = df[df[compartment_col] == c]
        times = rows[timestamp_col].sort_values()
        audit.append({"compartment": c, "rows": len(rows),
            "first_timestamp": times.min().isoformat(), "last_timestamp": times.max().isoformat(),
            "nonstandard_intervals": int(times.diff().dropna().ne(step).sum()),
            "development_rows": int((times < test_start).sum()),
            "test_rows": int((times >= test_start).sum())})
        if not (times < test_start).any() or not (times >= test_start).any():
            raise ValueError(f"Compartment {c} has no observations in development or test. Review coverage.")
    payloads = []
    manifests = []
    def register(relative, frame, outer, held, role, inner=None):
        if frame.empty:
            raise ValueError(f"Empty partition: {relative}. Review common coverage and inner split settings.")
        payloads.append((relative, frame))
        summary.append({"outer_fold": outer, "held_out_compartment": held,
            "inner_fold": inner, "role": role, "file": relative,
            "rows": len(frame), "compartments": "|".join(sorted(frame[compartment_col].unique())),
            "first_timestamp": frame[timestamp_col].min().isoformat(),
            "last_timestamp": frame[timestamp_col].max().isoformat()})
    for k, held in enumerate(compartments, 1):
        folder = f"outer_{k:02d}"
        allowed = dev[dev[compartment_col] != held].copy()
        unseen = test[test[compartment_col] == held].copy()
        seen = test[test[compartment_col] != held].copy()
        assert held not in allowed[compartment_col].unique()
        assert allowed[timestamp_col].max() < test[timestamp_col].min()
        register(f"{folder}/development.csv", allowed, k, held, "development")
        register(f"{folder}/test_all.csv", test, k, held, "test_all")
        register(f"{folder}/test_unseen.csv", unseen, k, held, "test_unseen")
        register(f"{folder}/test_seen.csv", seen, k, held, "test_seen")
        inner_metadata = []
        for j, (train_idx, val_idx) in enumerate(inner_indices, 1):
            fit_start, fit_end = dev_grid[train_idx[0]], dev_grid[train_idx[-1]]
            val_start, val_end = dev_grid[val_idx[0]], dev_grid[val_idx[-1]]
            fit = allowed[allowed[timestamp_col].between(fit_start, fit_end)].copy()
            val = allowed[allowed[timestamp_col].between(val_start, val_end)].copy()
            for c in compartments:
                if c != held and (not fit[compartment_col].eq(c).any() or not val[compartment_col].eq(c).any()):
                    raise ValueError(f"Outer {k}, inner {j}: compartment {c} absent from fit or validation.")
            assert fit[timestamp_col].max() < val[timestamp_col].min()
            register(f"{folder}/inner_{j:02d}/fit.csv", fit, k, held, "fit", j)
            register(f"{folder}/inner_{j:02d}/validation.csv", val, k, held, "validation", j)
            inner_metadata.append({"inner_fold": j, "fit_start": fit_start.isoformat(),
                "fit_end": fit_end.isoformat(), "validation_start": val_start.isoformat(),
                "validation_end": val_end.isoformat()})
        manifests.append((f"{folder}/manifest.json", {"outer_fold": k,
            "held_out_compartment": held, "development_compartments": [c for c in compartments if c != held],
            "test_start": test_start.isoformat(), "inner_splits": inner_metadata}))
    # Validate all partitions before writing files.
    out.mkdir(parents=True, exist_ok=True)
    for relative, frame in payloads:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    for relative, manifest in manifests:
        (out / relative).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out / "split_summary.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "coverage_audit.csv", index=False)
    metadata = {"input": str(source.resolve()), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "timestamp_column": timestamp_col, "compartment_column": compartment_col,
        "compartments": compartments, "outer_folds": len(compartments),
        "development_fraction_of_calendar_grid": development_ratio,
        "test_start": test_start.isoformat(), "interval_minutes": interval_minutes,
        "inner_splits": inner_splits, "gap_steps": gap_steps,
        "notes": ["Raw observation partitions only: no model-ready windows are generated.",
        "All measurements and existing quality flags retained; no imputation, scaling or synthetic faults.",
        "Regular calendar grid defines boundaries; missing observations are not filled.",
        "Construct windows per compartment and within continuous segments; targets must remain within their partition.",
        "Fit preprocessing only on each inner fit set; refit on outer development after tuning.",
        "Never tune or calibrate quantization on any final-test records or excluded-compartment observations.",
        "test_all.csv is duplicated across outer folds; do not treat repeated predictions as independent tests."]}
    (out / "experiment_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {len(compartments)} outer folds, each with {inner_splits} chronological validation folds.")
    print(f"Final testing starts: {test_start}. Output: {out}")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--timestamp-col", default="Timestamp")
    p.add_argument("--compartment-col", default="Compartment")
    p.add_argument("--interval-minutes", type=float, required=True)
    p.add_argument("--development-ratio", type=float, default=0.7)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--gap-steps", type=int, default=0)
    p.add_argument("--timestamp-mode", choices=["text", "excel"], default="text")
    p.add_argument("--date-format", default=None)
    p.add_argument("--expected-compartments", type=int, default=6,
                   help="Use 0 to accept any compartment count >=2.")
    a = p.parse_args()
    build_splits(a.input, a.output_dir, a.timestamp_col, a.compartment_col,
        a.interval_minutes, a.development_ratio, a.inner_splits, a.gap_steps,
        a.timestamp_mode, a.date_format, a.expected_compartments)

if __name__ == "__main__":
    main()