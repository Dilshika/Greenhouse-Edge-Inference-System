import os
import pandas as pd
import numpy as np

# Column mapping configuration since 1st and 2nd edition might have slightly different names
TARGET_COLUMNS = ["Timestamp", "Edition", "Compartment", "Temperature", "Humidity", "CO2"]

# Initial quality-screening configuration.
# Confirm the actual interval and inspect the flagged events before final use.
EXPECTED_INTERVAL_MINUTES = 5
FLATLINE_DURATION_MINUTES = 30

FLATLINE_TOLERANCES = {
    "Temperature": 0.05,  # degrees Celsius
    "Humidity": 0.10,     # relative-humidity percentage points
    "CO2": 1.0,          # ppm
}


def add_quality_flags(
    data,
    expected_interval_minutes=EXPECTED_INTERVAL_MINUTES,
    flatline_duration_minutes=FLATLINE_DURATION_MINUTES,
    tolerances=None,
):
    """
    Add causal, per-compartment missing-value and suspected-flatline flags.

    A flatline flag becomes True only after a complete trailing window
    of regularly spaced, valid observations has been received.

    Original measurement values are preserved.
    """
    if tolerances is None:
        tolerances = FLATLINE_TOLERANCES.copy()

    if expected_interval_minutes <= 0:
        raise ValueError("Expected interval must be positive.")

    if flatline_duration_minutes <= 0:
        raise ValueError("Flatline duration must be positive.")

    required_columns = [
        "Timestamp",
        "Edition",
        "Compartment",
        "Temperature",
        "Humidity",
        "CO2",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    result = data.copy()

    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"],
        errors="raise",
    )

    if result["Timestamp"].isna().any():
        raise ValueError(
            "Missing timestamps found. Resolve them before flagging."
        )

    group_columns = ["Edition", "Compartment"]
    identity_columns = group_columns + ["Timestamp"]

    if result.duplicated(identity_columns).any():
        examples = result.loc[
            result.duplicated(identity_columns, keep=False),
            identity_columns,
        ].head(10)

        raise ValueError(
            "Duplicate timestamps within a compartment were found. "
            "Resolve them before calculating flatline flags.\n"
            + examples.to_string(index=False)
        )

    result = result.sort_values(
        identity_columns
    ).reset_index(drop=True)

    # Convert measurements explicitly and preserve missing values.
    for variable in tolerances:
        original = result[variable]
        numeric = pd.to_numeric(original, errors="coerce")

        conversion_failures = original.notna() & numeric.isna()

        if conversion_failures.any():
            examples = original.loc[
                conversion_failures
            ].head(5).tolist()

            raise ValueError(
                f"Unexpected non-numeric values in '{variable}': "
                f"{examples}. Resolve documented missing-value codes first."
            )

        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(
                f"Infinite values found in '{variable}'. "
                "Resolve these explicitly before flagging."
            )

        result[variable] = numeric

    expected_interval = pd.Timedelta(
        minutes=expected_interval_minutes
    )

    # For 30 minutes at five-minute intervals:
    # ceil(30 / 5) + 1 = 7 observations.
    window_points = (
        int(
            np.ceil(
                flatline_duration_minutes
                / expected_interval_minutes
            )
        )
        + 1
    )

    output_groups = []

    for _, group in result.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        group = group.copy()

        # Start a new segment at every timestamp discontinuity.
        # Rolling windows therefore cannot bridge missing time intervals.
        time_difference = group["Timestamp"].diff()

        segment_id = (
            time_difference.ne(expected_interval)
            .cumsum()
        )

        for variable, tolerance in tolerances.items():
            if tolerance < 0:
                raise ValueError(
                    f"Tolerance for '{variable}' cannot be negative."
                )

            values = group[variable]

            group[f"{variable}_missing"] = values.isna()

            rolling_range = values.groupby(segment_id).transform(
                lambda segment: (
                    segment.rolling(
                        window=window_points,
                        min_periods=window_points,
                    ).max()
                    -
                    segment.rolling(
                        window=window_points,
                        min_periods=window_points,
                    ).min()
                )
            )

            # NaN rolling ranges indicate insufficient valid history.
            group[f"{variable}_suspected_flatline"] = (
                rolling_range.le(tolerance).fillna(False)
            )

            # Distinguishes "not flagged" from "not yet assessable".
            group[f"{variable}_flatline_assessable"] = (
                rolling_range.notna()
            )

        output_groups.append(group)

    result = pd.concat(output_groups, ignore_index=True)

    quality_columns = []

    for variable in tolerances:
        quality_columns.extend([
            f"{variable}_missing",
            f"{variable}_suspected_flatline",
        ])

    result["any_quality_flag"] = (
        result[quality_columns].any(axis=1)
    )

    return result.sort_values(
        ["Timestamp", "Edition", "Compartment"]
    ).reset_index(drop=True)


def determine_timestamp_col(df):
    """Finds the most likely timestamp column, accounting for GHtime and spaces."""
    possible_names = ["%time", "time", "timestamp", "date", "ghtime"]
    for col in df.columns:
        if col.strip().lower() in possible_names:
            return col
    return df.columns[0]

def determine_sensor_cols(df):
    """Maps the raw columns to the target columns, stripping hidden whitespace."""
    col_map = {"Temperature": None, "Humidity": None, "CO2": None}
    
    for col in df.columns:
        col_lower = col.strip().lower() # .strip() removes hidden spaces
        
        if col_lower in ['tair', 't_air', 'temperature'] and col_map["Temperature"] is None:
             col_map["Temperature"] = col
             
        elif col_lower in ['rhair', 'rh_air', 'humidity', 'rh'] and col_map["Humidity"] is None:
             col_map["Humidity"] = col
             
        elif col_lower in ['co2air', 'co2_air', 'co2'] and col_map["CO2"] is None:
             col_map["CO2"] = col
             
    return col_map

def audit_and_merge(base_dir):
    audit_results = []
    
    # We will collect the dataframes to merge them later
    first_edition_dfs = []
    second_edition_dfs = []

    for edition in ["first_edition", "second_edition"]:
        raw_dir = os.path.join(base_dir, edition, "raw")
        if not os.path.exists(raw_dir):
            print(f"Warning: Directory not found: {raw_dir}")
            continue

        # Each compartment has its own folder inside raw
        for compartment_name in os.listdir(raw_dir):
            compartment_path = os.path.join(raw_dir, compartment_name)
            
            if not os.path.isdir(compartment_path):
                continue
                
            # Search for the climate csv
            csv_path = None
            for filename in os.listdir(compartment_path):
                if filename.lower() == "greenhouseclimate.csv" or filename.lower() == "greenhouse_climate.csv":
                    csv_path = os.path.join(compartment_path, filename)
                    break
            
            if not csv_path:
                print(f"No climate CSV found in {compartment_path}")
                continue

            try:
                df = pd.read_csv(
                    csv_path,
                    skipinitialspace=True,
                    na_values=["NaN", "nan", "NAN", ""],
                    keep_default_na=True,
                    )

                # Clean surrounding whitespace in column headers.
                df.columns = df.columns.str.strip()
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            # --- 1. AUDIT PHASE ---
            time_col = determine_timestamp_col(df)
            sensor_cols = determine_sensor_cols(df)
            
            # Convert timestamp to datetime for analysis
            # Convert Excel serial dates (days since 1899-12-30) to standard datetime
            # Parse the verified Excel serial-day timestamps.
            excel_serial = pd.to_numeric(
                df[time_col],
                errors="coerce",
            )

            parsed_time = pd.to_datetime(
                excel_serial,
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )

            if parsed_time.isna().any():
                bad_rows = df.loc[
                    parsed_time.isna(),
                    [time_col],
                ].head(10)

                raise ValueError(
                    f"Unparseable timestamps in {csv_path}:\n"
                    f"{bad_rows.to_string(index=False)}"
                )

            # Correct only very small deviations from whole-second timestamps.
            rounded_time = parsed_time.dt.round("s")

            correction_seconds = (
                rounded_time - parsed_time
            ).abs().dt.total_seconds()

            # Guard against accidentally correcting genuine sub-second observations.
            # These settings assume:
            # 1. Excel serial-day timestamps have been verified.
            # 2. A five-minute recording grid has been verified.
            # 3. Second-edition source precision explains the sub-second offsets.

            TIMESTAMP_CORRECTION_TOLERANCES = {
                "first_edition": 0.001,
                "second_edition": 0.45,
            }

            excel_serial = pd.to_numeric(
                df[time_col],
                errors="coerce",
            )

            original_time = pd.to_datetime(
                excel_serial,
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )

            if original_time.isna().any():
                examples = df.loc[
                    original_time.isna(),
                    [time_col],
                ].head(10)

                raise ValueError(
                    f"Unparseable timestamps in {csv_path}:\n"
                    + examples.to_string(index=False)
                )

            # Normalise to whole seconds, NOT directly to five minutes.
            normalised_time = original_time.dt.round("s")

            correction_seconds = (
                normalised_time - original_time
            ).abs().dt.total_seconds()

            allowed_correction = TIMESTAMP_CORRECTION_TOLERANCES[edition]

            exceeds_tolerance = correction_seconds.gt(allowed_correction)

            if exceeds_tolerance.any():
                examples = pd.DataFrame({
                    "original": original_time,
                    "normalised": normalised_time,
                    "correction_seconds": correction_seconds,
                }).loc[exceeds_tolerance].head(10)

                raise ValueError(
                    f"{edition}/{compartment_name}: timestamp correction "
                    f"exceeds {allowed_correction} seconds. "
                    "Inspect source precision and recording schedule.\n"
                    + examples.to_string(index=False)
                )

            # Check that normalisation does not collapse distinct rows
            # onto the same timestamp.
            duplicate_mask = normalised_time.duplicated(keep=False)

            if duplicate_mask.any():
                examples = pd.DataFrame({
                    "original": original_time,
                    "normalised": normalised_time,
                }).loc[duplicate_mask].head(10)

                raise ValueError(
                    f"{edition}/{compartment_name}: duplicate timestamps "
                    "after normalisation. Resolve explicitly.\n"
                    + examples.to_string(index=False)
                )

            # Verify the intended five-minute calendar grid.
            # Missing observations are allowed; off-grid observations are not.
            grid_step = pd.Timedelta(minutes=5)
            grid_origin = normalised_time.min().normalize()

            remainders = (
                normalised_time - grid_origin
            ) % grid_step

            off_grid = remainders.ne(pd.Timedelta(0))

            if off_grid.any():
                examples = pd.DataFrame({
                    "original": original_time,
                    "normalised": normalised_time,
                }).loc[off_grid].head(10)

                raise ValueError(
                    f"{edition}/{compartment_name}: timestamps remain "
                    "off the verified five-minute grid.\n"
                    + examples.to_string(index=False)
                )

            # Preserve both the source representation and parsed original time.
            df["timestamp_source_value"] = df[time_col]
            df["timestamp_before_normalisation"] = original_time
            df["timestamp_correction_seconds"] = correction_seconds

            # Use this timestamp for auditing, quality flags and splitting.
            df["parsed_time"] = normalised_time

            print(
                f"{edition}/{compartment_name}: "
                f"{int(correction_seconds.gt(0).sum()):,} timestamps normalised; "
                f"maximum correction = {correction_seconds.max():.9f} s; "
                f"allowed = {allowed_correction} s"
            )

            num_records = len(df)
            
            if num_records > 0 and df['parsed_time'].notna().any():
                valid_times = df['parsed_time'].dropna()
                earliest = valid_times.min().strftime('%Y-%m-%d')
                latest = valid_times.max().strftime('%Y-%m-%d')
                date_range = f"{earliest} to {latest}"
                
                # Calculate typical interval
                # Sort first to ensure diffs are positive time steps
                sorted_times = valid_times.sort_values()
                if len(sorted_times) > 1:
                     intervals = sorted_times.diff().dt.total_seconds().dropna()
                     # Get the most common interval (mode) in minutes
                     mode_seconds = intervals.mode()
                     if not mode_seconds.empty:
                          typical_interval = f"{int(mode_seconds.iloc[0] / 60)} min"
                     else:
                          typical_interval = "Varies"
                else:
                    typical_interval = "N/A"
            else:
                date_range = "Unknown"
                typical_interval = "Unknown"

            # Missing percentages
            def get_missing_pct(col_name):
                if col_name and col_name in df.columns:
                    missing_count = df[col_name].isna().sum()
                    return f"{(missing_count / num_records) * 100:.1f}%"
                return "100% (Missing Col)"

            missing_temp = get_missing_pct(sensor_cols["Temperature"])
            missing_hum = get_missing_pct(sensor_cols["Humidity"])
            missing_co2 = get_missing_pct(sensor_cols["CO2"])

            audit_results.append({
                "Edition": edition.replace("_", " ").title(),
                "Compartment": compartment_name,
                "Records": num_records,
                "Date range": date_range,
                "Typical interval": typical_interval,
                "Missing temperature": missing_temp,
                "Missing humidity": missing_hum,
                "Missing CO₂": missing_co2
            })
            
            # --- 2. PREPARE PHASE ---
            if all(v is not None for v in sensor_cols.values()):
                clean_df = pd.DataFrame({
                    "Timestamp": df["parsed_time"],
                    "Edition": 1 if edition == "first_edition" else 2,
                    "Compartment": compartment_name,
                    "Temperature": df[sensor_cols["Temperature"]],
                    "Humidity": df[sensor_cols["Humidity"]],
                    "CO2": df[sensor_cols["CO2"]],
                })

                if clean_df["Timestamp"].isna().any():
                    bad_count = int(clean_df["Timestamp"].isna().sum())

                    raise ValueError(
                        f"{csv_path}: {bad_count} timestamps could not be parsed. "
                        "Check the timestamp column and its representation."
                )

                if edition == "first_edition":
                    first_edition_dfs.append(clean_df)
                else:
                    second_edition_dfs.append(clean_df)

    # Print Audit Table
    # Save Audit Table to Excel
    if audit_results:
        print("\nSaving Data Audit Table to Excel...")
        audit_df = pd.DataFrame(audit_results)
        audit_out_path = os.path.join(base_dir, "wageningen_dataset_audit.xlsx")
        
        try:
            # Using pandas to_excel to generate the file
            audit_df.to_excel(audit_out_path, index=False, sheet_name="Audit Summary")
            print(f"✅ Audit table saved successfully to: {audit_out_path}\n")
        except ImportError:
            print("⚠️ 'openpyxl' is required to save to Excel. Run: pip install openpyxl")
            print("Falling back to CSV...")
            audit_out_path = os.path.join(base_dir, "wageningen_dataset_audit.csv")
            audit_df.to_csv(audit_out_path, index=False)
            print(f"✅ Audit table saved as CSV to: {audit_out_path}\n")
    else:
        print("\nNo audit data generated to save.\n")

    # Save Merged Datasets
    def save_merged(dfs, edition_name):
        if not dfs:
            print(f"No data parsed for {edition_name}")
            return

        merged_df = pd.concat(dfs, ignore_index=True)

        # Preserve original observations and add quality flags.
        merged_df = add_quality_flags(merged_df)

        out_dir = os.path.join(
            base_dir,
            edition_name,
            "prepared",
        )

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(
            out_dir,
            f"{edition_name}_data.csv",
        )

        merged_df.to_csv(out_path, index=False)

        flag_columns = [
            column
            for column in merged_df.columns
            if column.endswith(
                ("_missing", "_suspected_flatline")
            )
        ]

        summary = (
            merged_df
            .groupby(["Edition", "Compartment"])[flag_columns]
            .sum()
            .astype(int)
            .reset_index()
        )

        row_counts = (
            merged_df
            .groupby(["Edition", "Compartment"])
            .size()
            .rename("Records")
            .reset_index()
        )

        summary = row_counts.merge(
            summary,
            on=["Edition", "Compartment"],
            how="left",
        )

        summary_path = os.path.join(
            out_dir,
            f"{edition_name}_quality_summary.csv",
        )

        summary.to_csv(summary_path, index=False)

        print(
            f"Saved prepared data ({len(merged_df):,} rows): "
            f"{out_path}"
        )

        print(f"Saved quality summary: {summary_path}")
        print(summary.to_string(index=False))

    save_merged(first_edition_dfs, "first_edition")
    save_merged(second_edition_dfs, "second_edition")

if __name__ == "__main__":
    # Ensure this points to the directory containing first_edition and second_edition
    # Based on the image, the script runs from inside the 'python' directory
    DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), "data") 
    
    if os.path.exists(DATA_BASE_DIR):
        audit_and_merge(DATA_BASE_DIR)
    else:
         print(f"Data directory not found at {DATA_BASE_DIR}")