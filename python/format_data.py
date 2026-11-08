import os
import pandas as pd
import numpy as np

# Column mapping configuration since 1st and 2nd edition might have slightly different names
TARGET_COLUMNS = ["Timestamp", "Edition", "Compartment", "Temperature", "Humidity", "CO2"]

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
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            # --- 1. AUDIT PHASE ---
            time_col = determine_timestamp_col(df)
            sensor_cols = determine_sensor_cols(df)
            
            # Convert timestamp to datetime for analysis
           # Convert Excel serial dates (days since 1899-12-30) to standard datetime
            df['parsed_time'] = pd.to_numeric(df[time_col], errors='coerce')
            df['parsed_time'] = pd.to_datetime(df['parsed_time'], unit='D', origin='1899-12-30')
            
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
            # Filter and rename for the final merged dataset
            if all(v is not None for v in sensor_cols.values()):
                clean_df = pd.DataFrame({
                    "Timestamp": df[time_col],
                    "Edition": 1 if edition == "first_edition" else 2,
                    "Compartment": compartment_name,
                    "Temperature": df[sensor_cols["Temperature"]],
                    "Humidity": df[sensor_cols["Humidity"]],
                    "CO2": df[sensor_cols["CO2"]]
                })
                
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
        # Drop rows where target variables are NaN
        merged_df = merged_df.dropna(subset=["Temperature", "Humidity", "CO2"])
        
        out_dir = os.path.join(base_dir, edition_name, "prepared")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{edition_name}_data.csv")
        
        merged_df.to_csv(out_path, index=False)
        print(f"Saved prepared data ({len(merged_df)} rows) to: {out_path}")

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