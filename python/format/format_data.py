import pandas as pd
import os

# 1. Load your specific file verbatim
print("Loading data...")
df = pd.read_csv("Greenhouse_climate.csv")

# 2. Map the original Wageningen column names to your pipeline's column names
# (If your CSV uses slightly different headers like 'T_air', adjust the left side of this list)
rename_map = {
    'Tair': 'Temperature',
    'RHair': 'Humidity',
    'CO2air': 'CO2',
}

# 3. Safely check which columns actually exist in your file 
# (in case the weather columns are in a separate file)
existing_cols = {old_name: new_name for old_name, new_name in rename_map.items() if old_name in df.columns}

# 4. Filter and rename the columns
df_final = df[list(existing_cols.keys())].rename(columns=existing_cols)

# 5. Save the output as climate_data.csv
output_path = "climate_data.csv"
df_final.to_csv(output_path, index=False)

print(f"Success! Converted data saved to {output_path}")
print(f"Columns included: {list(df_final.columns)}")