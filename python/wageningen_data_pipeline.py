import argparse
import json
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

#The core variables need to predict - as of now
TARGET_VARS = ['Temperature', 'Humidity', 'CO2']

# Forecast horizon in steps (e.g if data is coming 5 min intervals 1=5min, then 3=15 min)
HORIZON = 3


## Prepare rge mode functions
# adding the sensor noise
def add_sensor_noise(series, std_multiplier=0.03,drift_factor=0.0005,spike_prob=0.01):
    std_dev = series.std()
    if pd.isna(std_dev) or std_dev == 0:
        std_dev = 1.0

    # Gaussian noise (2-5% of signal std)
    noise = np.random.normal(0, std_dev * std_multiplier, len(series))

    #Gradual noise (2-5% of signal std)
    drift = np.linspace(0, std_dev * drift_factor * len(series), len(series))

    # 1% Occasional hardware spikes/glitches
    spikes = np.where(
        np.random.rand(len(series)) < spike_prob,
        std_dev * 5 * np.random.choice([-1,1],len(series)),
        0
    )

    return series + noise + drift + spikes

## Prepare the data
def prepare_data(data_dir):
    print("\n=== MODE: PREPARE DATASET ===")

    # look for the file in the provided directory
    csv_path = os.path.join(data_dir,"climate_data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n ERROR: Raw dataset not found at '{csv_path}'.\n"
            f"Please ensure you have generated 'climate_data.csv' and placed it in the '{data_dir}' folder."
        )

    print(f"Reading raw dataset from: {csv_path}")
    df= pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, columns:{list(df.columns)} ")

    # Keep only numeric feature columns and drop NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].dropna().reset_index(drop=True)

    # Verify our 3 target columns exist
    for col in TARGET_VARS:
        if col not in df.columns:
            raise KeyError(f"Required target column '{col}' not found in {df.columns.tolist()}")

    # 70/30 Chronological Split
    split_index = int(len(df) * 0.7)
    train_clean_df = df.iloc[:split_index].copy()
    test_clean_df = df.iloc[split_index:].copy()

    # Save the Clean versions
    train_clean_df.to_csv(os.path.join(data_dir, "train_clean.csv"), index=False)
    test_clean_df.to_csv(os.path.join(data_dir,"test_clean.csv"),index=False)

    print("Saved pristine datasets: 'train_clean.csv' and 'test_clean.csv' ")

    #Prepare noisy versions
    train_noisy_df = train_clean_df.copy()
    test_noisy_df = test_clean_df.copy()

    for col in df.columns:
        train_noisy_df[col] = add_sensor_noise(train_noisy_df[col])
        test_noisy_df[col] = add_sensor_noise(test_noisy_df[col])

    # Save the noisy versions
    train_noisy_df.to_csv(os.path.join(data_dir, "train_noisy.csv"), index=False)
    test_noisy_df.to_csv(os.path.join(data_dir, "test_noisy.csv"), index=False)
    print("Applied sensor noise and saved: 'train_noisy.csv' and 'test_noisy.csv' ")

    # --- CREATE SUPERVISED ARRAYS ---
    # Inputs (X) = Noisy sensor readings at time t
    # Targets (y) = Clean REAL ground truth [Temperature, Humidity, CO2] at time t + HORIZON
    X_train = train_noisy_df.iloc[:-HORIZON].values
    y_train = train_clean_df[TARGET_VARS].iloc[HORIZON:].values

    X_test = test_noisy_df.iloc[:-HORIZON].values
    y_test = test_clean_df[TARGET_VARS].iloc[HORIZON:].values

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print(f"Saved Numpy arrays (X_train, y_train, X_test, y_test) to {data_dir}")

## --- 2. TRAIN MODE FUNCTIONS ---
def create_sliding_windows(X_data, y_data, window=10):
    """
    Creates time-aligned 3D sliding windows for GRU/LSTM so that
    X[i : i+window] predicts the exact aligned target y[i + window - 1].
    """

    X_windows, y_windows = [], []
    for i in range(len(X_data) - window + 1):
        X_windows.append(X_data[i : i + window])
        y_windows.append(y_data[i + window - 1])
    return np.array(X_windows), np.array(y_windows)

def train_models(data_dir, models_dir):
    print("\n=== MODE: TRAIN MODELS ===")
    os.makedirs(models_dir, exist_ok=True)

    try:
        import xgboost as xgb
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, LSTM, Dense, Input
    except ImportError as e:
        print(f"Missing dependency for training: {e}")
        print("Please run: pip install xgboost tensorflow")
        return

    # Load Data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir,"y_train.npy"))

    n_features = X_train.shape[1]
    n_targets = y_train.shape[1]  # 3 targets: Temperature, Humidity, CO2
    window_size = 10

    print(f"Training on {X_train.shape[0]} samples | Features: {n_features} | Targets: {n_targets} ({TARGET_VARS})")

    # -- 1. Train XGBoost (Multi-output) --
    print("\n[1/3] Training Multi-Output XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost to {xgb_path}")

    # Prepare time-aligned 3D Sliding Windows for GRU & LSTM
    X_train_rnn, y_train_rnn = create_sliding_windows(X_train, y_train, window=window_size)
    print(f"\nPrepared RNN sliding windows -> X: {X_train_rnn.shape}, y: {y_train_rnn.shape}")

    # -- 2. Train GRU (Multi-output) --
    print("\n[2/3] Training Multi-Output GRU...")
    gru_model = Sequential([
        Input(shape=(window_size, n_features)),
        GRU(48),
        Dense(n_targets)  # Outputs [Temperature, Humidity, CO2]
    ])
    gru_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    gru_model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, verbose=1)
    gru_path = os.path.join(models_dir, "gru_model.keras")
    gru_model.save(gru_path)
    print(f"Saved GRU to {gru_path}")

    # -- 3. Train LSTM (Multi-output) --
    print("\n[3/3] Training Multi-Output LSTM...")
    lstm_model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(48),
        Dense(n_targets)  # Outputs [Temperature, Humidity, CO2]
    ])
    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    lstm_model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, verbose=1)
    lstm_path = os.path.join(models_dir, "lstm_model.keras")
    lstm_model.save(lstm_path)
    print(f"Saved LSTM to {lstm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wageningen Dataset Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "train"], 
                        help="Mode to run: prepare, train, or stream")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory for trained models")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        prepare_data(args.data_dir)
    elif args.mode == "train":
        train_models(args.data_dir, args.models_dir)