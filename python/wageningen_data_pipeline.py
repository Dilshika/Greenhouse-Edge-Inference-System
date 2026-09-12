import argparse
import json
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd


## Prepare rge mode functions
# adding the sensor noise
def add_sensor_noise(series, std_multiplier=0.03,drift_factor=0.0005,spike_prob=0.01):
    std_dev = series.std()
    if pd.isna(std_dev) or std_dev == 0:
        std_dev = 1.0

    # Gaussian noise (2-5% of signal std)
    noise = np.random.normal(0,std_dev*std_multiplier, len(series))

    #Gradual noise (2-5% of signal std)
    drift = np.linspace(0, std_dev * drift_factor * len(series), len(series))

    # 1% Occasional spikes/glitches
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

    # Clean data (drops NaNs for simplicity)
    df = df.dropna()
    split_index = int(len(df)*0.7)
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

    # Create supervised learning arrays from the Noisy data
    # (Predicting next step Temperature based on noisy current data)

    X_train = train_noisy_df.iloc[:-1].values
    y_train = train_noisy_df['Temperature'].iloc[1:].values
    
    X_test = test_noisy_df.iloc[:-1].values
    y_test = test_noisy_df['Temperature'].iloc[1:].values

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir,"y_train.npy"),y_train)
    np.save(os.path.join(data_dir,"X_test.npy"), X_test)
    np.save(os.path.join(data_dir,"y_test.npy"), y_test)

    print(f"Saved Numpy arrays (X_train, y_train, X_test, y_test) to {data_dir}")

## Train Mode Functions
def reshape_for_rnn(X_data, window=10):
    X_flat = np.array(X_data).flatten()
    n_features = X_data.shape[1]
    block_size = window * n_features

    remainder = X_flat.size % block_size
    if remainder !=0:
        X_flat = X_flat[:-remainder]

    return X_flat.reshape(-1, window, n_features)

def train_models(data_dir, models_dir):
    print("\n=== MODE: TRAIN MODELS ===")
    os.makedirs(models_dir, exist_ok=True)

    try:
        import xgboost as xgb
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, LSTM, Dense
    except ImportError as e:
        print(f"Missing dependency for training: {e}")
        print("Please run: pip install xgboost tensorflow")
        return

    # Load Data
    X_train = np.load(os.path.join(data_dir, "X_tain.npy"))
    y_train = np.load(os.path.join(data_dir,"y_train.npy"))

    # -- Train XGBoost --
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost to {xgb_path}")

    # Prepare Data for RNNs
    window_size =10
    X_train_rnn = reshape_for_rnn(X_train, window=window_size)

    #Trim y_train to match the reshaped X_train_run length
    y_train_rnn = y_train[:X_train_run.shape[0]]

    # -- Train GRU --
    print("\nTraining GRU ...")
    gru_model = Sequential([
        GRU(48, input_shape=(window_size, X_train.shape[1])),
        Dense(1)
    ])

    gru_model.compile(optimizer='adam', loss ='mse')
    gru_model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, verbose=1)
    gru_path = os.path.join(models_dir, "gru_model.h5")
    gru_model.save(gru_path)
    print(f"Saved GRU to {gru_path}")

    # -- Train LSTM --
    print("\nTraining LSTM...")
    lstm_model = Sequential([
        LSTM(48, input_shape=(window_size, X_train.shape[1])),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, verbose=1)
    lstm_path = os.path.join(models_dir, "lstn_model.h5")
    print(f"Saved LSTM to {lstm_path}")

#Strem mode Functions
def stream_data(data_dir, broker, port, topic, interval):
    print("\n== MODE: STREAM TELEMETRY ===")

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print ("Missing paho-mqtt. Please run: pip install paho-mqtt")
        return

    # use the noisy test data we generated in the updated prepare steps
    csv_path = os.path.join(data_dir, "test_clean.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run --moode prepare first.")
        return 

    df= pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from test set for streaming.")

    #Setup MQTT Client
    client = mqtt.Client()
    try:
        client.connect(broker, port, 60)
        print(f"Conencted to MQTT broker at {broker}:{port}")
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is Mosquitto running on {broker}:{port}?")
        return

    client.loop_start()

    publish_topic = f"{topic}/telemetry"
    print(f"Publishing to '{publish_topic}' every {interval} seconds... \n")

    try:
        for idx, row in df.iterrows():
            # Extract values (adding a tiny bit of fresh stream noise)
            temp = round(row['Temperature'] + np.random.normal(0, 0.1), 2)
            hum = round(row['Humidity'] + np.random.normal(0, 0.5), 1)
            co2 = round(row['CO2'] + np.random.normal(0, 2.0), 1)
            rad = round(row['Radiation'] + np.random.normal(0, 5.0), 1)
            out_temp = round(row['outside_temperature'], 2)
            out_hum = round(row['outside_humidity'], 1)
            
            # Publish each sensor individually to its own topic
            client.publish(f"{topic}/temperature", temp)
            client.publish(f"{topic}/humidity", hum)
            client.publish(f"{topic}/co2", co2)
            client.publish(f"{topic}/radiation", rad)
            client.publish(f"{topic}/outside_temperature", out_temp)
            client.publish(f"{topic}/outside_humidity", out_hum)
            
            print(f"[{idx}] Published 6 independent sensor readings.")
            
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wageningen Dataset Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "train", "stream"], 
                        help="Mode to run: prepare, train, or stream")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory for trained models")
    parser.add_argument("--mqtt-broker", type=str, default="localhost", help="MQTT broker address")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    
    # Minor tweak to the help text here:
    parser.add_argument("--mqtt-topic", type=str, default="greenhouse/sensors", 
                        help="BASE MQTT topic (script will publish to topic/temperature, topic/humidity, etc.)")
    
    parser.add_argument("--publish-interval", type=int, default=10, help="Seconds between MQTT messages")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        prepare_data(args.data_dir)
    elif args.mode == "train":
        train_models(args.data_dir, args.models_dir)
    elif args.mode == "stream":
        stream_data(args.data_dir, args.mqtt_broker, args.mqtt_port, args.mqtt_topic, args.publish_interval)