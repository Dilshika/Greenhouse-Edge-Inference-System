# Wageningen Dataset Pipeline 

### This pipeline has three modes

1. Prepare - Download Wageningen dataset, clean it, add sensor noise, split 70/30 into train/test

2. Train - Train XGBoost, GRU,LSTM models on the preparef training data

3. Stream - Simulate the real-time sensor telemetry over MQTT using the test dataset


## Installation

### Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   |  On Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements_pipeline.txt


## Step 1: Prepare Dataset(70/30 Split + Sensor Noise)

python wageningen_data_pipeline.py --mode prepare --data-dir ./data

#### what this does

Downloads the Wageningen Autonomous Greenhouse dataset(First Edition, 2018) from 4TU.ResearchData

Extracts and cleans climate columns (temperature. humidity, C02, radiation, outside weather)

Add realistic sensor noise:

- Gaussian noise (2-5% of signal std)
- Gradual sensor drift
- 1% occasional so=pikes/glitches

Splits 70% training / 30% testing

Saves

- data/train_clean.csv - raw training data
- data/test_clean.csv - raw testing data (for MQTT streaming)
- data/X_train.npy, y_train.npy - supervised learning arrays
- data/X_test.npy, y_test.npy - supervised learning arrays

  Output Example:
  Checks inside the data folder

- You can generate a side-by-side plot to verify the injected hardware noise.

Open the visualize_noise.py file.

Change the column_to_plot variable if you want to visualize a different sensor (it is currently set to 'Temperature').

Run the script:
python visualize_noise.py

This will generate and save a comparison plot image (e.g., temp_noise_visualization.png) directly inside your ./data folder. 
