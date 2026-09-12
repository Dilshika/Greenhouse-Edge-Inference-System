# Wageningen Dataset Pipeline 

# This pipeline has three modes

1. Prepare - Download Wageningen dataset, clean it, add sensor noise, split 70/30 into train/test

2. Train - Train XGBoost, GRU,LSTM models on the preparef training data

3. Stream - Simulate the real-time sensor telemetry over MQTT using the test dataset

