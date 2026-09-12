# Greenhouse Edge Inference System

Realtime climate forecasting and actuation for Wageningen-style greenhouse, split across Python ML pipeline, an edge inference service, and a Node.js orchestration/ dashboard layer connected over MQTT.

The system predicts temperature and humidity 15-30 minutes ahead using GRU, XGBoost and LSTM models trained on the Wageningen dataset, runs the selected model on a Raspberry Pi 5 (Currently simulated with Mac), and exposes live telemetry, forecasts, and manual overrides through a React Dashboard or n8n bot Telegram command

##Architecture

```
┌─────────────────────────────┐        ┌─────────────────────────────┐
│  Python — data science / ML │        │  Node.js — orchestration     │
│  ─────────────────────────  │        │  ─────────────────────────   │
│  wageningen_data_pipeline.py│        │  HIL simulator                │
│   - 70/30 dataset split     │        │  Backend API (REST/WSS)       │
│   - sensor noise injection  │  MQTT  │  MongoDB (telemetry + logs)   │
│   - trains XGBoost/GRU/LSTM │◄──────►│  n8n + Telegram alerts        │
│  Model inference service    │        │  React dashboard               │
└─────────────────────────────┘        └─────────────────────────────┘
                 ▲                                    ▲
                 │                MQTT (Eclipse Mosquitto)
                 ▼                                    │
┌───────────────────────────────────────────────────────────────────┐
│  Edge node — Raspberry Pi 5 (simulated or physical)                │
│   - subscribes: greenhouse/sensors/telemetry                       │
│   - predicts temperature/humidity 15–30 min ahead                  │
│   - rule-based thresholds → anomaly detection                      │
│   - publishes: greenhouse/actuate/cmd                              │
│   - optional EdgeSimulator: CPU%, memory, inference latency         │
└───────────────────────────────────────────────────────────────────┘
```


## Components
 
| Layer | Component | Status |
|---|---|---|
| Python | `wageningen_data_pipeline.py` — dataset prep, training, model export | to be built |
| Python | Model inference service — MQTT subscribe, real-time inference, publish actuation | to be built |
| Edge | Inference + rule-based anomaly detection + `EdgeSimulator` | to be built |
| Node.js | HIL simulator — replay dataset or stream live test data, inject outages | to be built |
| Node.js | Backend API — REST/WSS, MongoDB storage, decision attribution logs | to be built |
| Node.js | n8n + Telegram bot — mobile alerts, manual override commands | to be built |
| Node.js | React dashboard — telemetry, forecasts, model comparison, overrides | to be built |
 
## Dashboard
 
The dashboard (`GreenhouseDashboard.jsx`) is the operator-facing console for the system. It shows:
 
- **Live sensor tiles** for air temperature, relative humidity, CO2, PAR light, and soil moisture, each with a rolling sparkline and target band
- **Forecast chart** overlaying actual vs. predicted temperature, with the target range shaded and a horizon of 15–30 minutes
- **Model comparison** across XGBoost, GRU, and LSTM with live rolling MAE, so an operator can see which model is currently deployed and swap the forecast overlay
- **Anomaly & decision log** — timestamped, severity-coded, and attributed to the model or watchdog that raised it
- **Manual override panel** — per-actuator Auto/Manual mode with an on/off switch for roof vent, heating, irrigation, CO2 injector, and shade screen
- **Connection status** for the MQTT broker (Mosquitto) and edge node health (CPU, inference latency)
### Preview
 
The component ships with a self-contained telemetry simulator so it runs standalone with no backend — useful for UI iteration before the real MQTT/WSS wiring is in place.
 
## Getting started
 
### Prerequisites
 
- Node.js 18+
- Python 3.10+
- Eclipse Mosquitto (or another MQTT 3.1.1/5 broker)
- MongoDB
### 1. Train the models
 
```bash
cd python
pip install -r requirements.txt
python wageningen_data_pipeline.py
```
 
This downloads and splits the Wageningen dataset, adds sensor noise, trains XGBoost/GRU/LSTM, and writes model artifacts to `python/models/`.
 
### 2. Run the MQTT broker
 
```bash
mosquitto -c mosquitto.conf
```
 
### 3. Start the edge inference service
 
```bash
cd python/edge_inference
python inference_service.py --broker localhost --models ../models
```
 
### 4. Start the Node.js backend and dashboard
 
```bash
cd server
npm install
npm run dev        # backend API + WSS
 
cd ../dashboard
npm install
npm start           # React dashboard
```
 
### 5. (Optional) Hardware-in-the-loop simulation
 
```bash
cd server/hil-simulator
npm run replay -- --dataset wageningen --speed 4x
```
 
## MQTT topics
 
| Topic | Direction | Payload |
|---|---|---|
| `greenhouse/sensors/telemetry` | edge ← sensors | temperature, humidity, CO2, light, soil moisture |
| `greenhouse/predictions` | edge → backend | per-model forecast + active model |
| `greenhouse/actuate/cmd` | edge → actuators | actuation command from rule-based threshold check |
| `greenhouse/actuate/override` | dashboard → edge | manual override command |
| `greenhouse/alerts` | edge → backend/n8n | anomaly event with severity + attribution |
 
## Project structure
 
```
.
├── python/
│   ├── wageningen_data_pipeline.py
│   ├── edge_inference/
│   │   └── inference_service.py
│   └── models/
├── server/
│   ├── api/               # REST/WSS backend
│   ├── hil-simulator/
│   └── n8n-flows/
└── dashboard/
    └── src/
        └── GreenhouseDashboard.jsx
```
 
## Roadmap
 
- [ ] Build the Python model inference service (MQTT subscribe/publish, real-time inference)
- [ ] Implement edge-side rule-based anomaly detection
- [ ] Add `EdgeSimulator` for CPU/memory/latency benchmarking
- [ ] Wire the dashboard's WSS client to the backend API
- [ ] Persist decision attribution logs to MongoDB and surface them in the anomaly log panel

##License
 
