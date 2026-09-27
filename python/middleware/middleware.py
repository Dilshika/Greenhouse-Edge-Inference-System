import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from datetime import datetime, timezone

# Sensor cache (State Management)
sensor_cache = {
    "temperature": None,  
    "humidity": None,
    "co2": None,
}

# MQTT configuration
BROKER = 'localhost'
PORT = 1883 
BASE_TOPIC = 'greenhouse/sensors'  
PUBLISH_TOPIC = 'greenhouse/sensors/telemetry'

def predict(features):
    current_temp = features.get('temperature')
    if current_temp is None:
        return None

    # Simulating a prediction of the temperature 10 minutes in the future
    return round(current_temp + np.random.uniform(-0.5, 0.5), 2)

def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"Middleware connected to MQTT Broker at {BROKER}:{PORT}")
        # Subscribe to all subtopics under greenhouse/sensors
        client.subscribe(f"{BASE_TOPIC}/#")
        print(f"Listening to raw sensors on '{BASE_TOPIC}/#'...\n")
    else:
        print(f"Failed to connect. Return code: {reason_code}")

def on_message(client, userdata, msg):
    topic = msg.topic
    
    # CRITICAL: Prevent the middleware from processing its own predictions!
    if topic == PUBLISH_TOPIC:
        return

    try:
        value = float(msg.payload.decode('utf-8'))
        sensor_name = topic.split("/")[-1]
        
        if sensor_name in sensor_cache:
            sensor_cache[sensor_name] = value
            print(f" [Sensor Input] {sensor_name.upper()}: {value}")
            
    except ValueError:
        pass

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    
    client.on_connect = on_connect
    client.on_message = on_message
    
    # FIXED keepalive timeout back to 60
    client.connect(BROKER, PORT, 60)
    
    # loop_start() runs the MQTT network code in a background thread
    client.loop_start()
    
    print("Edge AI Middleware started. Press Ctrl+C to stop.\n")

    try:
        # The main thread handles the bundling and AI prediction loop
        while True:
            time.sleep(10)  # Aggregate and predict every 10 seconds
            
            # Check if we have received at least some data
            if any(v is not None for v in sensor_cache.values()):
                
                print("\n" + "="*50)
                print("RUNNING AI INFERENCE...")
                
                # 1. Grab current snapshot of sensors
                current_features = sensor_cache.copy()
                
                # 2. Run the ML Model
                predicted_temp = predict(current_features)
                
                # 3. Build the final enriched JSON payload
                payload = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "current_state": current_features,
                    "predictions": {
                        "predicted_temperature_next_step": predicted_temp
                    }
                }
                
                json_payload = json.dumps(payload)
                
                # 4. Publish to the unified telemetry topic for Node.js/Dashboard
                client.publish(PUBLISH_TOPIC, json_payload)
                
                print(f"PUBLISHED TO {PUBLISH_TOPIC}:")
                print(json.dumps(payload, indent=2))
                print("="*50 + "\n")
                
    except KeyboardInterrupt:
        print("\n Middleware shutting down.")
    finally:
        client.loop_stop()
        client.disconnect()