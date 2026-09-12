import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timezone

# 1. THE CACHE: Stores the latest reading from each sensor
sensor_cache = {
    "temperature": None,
    "humidity": None,
    "co2": None,
    "radiation": None,
    "outside_temperature": None,
    "outside_humidity": None
}

# 2. THE LISTENER: Runs automatically whenever a message arrives
def on_message(client, userdata, msg):
    try:
        # Get the topic and the value
        topic = msg.topic
        value = float(msg.payload.decode('utf-8'))
        
        # Extract the sensor name from the end of the topic 
        # (e.g., "greenhouse/sensors/temperature" -> "temperature")
        sensor_name = topic.split("/")[-1]
        
        # Update the cache if it's a sensor we care about
        if sensor_name in sensor_cache:
            sensor_cache[sensor_name] = value
            print(f"Received {sensor_name}: {value}")
            
    except ValueError:
        print(f"⚠️ Could not parse value from {msg.topic}")

# 3. SETUP MQTT
broker = "localhost"
port = 1883

client = mqtt.Client()
client.on_message = on_message
client.connect(broker, port, 60)

# Subscribe to ALL individual sensor topics using the '#' wildcard
client.subscribe("greenhouse/sensors/#")

# Start listening in the background
client.loop_start()

print("🚀 Aggregator started. Listening for individual sensors...")
print("Press Ctrl+C to stop.\n")

# 4. THE BUNDLER: Runs in a loop, aggregates data, and publishes
try:
    while True:
        # Wait 10 seconds between aggregations (adjust this to your needs)
        time.sleep(10)
        
        # Check if we have at least SOME data (avoid publishing empty payloads)
        # Note: If a sensor hasn't reported yet, its value will be `None`
        if any(v is not None for v in sensor_cache.values()):
            
            # Create the combined JSON payload
            payload = sensor_cache.copy()
            payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            
            json_payload = json.dumps(payload)
            
            # Publish to a single aggregated topic for your AI model
            client.publish("greenhouse/aggregated/telemetry", json_payload)
            print(f"\n BUNDLED & PUBLISHED: {json_payload}\n")
            
except KeyboardInterrupt:
    print("\n Aggregator stopped.")
finally:
    client.loop_stop()
    client.disconnect()