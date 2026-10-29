import json
import time
import csv
import os
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

TOPIC_TELEMETRY = "airpresenter/kelompok_08/telemetry"
CSV_FILE = "gesture_dataset.csv"
MODEL_FILE = "model.pkl"
WINDOW_SIZE = 20 

buffer = []

def extract_features(window):
    data = np.array(window)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    rms = np.sqrt(np.mean(np.square(data), axis=0))
    ptp = np.ptp(data, axis=0)
    return np.concatenate([means, stds, rms, ptp])

def on_message(client, userdata, msg):
    global buffer
    try:
        payload = json.loads(msg.payload.decode())
        row = [
            payload["accel"]["x"], payload["accel"]["y"], payload["accel"]["z"],
            payload["gyro"]["x"], payload["gyro"]["y"], payload["gyro"]["z"]
        ]
        buffer.append(row)
        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)
    except Exception:
        pass

def record_dataset():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe(TOPIC_TELEMETRY)
    client.loop_start()

    gestures = ["swipe_right", "swipe_left", "shake", "idle"]
    print("=== PEREKAMAN DATASET GESTUR ===")
    print("Pastikan HP sudah aktif mengirim data MQTT\n")

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [f"f_{i}" for i in range(24)] + ["label"]
            writer.writerow(headers)

        for label in gestures:
            input(f"\nTekan [ENTER] untuk mulai merekam 15 sampel gestur: '{label.upper()}'...")
            for i in range(30):
                print(f"  Siap-siap sampel {i+1}/15 dalam 2 detik...")
                time.sleep(1.5)
                buffer.clear()
                print("  >>> LAKUKAN GERAKAN SEKARANG (1 detik) <<<")
                while len(buffer) < WINDOW_SIZE:
                    time.sleep(0.02)
                features = extract_features(buffer[:WINDOW_SIZE])
                writer.writerow(list(features) + [label])
                print(f"  [Tersimpan] Sampel {i+1} untuk {label}")
    client.loop_stop()
    print(f"\nSelesai! Data tersimpan di {CSV_FILE}")

def train_model():
    print("\n=== MELATIH MODEL k-NN ===")
    data = []
    labels = []
    with open(CSV_FILE, mode="r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            labels.append(row[-1])

    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Akurasi Validasi Model k-NN: {acc:.2f}%")
    knn.fit(X, y)
    joblib.dump(knn, MODEL_FILE)
    print(f"Model berhasil disimpan ke'{MODEL_FILE}'")

if __name__ == "__main__":
    pilihan = input("Pilih mode: (1) Rekam Data & Latih Model, (2) Latih Model Saja dari CSV: ")
    if pilihan == "1":
        record_dataset()
        train_model()
    elif pilihan == "2":
        train_model()