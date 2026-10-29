import json
import time
import numpy as np
import pyautogui
import joblib
import paho.mqtt.client as mqtt
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

TOPIC_TELEMETRY = "airpresenter/kelompok_05/telemetry"
TOPIC_CONTROL   = "airpresenter/kelompok_05/control"
MODEL_FILE = "model.pkl"
try:
    model = joblib.load(MODEL_FILE)
    print(f"[AI] Model '{MODEL_FILE}' berhasil dimuat!")
except Exception as e:
    print(f"[PERINGATAN] '{MODEL_FILE}' belum ada. Jalankan collect_and_train.py dulu! ({e})")
    model = None

air_mouse_active = False
alpha = 0.35
ema_dx, ema_dy = 0.0, 0.0
SENSITIVITY = 25.0

WINDOW_SIZE = 20
window_buffer = []
last_gesture_time = 0
COOLDOWN = 1.2
prev_click_state = False

def extract_features(window):
    data = np.array(window)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    rms = np.sqrt(np.mean(np.square(data), axis=0))
    ptp = np.ptp(data, axis=0)
    return np.concatenate([means, stds, rms, ptp]).reshape(1, -1)

def on_message(client, userdata, msg):
    global air_mouse_active, ema_dx, ema_dy, window_buffer, last_gesture_time, prev_click_state

    try:
        payload = json.loads(msg.payload.decode())
        ax, ay, az = payload["accel"]["x"], payload["accel"]["y"], payload["accel"]["z"]
        gx, gy, gz = payload["gyro"]["x"], payload["gyro"]["y"], payload["gyro"]["z"]
        touch_click = payload.get("touch_click", False)

        omega_z = 0.0 if abs(gz) < 0.12 else gz
        omega_x = 0.0 if abs(gx) < 0.12 else gx

        ema_dx = alpha * omega_z + (1 - alpha) * ema_dx
        ema_dy = alpha * omega_x + (1 - alpha) * ema_dy

        if air_mouse_active:
            move_x = int(-ema_dx * SENSITIVITY)
            move_y = int(-ema_dy * SENSITIVITY)
            if move_x != 0 or move_y != 0:
                pyautogui.moveRel(move_x, move_y)

            if touch_click and not prev_click_state:
                pyautogui.click()
        prev_click_state = touch_click
        window_buffer.append([ax, ay, az, gx, gy, gz])
        if len(window_buffer) > WINDOW_SIZE:
            window_buffer.pop(0)

        now = time.time()
        gyro_energy = np.linalg.norm([gx, gy, gz])
        if len(window_buffer) == WINDOW_SIZE and (now - last_gesture_time > COOLDOWN) and gyro_energy > 0.8:
            if model is not None:
                features = extract_features(window_buffer)
                prediction = model.predict(features)[0]

                if prediction != "idle":
                    last_gesture_time = now
                    print(f"[PREDIKSI AI] Gestur Terdeteksi: {prediction.upper()}")

                    if prediction == "swipe_right":
                        pyautogui.press("right")
                    elif prediction == "swipe_left":
                        pyautogui.press("left")
                    elif prediction == "shake":
                        air_mouse_active = not air_mouse_active
                        print(f"[MODE] Air Mouse: {'AKTIF' if air_mouse_active else 'NONAKTIF'}")

                    status_msg = {
                        "gesture": prediction,
                        "air_mouse": air_mouse_active,
                        "timestamp": int(now * 1000)
                    }
                    client.publish(TOPIC_CONTROL, json.dumps(status_msg))

    except Exception as e:
        print("Error memproses pesan:", e)

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe(TOPIC_TELEMETRY)
    print("=== HOST RECEIVER AIRPRESENTER AKTIF ===")
    print(f"Mendengarkan topik: {TOPIC_TELEMETRY}")
    client.loop_forever()

if __name__ == "__main__":
    main()