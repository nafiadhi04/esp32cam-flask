from flask import Flask, Response
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import requests
import pytz
import os
import gdown 
from apscheduler.schedulers.background import BackgroundScheduler

# ==== KONFIGURASI YOLO ====
config_path = "cfg/yolov3_custom_training.cfg"
names_path = "data/classes.names"
def download_weights():
    weight_path = "backup/yolov3_custom_training_last.weights"
    if not os.path.exists(weight_path):
        print("📥 Downloading YOLO weights...")
        url = "https://drive.google.com/uc?id=1EiQHNkLz9y3J0h2CpjYWUHXNFOuaCC9N"  # ganti dengan ID kamu
        os.makedirs("backup", exist_ok=True)
        gdown.download(url, weight_path, quiet=False)
conf_threshold = 0.5
nms_threshold = 0.4

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

download_weights()  # ← ini baris penting
net = cv2.dnn.readNet("backup/yolov3_custom_training_last.weights", config_path)
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
except:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ==== GLOBAL VARIABEL ====
app = Flask(__name__)
stream_url = "http://192.168.137.129:81/stream"  # Ganti sesuai ESP32-CAM kamu
latest_frame = None
last_cat_seen = datetime.now()
cat_detection_start = None
cat_detected_total_duration = timedelta()
last_cat_lock = threading.Lock()

# ==== FUNGSI DETEKSI ====
def detect_objects(frame):
    global last_cat_seen, cat_detection_start, cat_detected_total_duration

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    found_cat = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                label = classes[class_id].lower()
                if "cat" in label or "kucing" in label:
                    found_cat = True

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        for i in indices:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    with last_cat_lock:
        if found_cat:
            last_cat_seen = datetime.now()
            if cat_detection_start is None:
                cat_detection_start = datetime.now()
        else:
            if cat_detection_start is not None:
                cat_detected_total_duration += datetime.now() - cat_detection_start
                cat_detection_start = None

    return frame

# ==== MJPEG STREAM READER ====
def stream_reader():
    global latest_frame
    try:
        r = requests.get(stream_url, stream=True)
        byte_data = bytes()

        for chunk in r.iter_content(chunk_size=1024):
            byte_data += chunk
            a = byte_data.find(b'\xff\xd8')  # JPEG start
            b = byte_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = byte_data[a:b+2]
                byte_data = byte_data[b+2:]
                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    processed = detect_objects(frame)
                    latest_frame = processed
    except Exception as e:
        print("🔴 Stream error:", e)

# ==== STREAMING VIDEO ====
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    global latest_frame
    if latest_frame is None:
        return "Belum ada frame", 503
    ret, buffer = cv2.imencode('.jpg', latest_frame)
    if not ret:
        return "Gagal encode", 500
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/last_seen')
def last_seen():
    with last_cat_lock:
        delta = datetime.now() - last_cat_seen
        return str(int(delta.total_seconds()))

@app.route('/detected_duration')
def detected_duration():
    with last_cat_lock:
        duration = cat_detected_total_duration
        if cat_detection_start is not None:
            duration += datetime.now() - cat_detection_start
        return str(int(duration.total_seconds()))

@app.route('/')
def index():
    return '''
    <h2>ESP32-CAM Stream</h2>
    <img src="/video_feed" width="640"><br>
    <a href="/snapshot" target="_blank">📸 Ambil Snapshot</a>
    '''

# ==== RESET HARIAN ====
def reset_daily_stats():
    global cat_detected_total_duration, cat_detection_start
    with last_cat_lock:
        cat_detected_total_duration = timedelta()
        cat_detection_start = None
    print(f"[RESET] Statistik direset {datetime.now().strftime('%H:%M:%S')}")

# ==== SCHEDULER ====
tz = pytz.timezone('Asia/Jakarta')
scheduler = BackgroundScheduler(timezone=tz)
scheduler.add_job(reset_daily_stats, trigger='cron', hour=0, minute=0)
scheduler.start()

# ==== START STREAM DAN FLASK ====
if __name__ == '__main__':
    threading.Thread(target=stream_reader, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
