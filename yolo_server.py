import os, cv2, time, threading, logging, requests
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, Response
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import gdown

# === Telegram Bot ===
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# === KONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN", "ISI_TOKEN_BOT")
CHAT_ID = os.getenv("CHAT_ID", "ISI_CHAT_ID")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
STREAM_URL = os.getenv("ESP_STREAM", "http://192.168.137.129:81/stream")

# === PATH YOLO ===
CFG_PATH = "cfg/yolov3_custom_training.cfg"
NAMES_PATH = "data/classes.names"
WEIGHTS_PATH = "backup/yolov3_custom_training_last.weights"
GDRIVE_ID = "1EiQHNkLz9y3J0h2CpjYWUHXNFOuaCC9N"  # Ganti ID

# === DOWNLOAD WEIGHT (Render tidak bisa commit >100MB) ===
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs("backup", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

# === INIT ===
app = Flask(__name__)
latest_frame = None
last_cat_seen = datetime.now()
cat_detection_start = None
cat_detected_total_duration = timedelta()
last_cat_lock = threading.Lock()

# === Load YOLO ===
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]
download_weights()
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === Deteksi YOLO ===
def detect_objects(frame):
    global last_cat_seen, cat_detection_start, cat_detected_total_duration
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outs = net.forward(output_layers)

    found_cat = False
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                cx, cy, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(cx - w / 2), int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                if "cat" in classes[class_id].lower() or "kucing" in classes[class_id].lower():
                    found_cat = True

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    with last_cat_lock:
        if found_cat:
            last_cat_seen = datetime.now()
            if cat_detection_start is None:
                cat_detection_start = datetime.now()
        else:
            if cat_detection_start:
                cat_detected_total_duration += datetime.now() - cat_detection_start
                cat_detection_start = None

    return frame

# === Stream Reader Thread ===
def stream_reader():
    global latest_frame
    try:
        r = requests.get(STREAM_URL, stream=True)
        byte_data = b""
        for chunk in r.iter_content(1024):
            byte_data += chunk
            a, b = byte_data.find(b'\xff\xd8'), byte_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = byte_data[a:b+2]
                byte_data = byte_data[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    latest_frame = detect_objects(img)
    except Exception as e:
        print("⚠️ Stream error:", e)

# === Flask Route ===
@app.route('/')
def home():
    return '<h1>ESP32-CAM Stream</h1><img src="/video_feed"><br><a href="/snapshot">Snapshot</a>'

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    if latest_frame is None:
        return "No frame yet", 503
    _, buffer = cv2.imencode('.jpg', latest_frame)
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
        if cat_detection_start:
            duration += datetime.now() - cat_detection_start
        return str(int(duration.total_seconds()))

# === Telegram Bot Snapshot ===
logging.basicConfig(level=logging.INFO)
bot = Bot(BOT_TOKEN)
tz = pytz.timezone("Asia/Jakarta")

async def send_snapshot():
    try:
        res = requests.get(f"http://localhost:{FLASK_PORT}/snapshot")
        if res.status_code == 200:
            await bot.send_photo(chat_id=CHAT_ID, photo=res.content, caption="📸 Snapshot Otomatis")
            ls = requests.get(f"http://localhost:{FLASK_PORT}/last_seen").text
            dd = requests.get(f"http://localhost:{FLASK_PORT}/detected_duration").text
            ls_sec = int(ls); dd_sec = int(dd)
            msg = f"⏱ Terakhir terlihat: {ls_sec//3600} jam {ls_sec%3600//60} menit lalu.\n"
            msg += f"🐾 Total deteksi hari ini: {dd_sec//3600} jam {dd_sec%3600//60} menit."
            if ls_sec >= 3*3600:
                msg += "\n⚠️ Kucing tidak terlihat lebih dari 3 jam!"
            await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logging.error("Telegram error: " + str(e))

async def handle_ss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_snapshot()

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot aktif. Gunakan /ss untuk snapshot manual.")

def setup_bot():
    app_bot = ApplicationBuilder().token(BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("ss", handle_ss))
    app_bot.add_handler(CommandHandler("start", handle_start))

    scheduler = BackgroundScheduler(timezone=tz)
    for hour in [6,11,16,20]:
        scheduler.add_job(lambda: app_bot.create_task(send_snapshot()), trigger="cron", hour=hour, minute=0)
    scheduler.start()

    threading.Thread(target=app_bot.run_polling, daemon=True).start()

# === Scheduler Reset Deteksi ===
def reset_daily():
    global cat_detected_total_duration, cat_detection_start
    with last_cat_lock:
        cat_detected_total_duration = timedelta()
        cat_detection_start = None

sched = BackgroundScheduler(timezone=tz)
sched.add_job(reset_daily, trigger='cron', hour=0, minute=0)
sched.start()

# === MAIN ===
if __name__ == '__main__':
    threading.Thread(target=stream_reader, daemon=True).start()
    setup_bot()
    app.run(host=FLASK_HOST, port=FLASK_PORT)
