import cv2
import numpy as np
import tensorflow as tf
import pickle
import csv
import os
import pygame
import requests
import joblib
import cvlib as cv
import hashlib
import os
os.environ["GLOG_minloglevel"] = "2"
import mediapipe as mp
from datetime import datetime
from tensorflow.keras.models import load_model
from threading import Thread
from time import time, sleep
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Imgur Config ===
IMGUR_CLIENT_ID = 'cc71a5b975958d0'  # Replace this!

# === Internet Mode & Pending Files ===
ONLINE_MODE = False
PENDING_ALERTS_FILE = "pending_alerts.txt"
PENDING_UPLOADS_FILE = "pending_uploads.txt"

# === Internet Check ===
def has_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.RequestException:
        return False

def internet_monitor():
    """Background thread to update ONLINE_MODE and process pending tasks."""
    global ONLINE_MODE
    while True:
        connected = has_internet()
        if connected != ONLINE_MODE:
            ONLINE_MODE = connected
            if ONLINE_MODE:
                print("[INFO] Internet connection restored. Online mode enabled.")
                send_pending_uploads()
                send_pending_alerts()
            else:
                print("[WARN] Internet connection lost. Offline mode enabled.")
        sleep(1)  # faster detection

# === Pending Storage ===
def save_pending_alert(message):
    with open(PENDING_ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")
    print(f"[INFO] Alert saved to pending list: {message}")

def save_pending_upload(image_path, alert_msg):
    with open(PENDING_UPLOADS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{image_path}|{alert_msg}\n")
    print(f"[INFO] Image upload saved to pending list: {image_path}")

# === Process Pending ===
def send_pending_alerts():
    if not os.path.exists(PENDING_ALERTS_FILE):
        return
    with open(PENDING_ALERTS_FILE, "r", encoding="utf-8") as f:
        alerts = [line.strip() for line in f if line.strip()]
    if not alerts:
        return
    print(f"[INFO] Sending {len(alerts)} pending alerts...")
    for msg in alerts:
        send_alert(msg, save_if_offline=False)
        send_telegram_alert(msg, save_if_offline=False)
    open(PENDING_ALERTS_FILE, "w").close()

def send_pending_uploads():
    if not os.path.exists(PENDING_UPLOADS_FILE):
        return
    with open(PENDING_UPLOADS_FILE, "r", encoding="utf-8") as f:
        uploads = []
        for line in f:
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                uploads.append(parts)
            else:
                print(f"[WARN] Skipping malformed pending upload entry: {line.strip()}")
    if not uploads:
        return
    print(f"[INFO] Uploading {len(uploads)} pending images...")
    for img_path, msg in uploads:
        if os.path.exists(img_path):
            image_url = upload_to_imgur(img_path)
            full_msg = msg.replace("{image_url}", image_url)
            send_alert(full_msg, save_if_offline=False)
            send_telegram_alert(full_msg, save_if_offline=False)
    open(PENDING_UPLOADS_FILE, "w").close()

# === Upload to Imgur (with offline queuing) ===
def upload_to_imgur(image_path, alert_msg=None):
    if not ONLINE_MODE:
        print("[WARN] Skipping Imgur upload (offline mode).")
        if alert_msg:
            save_pending_upload(image_path, alert_msg)
        return "{image_url}"
    headers = {'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'}
    try:
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                "https://api.imgur.com/3/image",
                headers=headers,
                files={'image': image_file},
                timeout=5
            )
        if response.status_code == 200:
            return response.json()['data']['link']
        else:
            print("[ERROR] Failed to Upload Image to Imgur")
            return "Image upload failed"
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Imgur upload failed: {e}")
        return "Image upload skipped (error)"

# === CallMeBot WhatsApp Config ===
CALLMEBOT_PHONE = '233257269689'
CALLMEBOT_API_KEY = '9626404'

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = '7563143642:AAHBfMuwwUYgBueIG3JZ2A-HcFmkAD2QyUo'
TELEGRAM_CHAT_ID = '754065094'

# === Face Alignment Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def align_face_with_mediapipe(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        return None
    h, w, _ = img.shape
    landmarks = results.multi_face_landmarks[0].landmark
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    avg = (left_eye + right_eye) / 2
    center = (int(avg[0]), int(avg[1]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (w, h))
    return aligned

def play_sound(file_path):
    def task():
        try:
            pygame.mixer.init()
            sound = pygame.mixer.Sound(file_path)
            sound.play()
        except Exception as e:
            print(f"[ERROR] Could Not Play Sound: {e}")
    Thread(target=task).start()

def send_alert(message, save_if_offline=True):
    if not ONLINE_MODE:
        if save_if_offline:
            save_pending_alert(message)
        print("[WARN] Skipping WhatsApp alert (offline mode).")
        return
    def task():
        try:
            encoded_msg = requests.utils.quote(message)
            url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE}&text={encoded_msg}&apikey={CALLMEBOT_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("[INFO] WhatsApp alert sent successfully.")
            else:
                print(f"[ERROR] WhatsApp Failed. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] WhatsApp Error: {e}")
    Thread(target=task).start()

def send_telegram_alert(message, save_if_offline=True):
    if not ONLINE_MODE:
        if save_if_offline:
            save_pending_alert(message)
        print("[WARN] Skipping Telegram alert (offline mode).")
        return
    def task():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            response = requests.post(url, data=payload, timeout=5)
            if response.status_code == 200:
                print("[INFO] Telegram alert sent successfully.")
            else:
                print(f"[ERROR] Telegram Failed. Status: {response.status_code}")
                print("Response:", response.text)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Telegram Error: {e}")
    Thread(target=task).start()

# === Main ===
if __name__ == "__main__":
    ONLINE_MODE = has_internet()
    print("[INFO] Internet check complete. Mode:", "Online" if ONLINE_MODE else "Offline")
    # Process pending alerts/uploads immediately if already online at startup
    if ONLINE_MODE:
        send_pending_uploads()
        send_pending_alerts()
    Thread(target=internet_monitor, daemon=True).start()

    print("[INFO] Loading Embedding Model and SVM Classifier...")
    embedding_model = load_model("model/embedding_model.keras")
    svm_clf = joblib.load("model/svm_classifier.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    AUTHORIZED_NAMES = list(label_encoder.classes_)

    AUTHORIZED_NAMES = ["Jerome"]

    LOG_FILE = "access_log.csv"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Name", "Confidence", "Status"])

    GRANTED_SOUND = "access_granted.wav"
    DENIED_SOUND = "access_denied.wav"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot Open Webcam.")
        exit()
    print("[INFO] Webcam Started. Press 'q' to Quit.")

    print("[INFO] Showing camera preview for 2 seconds...")
    start_time = time()
    while time() - start_time < 3:
        ret, preview_frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Preview - Hold still!", preview_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    max_retries = 3
    retry_count = 0
    face_detected = False

    while retry_count < max_retries and not face_detected:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        faces, confidences = cv.detect_face(frame)
        if len(faces) == 0:
            print(f"[INFO] No face detected. Retrying ({retry_count+1}/{max_retries})...")
            retry_count += 1
            cv2.imshow("Snapshot", frame)
            cv2.waitKey(2000)
        else:
            face_detected = True
            for idx, (startX, startY, endX, endY) in enumerate(faces):
                face = frame[startY:endY, startX:endX]
                aligned = align_face_with_mediapipe(face)
                if aligned is None:
                    print("[INFO] Face alignment failed, skipping this face.")
                    continue
                aligned = cv2.resize(aligned, (160, 160))
                aligned = preprocess_input(aligned.astype(np.float32))
                aligned = aligned.reshape(1, 160, 160, 3)

                embedding = embedding_model.predict(aligned, verbose=0)[0]
                probs = svm_clf.predict_proba([embedding])[0]
                predicted_index = np.argmax(probs)
                confidence = probs[predicted_index]
                name = label_encoder.inverse_transform([predicted_index])[0]

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if confidence > 0.90 and name.strip().lower() in {n.strip().lower() for n in AUTHORIZED_NAMES}:
                    status = f"ACCESS GRANTED: {name}"
                    color = (0, 255, 0)
                    play_sound(GRANTED_SOUND)
                    log_status = "GRANTED"

                    GRANTED_FOLDER = "granted_photos"
                    os.makedirs(GRANTED_FOLDER, exist_ok=True)
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_filename = os.path.join(GRANTED_FOLDER, f"granted_{timestamp_str}.jpg")
                    cv2.imwrite(img_filename, frame)
                    alert_msg = f"✅ Access Granted to {name} at {timestamp}.\nPhoto: {{image_url}}"
                    image_url = upload_to_imgur(img_filename, alert_msg)
                    if image_url != "{image_url}":
                        alert_msg = alert_msg.replace("{image_url}", image_url)
                    send_alert(alert_msg)
                    send_telegram_alert(alert_msg)

                else:
                    status = "ACCESS DENIED"
                    color = (0, 0, 255)
                    play_sound(DENIED_SOUND)
                    log_status = "DENIED"

                    DENIED_FOLDER = "denied_photos"
                    os.makedirs(DENIED_FOLDER, exist_ok=True)
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_filename = os.path.join(DENIED_FOLDER, f"denied_{timestamp_str}.jpg")
                    cv2.imwrite(img_filename, frame)
                    alert_msg = f"🚨 Access Denied to {name} at {timestamp}.\nPhoto: {{image_url}}"
                    image_url = upload_to_imgur(img_filename, alert_msg)
                    if image_url != "{image_url}":
                        alert_msg = alert_msg.replace("{image_url}", image_url)
                    send_alert(alert_msg)
                    send_telegram_alert(alert_msg)

                with open(LOG_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, name, round(confidence, 3), log_status])

                label_text = f"{name} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label_text, (startX, endY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, status, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Face Recognition Result", frame)
            cv2.waitKey(3000)

    if not face_detected:
        print("[INFO] No face detected after 3 attempts. Access Denied.")
        play_sound(DENIED_SOUND)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"🚨 Access Denied. No face detected after multiple attempts at {timestamp}."
        send_alert(alert_msg)
        send_telegram_alert(alert_msg)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam Closed. Exiting Program.")