import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
from datetime import datetime

# Initialize pygame
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")
siren_sound = pygame.mixer.Sound("siren.wav")

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye landmark indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR function
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Image preprocessing for low light
def enhance_brightness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(enhanced, table)

# Constants
CLOSED_EYES_DURATION = 5
FACE_LOST_DURATION = 5
MAX_WARNINGS = 3
INITIAL_CALIBRATION_FRAMES = 30

# States
eye_closed_start_time = None
face_lost_start_time = None
warning_count = 0
siren_triggered = False
ear_baseline = 0
calibration_count = 0
ear_threshold = None

os.makedirs("alerts", exist_ok=True)
cap = cv2.VideoCapture(0)

print("🚘 Starting Driver Drowsiness Detection...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    frame = enhance_brightness(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(frame_rgb)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results.multi_face_landmarks:
        face_lost_start_time = None
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # EAR Calibration for personalized threshold
        if calibration_count < INITIAL_CALIBRATION_FRAMES:
            ear_baseline += avg_ear
            calibration_count += 1
            cv2.putText(frame, "Calibrating... Please keep eyes open", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Driver Drowsiness Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        elif ear_threshold is None:
            ear_threshold = (ear_baseline / INITIAL_CALIBRATION_FRAMES) * 0.75  # 75% of baseline

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Alerts: {warning_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if avg_ear < ear_threshold:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elapsed = time.time() - eye_closed_start_time
            cv2.putText(frame, "Eyes Closed", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if elapsed >= CLOSED_EYES_DURATION:
                if warning_count < MAX_WARNINGS:
                    cv2.imwrite(f"alerts/drowsy_{timestamp}.png", frame)
                    pygame.mixer.Sound.play(alert_sound)
                    warning_count += 1
                    print(f"⚠️ Alert {warning_count}/3 triggered!")
                    eye_closed_start_time = None
                elif not siren_triggered:
                    pygame.mixer.Sound.play(siren_sound)
                    print("🚨 Vehicle stopped due to drowsiness!")
                    siren_triggered = True
        else:
            eye_closed_start_time = None
            cv2.putText(frame, "Eyes Open", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for x, y in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
    else:
        cv2.putText(frame, "⚠️ Face not detected!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if face_lost_start_time is None:
            face_lost_start_time = time.time()
        elif time.time() - face_lost_start_time > FACE_LOST_DURATION:
            if not siren_triggered:
                pygame.mixer.Sound.play(siren_sound)
                print("🚨 No face detected! Siren triggered.")
                siren_triggered = True
            cv2.imwrite(f"alerts/face_not_detected_{timestamp}.png", frame)

    if warning_count >= MAX_WARNINGS:
        cv2.putText(frame, "🚨 Vehicle stopped due to drowsiness!", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Cleanup
face_mesh.close()