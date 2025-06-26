import sys
import os
from collections import deque
import bz2
import cv2
import time
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance as dist

# Restart with Python 3.11 if needed
if "Python311" not in sys.executable:
    python311_path = r"C:\Users\ranah\AppData\Local\Programs\Python\Python311\python.exe"
    os.execv(python311_path, [python311_path] + sys.argv)

# =========== MediaPipe ===========
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =========== Parameters ===========
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.5
NOD_THRESHOLD = 15

blink_counter = 0
yawn_counter = 0
nod_counter = 0
prev_y = None
emotion_history = deque(maxlen=10)
neutral_like_sad = 0

# =========== Webcam ===========
cap = cv2.VideoCapture(0)
print("Driver monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # ========== Emotion Detection ==========
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emotion_history.append(emotion)
        common_emotion = max(set(emotion_history), key=emotion_history.count)
    except:
        common_emotion = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            def get_point(idx):
                pt = face_landmarks.landmark[idx]
                return np.array([int(pt.x * w), int(pt.y * h)])

            # Eye aspect ratio
            left_idxs = [33, 160, 158, 133, 153, 144]
            right_idxs = [362, 385, 387, 263, 373, 380]
            left_eye = np.array([get_point(i) for i in left_idxs])
            right_eye = np.array([get_point(i) for i in right_idxs])

            leftEAR = (dist.euclidean(left_eye[1], left_eye[5]) + dist.euclidean(left_eye[2], left_eye[4])) / (2.0 * dist.euclidean(left_eye[0], left_eye[3]))
            rightEAR = (dist.euclidean(right_eye[1], right_eye[5]) + dist.euclidean(right_eye[2], right_eye[4])) / (2.0 * dist.euclidean(right_eye[0], right_eye[3]))
            ear = (leftEAR + rightEAR) / 2.0

            # Mouth aspect ratio
            top = get_point(13)
            bottom = get_point(14)
            left = get_point(78)
            right = get_point(308)
            mar = dist.euclidean(top, bottom) / dist.euclidean(left, right)

            # Head nod detection
            nose = get_point(1)
            if prev_y is not None and abs(prev_y - nose[1]) > NOD_THRESHOLD:
                nod_counter += 1
                cv2.putText(frame, "⚠ Head Nodding!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            prev_y = nose[1]

            # Drowsiness
            if ear < EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "⚠ Drowsiness Detected!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                blink_counter = 0

            # Yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                cv2.putText(frame, "⚠ Yawning!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Emotion fix: sad looks like neutral
            if common_emotion == "sad":
                if mar < 0.3 and ear > EAR_THRESHOLD:
                    neutral_like_sad += 1
                else:
                    neutral_like_sad = 0
                if neutral_like_sad >= 5:
                    common_emotion = "neutral"

            # Draw landmarks
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Show emotion label
    cv2.putText(frame, f"Emotion: {common_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Driver Emotion & Fatigue Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

