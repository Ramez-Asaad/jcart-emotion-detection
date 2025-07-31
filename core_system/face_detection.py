"""
DRIVER EMOTION & FATIGUE DETECTION MODULE

Core Functions:
- Real-time driver monitoring via webcam/ZED camera
- Emotion detection (happy, sad, angry, neutral, fear) using DeepFace
- Drowsiness detection through eye aspect ratio (EAR) monitoring
- Yawning detection via mouth aspect ratio (MAR) analysis
- Head nodding detection for fatigue assessment
- Multi-face support with per-face state tracking

Key Features:
- ZED stereo camera integration (uses left image only)
- Custom emotion correction rules to reduce false positives
- Temporal smoothing using emotion history buffers
- Smart sad detection based on mouth corner positions
- Fear emotion filtering (requires 40% occurrence in recent frames)

Usage: Primary driver monitoring system for automotive safety applications.
Runs as standalone real-time monitoring or can be adapted for integration
with vehicle systems. Displays live video feed with overlay warnings.
"""

import sys
import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from collections import deque, Counter, defaultdict

# =========== Parameters ===========
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.7
NOD_THRESHOLD = 15

# =========== MediaPipe Setup ===========
mp_face_mesh = mp.solutions.face_mesh
# Allow up to 5 faces at once
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

# =========== Helper Functions ===========
def eye_aspect_ratio(eye):
    # EAR calculation for 6 points (MediaPipe indices for left/right eye)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio_mediapipe(landmarks):
    # Use outer lip points: 13 (top), 14 (bottom), 78 (left), 308 (right)
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

# =========== Indices for MediaPipe Landmarks ===========
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # 6 points
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # 6 points
MOUTH_IDX = [78, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88]  # Use subset for MAR
NOSE_TIP_IDX = 1  # Nose tip for nod detection

# =========== State (per face) ===========
# Use defaultdicts to keep state for each face index
blink_counter = defaultdict(int)
yawn_counter = defaultdict(int)
nod_counter = defaultdict(int)
prev_y = defaultdict(lambda: None)
emotion_history = defaultdict(lambda: deque(maxlen=15))
neutral_like_sad = defaultdict(int)

# =========== Webcam ===========
# Set ZED camera to highest available resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # ZED default high-res width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # ZED default high-res height
print("Driver monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Rotate the frame 180 degrees to correct for upside-down camera
    # --- Extract left image from ZED stereo frame ---
    h, w, _ = frame.shape
    left_img = frame[:, :w // 2]  # Use left half only
    rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = left_img.shape  # Update shape for downstream code
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Get landmark coordinates
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            # Bounding box
            xs, ys = landmarks[:, 0], landmarks[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            # Crop a margin around the face for emotion detection
            margin = 40
            x_min_c = max(x_min - margin, 0)
            x_max_c = min(x_max + margin, w)
            y_min_c = max(y_min - margin, 0)
            y_max_c = min(y_max + margin, h)
            face_img = left_img[y_min_c:y_max_c, x_min_c:x_max_c]
            # Emotion detection
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
                emotion = result[0]['dominant_emotion']
            except Exception:
                emotion = "Unknown"
            # EAR (eye aspect ratio)
            leftEye = landmarks[LEFT_EYE_IDX]
            rightEye = landmarks[RIGHT_EYE_IDX]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            # MAR (mouth aspect ratio)
            mar = mouth_aspect_ratio_mediapipe(landmarks)
            # Custom rule: If 'fear' but mouth and eyes are not wide open, treat as neutral
            if emotion == "fear":
                if mar < 0.5 and ear < 0.3:
                    emotion = "neutral"
            # Head nod detection
            nose_point = landmarks[NOSE_TIP_IDX]
            if prev_y[face_idx] is not None and abs(prev_y[face_idx] - nose_point[1]) > NOD_THRESHOLD:
                nod_counter[face_idx] += 1
                cv2.putText(frame, f"⚠ Head Nodding!", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            prev_y[face_idx] = nose_point[1]
            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                blink_counter[face_idx] += 1
                if blink_counter[face_idx] >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, f"⚠ Drowsiness Detected!", (x_min, y_max + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                blink_counter[face_idx] = 0
            # Yawning detection
            if mar > MAR_THRESHOLD:
                yawn_counter[face_idx] += 1
                cv2.putText(frame, f"⚠ Yawning!", (x_min, y_max + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            # Emotion correction (smart sad → neutral)
            if emotion == "sad":
                if mar < 0.3 and ear > EAR_THRESHOLD:
                    neutral_like_sad[face_idx] += 1
                else:
                    neutral_like_sad[face_idx] = 0
                if neutral_like_sad[face_idx] >= 5:
                    emotion = "neutral"
            # Custom sad detection: corners below center = sad mouth
            left_corner_y = landmarks[78][1]
            right_corner_y = landmarks[308][1]
            center_top_y = landmarks[13][1]
            center_bottom_y = landmarks[14][1]
            center_y = (center_top_y + center_bottom_y) / 2
            if left_corner_y > center_y and right_corner_y > center_y:
                emotion = "sad"
            # Smoothing: Only show 'fear' if it appears in at least 40% of the last 80 frames
            emotion_history[face_idx].append(emotion)
            fear_count = sum([e == "fear" for e in emotion_history[face_idx]])
            if fear_count < 0.4 * len(emotion_history[face_idx]):
                most_common_emotion = Counter(emotion_history[face_idx]).most_common(1)[0][0]
                display_emotion = most_common_emotion
            else:
                display_emotion = "fear"
            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {display_emotion}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Draw landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    # Show only the left image in the output window
    cv2.imshow("Driver Emotion & Fatigue Monitor", left_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()