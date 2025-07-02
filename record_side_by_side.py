import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from collections import deque, Counter, defaultdict
from datetime import datetime

# =========== Parameters ===========
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.7
NOD_THRESHOLD = 15

# =========== MediaPipe Setup ===========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1

# =========== State (per face) ===========
blink_counter = defaultdict(int)
yawn_counter = defaultdict(int)
nod_counter = defaultdict(int)
prev_y = defaultdict(lambda: None)
emotion_history = defaultdict(lambda: deque(maxlen=15))
neutral_like_sad = defaultdict(int)

# =========== Video Capture ===========
cap_passenger = cv2.VideoCapture(6)
cap_passenger.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_passenger.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap_front = cv2.VideoCapture(0)
cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# =========== Video Writer ===========
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_side_by_side.avi', fourcc, 20.0, (1920, 540))

print("Recording started. Press 'q' to stop.")

while True:
    ret1, frame1 = cap_passenger.read()
    ret2, frame2 = cap_front.read()
    if not ret1 or not ret2:
        break

    # --- Passenger camera: process left lens and run emotion detection ---
    frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
    # --- Crop 20% from all sides of the left lens image ---
    h1, w1, _ = frame1.shape
    left_img1 = frame1[:, :w1 // 2]
    crop_x = int(left_img1.shape[1] * 0.2)
    crop_y = int(left_img1.shape[0] * 0.2)
    crop_w = left_img1.shape[1] - 2 * crop_x
    crop_h = left_img1.shape[0] - 2 * crop_y
    cropped_img1 = left_img1[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    # --- Enhance lighting: histogram equalization on Y channel ---
    yuv = cv2.cvtColor(cropped_img1, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced_img1 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    rgb1 = cv2.cvtColor(enhanced_img1, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb1)
    h, w, _ = enhanced_img1.shape

    display_img1 = enhanced_img1.copy()
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            xs, ys = landmarks[:, 0], landmarks[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            margin = 40
            x_min_c = max(x_min - margin, 0)
            x_max_c = min(x_max + margin, w)
            y_min_c = max(y_min - margin, 0)
            y_max_c = min(y_max + margin, h)
            face_img = left_img1[y_min_c:y_max_c, x_min_c:x_max_c]
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
                emotion = result[0]['dominant_emotion']
            except Exception:
                emotion = "Unknown"
            leftEye = landmarks[LEFT_EYE_IDX]
            rightEye = landmarks[RIGHT_EYE_IDX]
            leftEAR = np.linalg.norm(leftEye[1] - leftEye[5]) + np.linalg.norm(leftEye[2] - leftEye[4])
            leftEAR /= (2.0 * np.linalg.norm(leftEye[0] - leftEye[3]))
            rightEAR = np.linalg.norm(rightEye[1] - rightEye[5]) + np.linalg.norm(rightEye[2] - rightEye[4])
            rightEAR /= (2.0 * np.linalg.norm(rightEye[0] - rightEye[3]))
            ear = (leftEAR + rightEAR) / 2.0
            if emotion == "fear":
                if ear < 0.3:
                    emotion = "neutral"
            emotion_history[face_idx].append(emotion)
            most_common_emotion = Counter(emotion_history[face_idx]).most_common(1)[0][0]
            display_emotion = most_common_emotion
            cv2.rectangle(display_img1, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img1, f"Emotion: {display_emotion}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for (x, y) in landmarks:
                cv2.circle(display_img1, (x, y), 1, (0, 255, 0), -1)

    # --- Front camera: rotate and process left lens only ---
    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    h2, w2, _ = frame2.shape
    left_img2 = frame2[:, :w2 // 2]
    # --- Resize both for side-by-side display ---
    display_img1 = cv2.resize(display_img1, (960, 540))
    left_img2_resized = cv2.resize(left_img2, (960, 540))

    # --- Combine side by side ---
    combined = np.hstack((display_img1, left_img2_resized))

    # --- Add timestamp ---
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(combined, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # --- Show and record ---
    cv2.imshow('Passenger (left) | Front (right)', combined)
    out.write(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_passenger.release()
cap_front.release()
out.release()
cv2.destroyAllWindows()
