import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from collections import deque, Counter, defaultdict
from datetime import datetime
from face_detection_passenger import analyze_faces_and_draw, get_face_mesh
from face_detection_passenger import smart_enhance
# =========== Parameters ===========
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.7
NOD_THRESHOLD = 15

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

def crop_and_enhance(img, intensity=0.5):
    h, w, _ = img.shape
    left_img = img[:, :w // 2]
    crop_x = int(left_img.shape[1] * 0.2)
    crop_y = int(left_img.shape[0] * 0.2)
    crop_w = left_img.shape[1] - 2 * crop_x
    crop_h = left_img.shape[0] - 2 * crop_y
    cropped = left_img[crop_y:crop_y+crop_h, crop_x+200:crop_x+crop_w+100]
    yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced = smart_enhance(cropped, intensity)  # Use smart_enhance from face_detection_passenger
    return enhanced

def eye_aspect_ratio(eye):
    # EAR calculation for 6 points (MediaPipe indices for left/right eye)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_emotion_and_draw(face_mesh, img, w, h):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    display_img = img.copy()
    # Use a rolling average for EAR to improve drowsiness detection
    if 'ear_history' not in detect_emotion_and_draw.__dict__:
        detect_emotion_and_draw.ear_history = defaultdict(lambda: deque(maxlen=5))
    ear_history = detect_emotion_and_draw.ear_history
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
            face_img = img[y_min_c:y_max_c, x_min_c:x_max_c]
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
                emotion = result[0]['dominant_emotion']
            except Exception:
                emotion = "Unknown"
            leftEye = landmarks[LEFT_EYE_IDX]
            rightEye = landmarks[RIGHT_EYE_IDX]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            # Rolling average for EAR
            ear_history[face_idx].append(ear)
            avg_ear = np.mean(ear_history[face_idx])
            # Debug: print EAR value
            print(f"EAR (face {face_idx}): {ear:.3f}, avg: {avg_ear:.3f}")
            # Drowsiness detection with rolling average
            if avg_ear < EAR_THRESHOLD:
                blink_counter[face_idx] += 1
                if blink_counter[face_idx] >= EAR_CONSEC_FRAMES:
                    cv2.putText(display_img, "\u26A0 Drowsiness Detected!", (x_min, y_max + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                blink_counter[face_idx] = 0
            # --- EMOTION CUSTOM RULES (from latest scripts) ---
            mar = abs(landmarks[13][1] - landmarks[14][1]) / max(abs(landmarks[78][0] - landmarks[308][0]), 1)
            # If 'fear' but mouth and eyes are not wide open, treat as neutral
            if emotion == "fear":
                if mar < 0.5 and ear < 0.3:
                    emotion = "neutral"
            # If 'angry' but eyebrows not lowered or eyes not squinted, treat as neutral
            if emotion == "angry":
                left_brow = landmarks[70][1]
                left_eye_top = landmarks[159][1]
                right_brow = landmarks[300][1]
                right_eye_top = landmarks[386][1]
                if (left_brow - left_eye_top > 45) and (right_brow - right_eye_top > 45):
                    emotion = "neutral"
                if ear > 0.32:
                    emotion = "neutral"
            # If 'surprised' but eyes/mouth not wide open, treat as neutral
            if emotion == "surprised":
                if ear < 0.22 or mar < 0.32:
                    emotion = "neutral"
            # Force 'surprised' if geometry is strongly surprised, even if DeepFace says 'fear'
            if (ear > 0.28 and mar > 0.5) and emotion in ["fear", "neutral"]:
                emotion = "surprised"
            # Restrictive rule: Only set 'angry' if at least three cues are present and DeepFace did not detect a strong emotion
            brow_distance = abs(landmarks[70][0] - landmarks[300][0])
            brow_inner_distance = abs(landmarks[105][0] - landmarks[334][0])
            left_corner_y = landmarks[78][1]
            right_corner_y = landmarks[308][1]
            center_top_y = landmarks[13][1]
            center_bottom_y = landmarks[14][1]
            center_y = (center_top_y + center_bottom_y) / 2
            strong_emotions = ["happy", "surprised", "fear", "disgust", "sad"]
            cues = 0
            if brow_distance < 60:
                cues += 1
            if brow_inner_distance < 30:
                cues += 1
            if mar < 0.28:
                cues += 1
            if left_corner_y >= center_y - 5 or right_corner_y >= center_y - 5:
                cues += 1
            if cues >= 2 and emotion not in strong_emotions:
                emotion = "angry"
            emotion_history[face_idx].append(emotion)
            most_common_emotion = Counter(emotion_history[face_idx]).most_common(1)[0][0]
            display_emotion = most_common_emotion
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"Emotion: {display_emotion}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for (x, y) in landmarks:
                cv2.circle(display_img, (x, y), 1, (0, 255, 0), -1)
    return display_img

def process_front_camera(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h, w, _ = frame.shape
    left_img = frame[:, :w // 2]
    return left_img

def main():
    cap_passenger = cv2.VideoCapture(4)
    cap_passenger.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_passenger.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap_front = cv2.VideoCapture(6)
    # cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create output filename with day and timestamp
    now = datetime.now()
    out_filename = now.strftime('video %d %H-%M-%S.avi')
    out = cv2.VideoWriter(out_filename, fourcc, 20.0, (1920, 540))
    face_mesh = get_face_mesh()
    print(f"Recording started. Saving to {out_filename}. Press 'q' to stop.")
    while True:
        ret1, frame1 = cap_passenger.read()
        #ret2, frame2 = cap_front.read()
        if not ret1:
            break
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
        frame1 = crop_and_enhance(frame1, 1)  # Use smart_enhance from face_detection_passenger
        # Ensure frame1 is processed before analyzing faces
        h, w, _ = frame1.shape
        display_img1 = analyze_faces_and_draw(frame1, face_mesh)
        #frame2_processed = process_front_camera(frame2)
        display_img1 = cv2.resize(display_img1, (960, 540))
        #left_img2_resized = cv2.resize(frame2_processed, (960, 540))
       # combined = np.hstack((display_img1, left_img2_resized))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(display_img1, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Passenger (left) | Front (right)', display_img1)
        out.write(display_img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_passenger.release()
    # cap_front.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
