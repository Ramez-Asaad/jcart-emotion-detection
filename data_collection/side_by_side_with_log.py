"""
STRUCTURED DATA COLLECTION WITH DUAL-VIEW LOGGING

PURPOSE:
Automated data collection script for generating research datasets with standardized
naming and dual emotion tracking. Creates side-by-side comparison videos with CSV logs.

CORE FUNCTIONALITY:
- Auto-generates weekday-based filenames (e.g., "wed 1 10min.avi")
- Configurable recording duration (VIDEO_DURATION_MIN parameter)
- Dual emotion logging (emotion_1, emotion_2) for multi-face scenarios
- Side-by-side display: Enhanced Detection | Raw Feed
- 5-second interval logging for temporal analysis

PROCESSING PIPELINE:
- Camera capture (index 4) at 1920x1080, rotated 180°
- Crop to left half + additional 20% margin cropping
- smart_enhance() for detection view, raw cropping for comparison
- analyze_faces_and_draw() for real-time emotion detection
- CSV logging with timestamp, emotion_1, emotion_2 columns

SYSTEM INTEGRATION:
- Standalone data collection tool (not imported by other modules)
- Imports from: core_system.face_detection_passenger, core_system.compare_crop_enhance
- Output consumed by: advanced_log_analysis.py for performance metrics
- Primary data collection method

USAGE: Main research data collection script. Generates standardized datasets for analysis.
Run with configurable duration, press 'q' to stop early. Requires manual annotation post-recording.
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
import os
from core_system.face_detection_passenger import analyze_faces_and_draw, get_face_mesh
from core_system.compare_crop_enhance import smart_enhance

# === Configurable video duration (in minutes) ===
VIDEO_DURATION_MIN = 10 # Change as needed

# === Auto-generate video and log file names ===
weekday = datetime.now().strftime('%a').lower()  # e.g., 'mon', 'tue', ...
existing = [f for f in os.listdir('.') if f.startswith(weekday) and f.endswith('.avi')]
video_number = 1
while True:
    candidate = f"{weekday} {video_number} {VIDEO_DURATION_MIN}min.avi"
    if candidate not in existing:
        break
    video_number += 1
video_name = f"{weekday} {video_number} {VIDEO_DURATION_MIN}min.avi"
log_name = f"{weekday} {video_number} {VIDEO_DURATION_MIN}min.log"

# === Video setup ===
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
face_mesh = get_face_mesh()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, 20.0, (1920, 540))

# === Logging setup ===
log_file = open(log_name, 'w')
log_file.write('timestamp,emotion_1,emotion_2\n')

start_time = datetime.now()
last_log_time = start_time

print(f"Recording started. Saving to {video_name}. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Crop the left half only (single camera view)
    h, w, _ = frame.shape
    left_frame = frame[:, :w // 2]
    # Crop further (as in your original crop_and_enhance logic)
    crop_x = int(left_frame.shape[1] * 0.2)
    crop_y = int(left_frame.shape[0] * 0.2)
    crop_w = left_frame.shape[1] - 2 * crop_x
    crop_h = left_frame.shape[0] - 2 * crop_y
    cropped_left = left_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    enhanced_left = smart_enhance(cropped_left, 1)
    frame_proc, detected_emotions = analyze_faces_and_draw(enhanced_left.copy(), face_mesh)
    # detected_emotions should be a list of emotions for each detected face
    if not isinstance(detected_emotions, list):
        detected_emotions = [detected_emotions]
    # Pad to always have two values for logging
    while len(detected_emotions) < 2:
        detected_emotions.append('none')
    emotion_1, emotion_2 = detected_emotions[:2]
    # Raw: use the same cropped region, but without enhancement
    frame_raw = cv2.resize(cropped_left, (960, 540))
    frame_proc = cv2.resize(frame_proc, (960, 540))
    combined = np.concatenate([frame_proc, frame_raw], axis=1)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(combined, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Detection (left) | Raw (right)', combined)
    out.write(combined)

    now = datetime.now()
    if (now - last_log_time).total_seconds() >= 5:
        log_file.write(f'{timestamp},{emotion_1},{emotion_2}\n')
        log_file.flush()
        last_log_time = now

    if (now - start_time) > timedelta(minutes=VIDEO_DURATION_MIN):
        print("Reached requested duration. Stopping.")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")
print(f"Log saved as {log_name}")
print("After recording, please annotate the log file with the actual emotion for each timestamp.")
