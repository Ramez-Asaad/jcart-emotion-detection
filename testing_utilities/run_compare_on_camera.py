"""
REAL-TIME CAMERA COMPARISON & TESTING UTILITY

PURPOSE:
Live comparison tool for evaluating image enhancement pipeline effectiveness.
Displays original camera feed vs. processed+emotion-annotated results side-by-side.

CORE FEATURES:
- Real-time video processing through complete enhancement pipeline
- Side-by-side display: Original | Cropped+Enhanced+Annotated
- Visual feedback for algorithm development and demonstration

PROCESSING PIPELINE:
1. find_faces_and_context() - Face region extraction with padding
2. smart_enhance() - Advanced lighting enhancement (histogram eq + CLAHE)  
3. detect_emotions_on_context() - Emotion detection with visual annotation

USAGE:
- Development testing and algorithm tuning
- Research validation and demonstration
- Quality assessment of enhancement improvements
- Standalone utility (not imported by other modules)

RUN: python run_compare_on_camera.py (ESC to exit)
DEPENDENCIES: OpenCV, NumPy, compare_crop_enhance module
"""

import cv2
import os
import sys
import numpy as np
from compare_crop_enhance import find_faces_and_context, smart_enhance, detect_emotions_on_context

def get_processed_and_annotated(frame):
    # Find region of interest (all faces)
    context_img, _, _ = find_faces_and_context(frame, padding_ratio=0.5)
    # Smart image processing on context window
    processed = smart_enhance(context_img)
    # Detect faces and emotions on processed image as a new image
    _, detections, _ = find_faces_and_context(processed, padding_ratio=0)
    emotions = detect_emotions_on_context(processed, detections, (0, 0))
    # Annotate processed image
    annotated = processed.copy()
    for (x, y, w, h, emotion) in emotions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera.")
        sys.exit(1)
    print("Press ESC to exit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break
        annotated = get_processed_and_annotated(frame)
        # Resize original to match processed width if needed
        h0 = frame.shape[0]
        annotated_resized = cv2.resize(annotated, (annotated.shape[1], h0))
        orig_resized = cv2.resize(frame, (annotated.shape[1], h0))
        comparison = np.hstack((orig_resized, annotated_resized))
        cv2.imshow("Original (left) | Cropped+Processed+Emotions (right)", comparison)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()
