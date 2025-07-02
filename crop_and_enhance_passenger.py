import cv2
import numpy as np

# Set input and output video paths
input_path = 'passenger_camera_input.avi'  # Change to your input file or use camera index
output_path = 'passenger_camera_cropped_processed.avi'

# Open input video
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate crop margins (20% on all sides)
crop_x = int(width * 0.2)
crop_y = int(height * 0.2)
crop_w = width - 2 * crop_x
crop_h = height - 2 * crop_y

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

print('Processing video...')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Crop 20% from all sides
    cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    # --- Image processing for lighting robustness ---
    # Convert to YUV and apply histogram equalization to the Y channel
    yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    # Optionally, apply CLAHE for even more robustness
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    # lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    # processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out.write(processed)
    cv2.imshow('Processed Passenger Camera', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('Done. Output saved to', output_path)
