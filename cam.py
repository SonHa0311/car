import cv2
import os

# Define the output directory
output_dir = 'output_cam'
output_file = os.path.join(output_dir, 'self.mp4')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30  # Set default FPS

print(f"Camera properties detected: Width={frame_width}, Height={frame_height}, FPS={frame_rate}")

# Define the crop region
x_start, x_end = 230, frame_width  
y_start, y_end = 0, frame_height   
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cropped_width = x_end - x_start
cropped_height = y_end - y_start
out = cv2.VideoWriter(output_file, fourcc, frame_rate, (cropped_width, cropped_height))

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cropped_frame = frame[y_start:y_end, x_start:x_end]

    out.write(cropped_frame)

    cv2.imshow('Recording', cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_file}")
