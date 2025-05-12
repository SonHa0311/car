import cv2
import numpy as np
import threading
import time
from pyapriltags import Detector
from scipy.spatial.distance import cdist

# ========== GLOBAL ==========
REFERENCE_POINTS = {
    2: (-2.250, 1.350),
    3: (2.25, 1.35),
    4: (2.25, -1.35),
    5: (-2.25, -1.35)
}

shared_data = {
    'H': None,
    'start_sent': False,
    'stop_threads': False
}

lock = threading.Lock()

# ========== HOMOGRAPHY ==========
def get_homography_matrix(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(families="tag25h9")
    src_pts, dst_pts = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        detected_tags = {tag.tag_id: tag.center for tag in tags if tag.tag_id in REFERENCE_POINTS}
        if len(detected_tags) == len(REFERENCE_POINTS):
            for tag_id in [2, 3, 4, 5]:
                src_pts.append(detected_tags[tag_id])
                dst_pts.append(REFERENCE_POINTS[tag_id])
            break

    cap.release()
    if len(src_pts) == len(REFERENCE_POINTS):
        H, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
        return H
    return None

def pixel_to_real(pixel_point, H):
    h_cam, h_xe = 323, 13
    pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1])
    real_homogeneous = np.dot(H, pixel_homogeneous)
    car_x = (real_homogeneous[0] / real_homogeneous[2]) * (h_cam - h_xe) / h_cam
    car_y = (real_homogeneous[1] / real_homogeneous[2]) * (h_cam - h_xe) / h_cam
    return (car_x, car_y)

# ========== THREAD 1: SEND START POSITION ==========
def detect_start_position(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(families="tag25h9")

    while cap.isOpened() and not shared_data['stop_threads']:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            if tag.tag_id == 13:
                tag_x, tag_y = map(int, tag.center)
                with lock:
                    if shared_data['H'] is None:
                        shared_data['H'] = get_homography_matrix(video_path)

                    if shared_data['H'] is not None and not shared_data['start_sent']:
                        car_x, car_y = pixel_to_real((tag_x, tag_y), shared_data['H'])
                        print(f"[THREAD 1] START POSITION: X={car_x:.2f}, Y={car_y:.2f}")
                        shared_data['start_sent'] = True
                        cap.release()
                        return

# ========== THREAD 2: CONTINUOUS TRACKING ==========
def track_position(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(families="tag25h9")

    while cap.isOpened() and not shared_data['stop_threads']:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            if tag.tag_id == 13:
                tag_x, tag_y = map(int, tag.center)

                with lock:
                    if shared_data['H'] is not None and shared_data['start_sent']:
                        car_x, car_y = pixel_to_real((tag_x, tag_y), shared_data['H'])

                        top_left = tag.corners[0]
                        top_right = tag.corners[1]
                        mid_x = (top_left[0] + top_right[0]) / 2
                        mid_y = (top_left[1] + top_right[1]) / 2
                        car_direction = np.array([mid_x - tag_x, mid_y - tag_y])
                        x_axis_vector = np.array(REFERENCE_POINTS[4]) - np.array(REFERENCE_POINTS[5])

                        car_yaw_rad = np.arctan2(
                            car_direction[1] * x_axis_vector[0] - car_direction[0] * x_axis_vector[1],
                            car_direction[0] * x_axis_vector[0] + car_direction[1] * x_axis_vector[1]
                        )
                        car_yaw_deg = np.rad2deg(car_yaw_rad)

                        print(f"[THREAD 2] Feedback → X={car_x:.2f}, Y={car_y:.2f}, Yaw={car_yaw_deg:.2f}")
        time.sleep(0.05)  # Giảm tải CPU

# ========== MAIN ==========
if __name__ == "__main__":
    video_path = "/Users/laptopjp/Desktop/VS_Code/HybridAStar/tester1.mp4"

    t1 = threading.Thread(target=detect_start_position, args=(video_path,))
    t2 = threading.Thread(target=track_position, args=(video_path,))

    t1.start()
    t2.start()

    try:
        while t1.is_alive() or t2.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        shared_data['stop_threads'] = True

    t1.join()
    t2.join()
    print("All threads stopped.")
