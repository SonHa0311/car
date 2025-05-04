import cv2
import numpy as np
from scipy.spatial.distance import cdist
from pyapriltags import Detector

def find_intersection(ray_origin, ray_angle, line_points):
    direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    ray_end = ray_origin + direction * 1000
    distances = cdist(line_points, [ray_end])
    closest_index = np.argmin(distances)
    return line_points[closest_index]

def calculate_angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

REFERENCE_POINTS = {
    2: (-2.250, 1.350),
    3: (2.25, 1.35),
    4: (2.25, -1.35),
    5: (-2.25, -1.35)
}

def get_homography_matrix():
    src_pts = []
    dst_pts = []

    cap = cv2.VideoCapture('/Users/laptopjp/Desktop/VS_Code/HybridAStar/tester1.mp4')
    detector = Detector(families="tag25h9")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        detected_tags = {tag.tag_id: tag.center for tag in tags if tag.tag_id in REFERENCE_POINTS}
        if len(detected_tags) == len(REFERENCE_POINTS):
            for tag_id in [2, 3, 4, 5]:
                if tag_id in detected_tags:
                    src_pts.append(detected_tags[tag_id])
                    dst_pts.append(REFERENCE_POINTS[tag_id])
            break

    cap.release()

    if len(src_pts) == len(REFERENCE_POINTS):
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H
    else:
        print("Could not detect all reference tags!")
        return None

def pixel_to_real(pixel_point, H):
    if H is None:
        return None
    pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1])
    real_homogeneous = np.dot(H, pixel_homogeneous)
    car_x = real_homogeneous[0] / real_homogeneous[2]
    car_y = real_homogeneous[1] / real_homogeneous[2]
    return (car_x, car_y)

def real_to_pixel(real_point, H):
    if H is None:
        return None
    real_homogeneous = np.array([real_point[0], real_point[1], 1])
    H_inv = np.linalg.inv(H)
    pixel_homogeneous = np.dot(H_inv, real_homogeneous)
    pixel_x = pixel_homogeneous[0] / pixel_homogeneous[2]
    pixel_y = pixel_homogeneous[1] / pixel_homogeneous[2]
    return (int(pixel_x), int(pixel_y))

def process_video(video_path, client_socket=None):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(families="tag25h9")

    H = get_homography_matrix()
    if H is None:
        print("Could not initialize coordinate transformation!")
        return

    if not cap.isOpened():
        print("Không thể mở video!")
        return

    random_point_pixel = None
    random_point_real = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal random_point_pixel, random_point_real
        if event == cv2.EVENT_LBUTTONDOWN:
            if random_point_pixel is None:
                random_point_pixel = (x, y)
                real_coords = pixel_to_real((x, y), H)
                if real_coords:
                    random_point_real = real_coords
                    print(f"\nSelected random point:")
                    print(f"Pixel coordinates: ({x}, {y})")
                    print(f"Real coordinates: ({real_coords[0]:.3f}, {real_coords[1]:.3f})")
            else:
                real_coords = pixel_to_real((x, y), H)
                if real_coords:
                    car_x, car_y = real_coords
                    print(f"Clicked point - Pixel: ({x}, {y}) - Real: ({car_x:.3f}, {car_y:.3f})")
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"({car_x:.2f}, {car_y:.2f})",
                               (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.namedWindow("AprilTag Detection")
    cv2.setMouseCallback("AprilTag Detection", mouse_callback)

    print("\nClick anywhere on the frame to select a random point...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        if random_point_pixel:
            x_goal, y_goal = random_point_pixel
            cv2.circle(frame, (x_goal, y_goal), 5, (0, 255, 255), -1)
            if random_point_real:
                cv2.putText(frame, f"Random: ({random_point_real[0]:.2f}, {random_point_real[1]:.2f})",
                           (x_goal + 10, y_goal - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        tags = detector.detect(gray)
        car_x, car_y, car_yaw = None, None, None

        for tag in tags:
            tag_x, tag_y = map(int, tag.center)

            if tag.tag_id == 13:
                real_coords = pixel_to_real((tag_x, tag_y), H)
                if real_coords:
                    car_x, car_y = real_coords
                    print(f"Car (ID:13) - Real: ({car_x:.3f}, {car_y:.3f})")

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

                print(f"Car Yaw: {car_yaw_deg:.2f} degrees ({car_yaw_rad:.2f} radians)")

                arrow_x = int(tag_x + 30 * np.cos(car_yaw_rad))
                arrow_y = int(tag_y + 30 * np.sin(car_yaw_rad))
                cv2.arrowedLine(frame, (tag_x, tag_y), (arrow_x, arrow_y), (0, 255, 0), 2, tipLength=0.3)

            cv2.circle(frame, (tag_x, tag_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}",
                        (tag_x + 10, tag_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video('/Users/laptopjp/Desktop/VS_Code/HybridAStar/tester1.mp4')
