import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import socket
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from pyapriltags import Detector
from scipy.interpolate import LinearNDInterpolator

# Tọa độ pixel và tọa độ thực (map)
pts_pixel = np.array([[201, 645], [1225, 639], [234, 52], [1258, 45]])
pts_real = np.array([[0, 0], [4.94, 0], [0, 3.2], [4.94, 3.2]])

interpolator_x = LinearNDInterpolator(pts_pixel, pts_real[:, 0])
interpolator_y = LinearNDInterpolator(pts_pixel, pts_real[:, 1])

def pixel_to_real(pixel_coords):
    pixel_coords = np.array(pixel_coords, dtype='float32')
    real_x = interpolator_x(pixel_coords)
    real_y = interpolator_y(pixel_coords)
    # Convert numpy arrays to scalar values
    return float(real_x), float(real_y)

def calculate_angle(p1, p2):
    """ Tính góc giữa hai điểm so với trục x dương """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi  # Đổi sang độ

def send_position(client_socket, real_x, real_y, car_yaw):
    try:
        data = json.dumps({"x": real_x, "y": real_y, "theta": car_yaw})
        client_socket.sendall(data.encode('utf-8'))
        print(f"Đã gửi: {data}")
    except Exception as e:
        print(f"Lỗi gửi dữ liệu: {e}")

def process_video(video_path, client_socket):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(families="tag25h9")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        pts_src = np.float32([[0, height], [width, height], [width * 0.9, height * 0.1], [width * 0.1, height * 0.1]])
        pts_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))
        birdseye_gray = cv2.warpPerspective(gray, matrix, (width, height))

        tags = detector.detect(birdseye_gray)
        car_x, car_y, car_yaw = None, None, None

        for tag in tags:
            tag_x, tag_y = map(int, tag.center)
            real_x, real_y = pixel_to_real((tag_x, tag_y))

            if tag.tag_id == 13:
                top_left = tag.corners[0]  
                top_right = tag.corners[1]  
                mid_x = (top_left[0] + top_right[0]) / 2
                mid_y = (top_left[1] + top_right[1]) / 2
                car_yaw = calculate_angle((tag_x, tag_y), (mid_x, mid_y))
                car_x, car_y = real_x, real_y

                arrow_x = int(tag_x + 30 * np.cos(car_yaw * np.pi / 180))
                arrow_y = int(tag_y + 30 * np.sin(car_yaw * np.pi / 180))
                cv2.arrowedLine(birdseye_frame, (tag_x, tag_y), (arrow_x, arrow_y), (0, 255, 0), 2, tipLength=0.3)

            cv2.circle(birdseye_frame, (tag_x, tag_y), 5, (0, 0, 255), -1)
            cv2.putText(birdseye_frame, f"ID: {tag.tag_id} ({float(real_x):.2f}, {float(real_y):.2f})",
                        (tag_x + 10, tag_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if car_x is not None and car_y is not None and car_yaw is not None:
            send_position(client_socket, car_x, car_y, car_yaw)

        cv2.imshow("AprilTag Detection", frame)
        cv2.imshow("Bird's Eye View", birdseye_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    server_ip = "127.0.0.1"
    server_port = 5001
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    
    process_video("/Users/laptopjp/Desktop/VS_Code/HybridAStar/output5.mp4", client_socket)
    
    client_socket.close()


# 📌 Hàm tải dữ liệu các đường biên đã lưu
def load_lines():
    try:
        with open("lines_data2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        return []


# 📌 Nội suy spline cho đường biên
def interpolate_spline(line, num_points=5000):
    line = np.array(line)
    tck, u = splprep([line[:, 0], line[:, 1]], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T


# 📌 Tìm điểm giao của tia từ trung tâm với đường biên
def find_intersection(ray_origin, ray_angle, line_points):
    direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    ray_end = ray_origin + direction * 1000
    distances = cdist(line_points, [ray_end])
    closest_index = np.argmin(distances)
    return line_points[closest_index]


# 📌 Tính đường trung tâm từ 2 đường biên
def calculate_center_line(line1, line2, center, num_rays=360):
    line1_interp = interpolate_spline(line1)
    line2_interp = interpolate_spline(line2)
    center_points = []
    for angle in np.linspace(0, 2 * np.pi, num_rays):
        p1 = find_intersection(center, angle, line1_interp)
        p2 = find_intersection(center, angle, line2_interp)
        center_point = (p1 + p2) / 2
        center_points.append(center_point)
    return np.array(center_points)

# 📌 Vẽ đường trung tâm và hiển thị AprilTag
def draw_center_line(tag_positions):
    image = cv2.imread("captured_birdseye.png", cv2.IMREAD_GRAYSCALE)
    # cropped_image = image[33:650, 198:1270]
    cropped_image = image[0:1080, 0:1920]

    _, binary_map = cv2.threshold(cropped_image, 175, 255, cv2.THRESH_BINARY_INV)

    all_lines = load_lines()
    if len(all_lines) < 2:
        print("Cần ít nhất 2 đường để vẽ trung tâm!")
        return

    line1, line2 = np.array(all_lines[0]), np.array(all_lines[1])
    center = np.mean(np.vstack((line1, line2)), axis=0)
    center_line = calculate_center_line(line1, line2, center)

    fig, ax = plt.subplots()
    # ax.imshow(binary_map, cmap='gray', extent=[198, 1270, 650, 33])
    ax.imshow(binary_map, cmap='gray', extent=[0, 1980, 1080, 0])


    # Vẽ đường biên
    line1_smooth = interpolate_spline(line1)
    line2_smooth = interpolate_spline(line2)
    ax.plot(line1_smooth[:, 0], line1_smooth[:, 1], color='blue', label="Đường 1")
    ax.plot(line2_smooth[:, 0], line2_smooth[:, 1], color='red', label="Đường 2")
    ax.plot(center_line[:, 0], center_line[:, 1], color='green', linestyle='--', linewidth=2, label="Đường trung tâm")

    # Vẽ các AprilTag
    colors = {1: 'cyan', 2: 'magenta', 3: 'yellow', 4: 'lime', 5: 'orange', 13: 'purple', 12345: 'pink'}
    for tag_id, positions in tag_positions.items():
        if len(positions) > 0:
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], color=colors[tag_id], label=f"Tag {tag_id}")
            ax.plot(positions[:, 0], positions[:, 1], color=colors[tag_id], linestyle='-', linewidth=2)

    ax.legend()
    plt.show()


# 📌 Chạy chương trình
# tag_positions = process_video('/Users/laptopjp/Downloads/analysis/output5.mp4')
# draw_center_line(tag_positions)

