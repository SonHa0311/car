import cv2
import apriltag
import numpy as np
import socket
import json

# Khởi tạo detector
options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)

# Ma trận camera và hệ số méo (đã hiệu chỉnh)
camera_matrix = np.array([[1075.1157056, 0, 919.11051861], 
                          [0, 1080.27946582, 540], 
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([[0.40254436, -1.7533671, -0.01458197, -0.03478158, 4.14821133]])

# Tọa độ thực tế của các tag (mét)
real_world_coords = {
    1: (0, 0),
    2: (0, 3.2),
    3: (4.9, 3.2),
    4: (4.9, 0)
}

# Kết nối TCP với File 2
server_ip = "0.0.0.0"  # IP của máy chủ (File 2)
server_port = 5001        # Cổng kết nối
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Đọc video từ file
cap = cv2.VideoCapture("/Users/laptopjp/Desktop/VS_Code/HybridAStar/self.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    
    image_points = []  # Tọa độ pixel của ID 1-4
    world_points = []  # Tọa độ thực tế của ID 1-4
    id0_pixel = None
    orientation = None

    for tag in results:
        tag_id = tag.tag_id
        if tag_id in real_world_coords:
            image_points.append(tag.center)
            world_points.append(real_world_coords[tag_id])
        elif tag_id == 0:
            id0_pixel = tag.center
        
        # Tính hướng của xe nếu phát hiện ID 0
        if tag_id == 0:
            image_points_tag = np.array(tag.corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(
                np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
                image_points_tag, camera_matrix, dist_coeffs
            )
            if success:
                orientation = rvec[2][0]  # Góc quay quanh trục Z

    if len(image_points) == 4 and id0_pixel is not None:
        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Tính ma trận homography
        H, _ = cv2.findHomography(image_points, world_points)
        
        # Chuyển đổi tọa độ ID0
        id0_homog = np.array([id0_pixel[0], id0_pixel[1], 1], dtype=np.float32).reshape(3, 1)
        real_id0 = np.dot(H, id0_homog)
        real_id0 /= real_id0[2]  # Chuẩn hóa
        real_id0_x, real_id0_y = real_id0[0, 0], real_id0[1, 0]
        
        # Hiển thị tọa độ thực của ID0
        cv2.putText(frame, f"ID0: ({real_id0_x:.2f}, {real_id0_y:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Gửi dữ liệu qua socket
        if orientation is not None:
            data = json.dumps({"x": real_id0_x, "y": real_id0_y, "theta": orientation})
            client_socket.sendall(data.encode('utf-8'))

    cv2.imshow("AprilTag Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client_socket.close()
cv2.destroyAllWindows()
