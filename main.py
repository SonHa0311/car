# main.py
from dinhvi import process_video

if __name__ == "__main__":
    video_path = '/Users/laptopjp/Desktop/VS_Code/HybridAStar/tester1.mp4'  # Đường dẫn đến video của bạn
    car_x, car_y, car_yaw = process_video(video_path)
    
    if car_x is not None and car_y is not None and car_yaw is not None:
        print(f"Car Position: X = {car_x:.3f}, Y = {car_y:.3f}")
        print(f"Car Yaw: {car_yaw:.2f} degrees")
    else:
        print("Không thể xác định vị trí và góc xe.")
