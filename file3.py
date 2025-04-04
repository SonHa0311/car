import socket
import serial

# Cấu hình server TCP
server_ip = "0.0.0.0"  # Lắng nghe trên mọi địa chỉ IP
server_port = 5002

# Cấu hình Serial để giao tiếp với STM32
ser = serial.Serial("/dev/serial0", 115200, timeout=1)  # UART trên Raspberry Pi

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((server_ip, server_port))
        server.listen(1)
        print(f"File 3 đang lắng nghe trên {server_ip}:{server_port}")

        conn, addr = server.accept()
        with conn:
            print(f"Kết nối từ File 2: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break

                command = data.decode('utf-8').strip()
                print(f"Nhận lệnh: {command}")

                # Gửi tín hiệu xuống STM32 qua UART
                ser.write(command.encode())

# Hàm chính
def main():
    try:
        start_server()
    except KeyboardInterrupt:
        print("Dừng chương trình")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
