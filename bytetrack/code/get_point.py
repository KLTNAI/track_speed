
import cv2
import numpy as np

video_path = r"E:\Study\deeplearning\final\track_name_deepsort\data\car6.mp4"

points = []

# Kích thước hiển thị mong muốn
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Tính toán tọa độ thực tế trên video gốc
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        
        print(f"Click tại màn hình: [{x}, {y}] -> Tọa độ thực tế: [{real_x}, {real_y}]")
        points.append([real_x, real_y])
        
        # Vẽ điểm lên ảnh hiển thị
        cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Lay toa do PRO', img_resized)

cap = cv2.VideoCapture(video_path)
ret, img = cap.read()

if ret:
    # Lấy kích thước gốc của video
    orig_h, orig_w = img.shape[:2]
    
    # Tính tỉ lệ phóng đại
    scale_x = orig_w / DISPLAY_WIDTH
    scale_y = orig_h / DISPLAY_HEIGHT
    
    print(f"Video gốc: {orig_w}x{orig_h}")
    print(f"Màn hình hiển thị: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"Tỉ lệ Scale: x={scale_x:.2f}, y={scale_y:.2f}")

    # Resize ảnh để hiển thị cho dễ nhìn
    img_resized = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    cv2.imshow('Lay toa do PRO', img_resized)
    cv2.setMouseCallback('Lay toa do PRO', click_event)
    
    print("\n--- HƯỚNG DẪN ---")
    print("Hãy click 4 điểm theo thứ tự: [Trên-Trái] -> [Trên-Phải] -> [Dưới-Phải] -> [Dưới-Trái]")
    print("Nhấn phím bất kỳ để kết thúc.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("COPY DÒNG DƯỚI ĐÂY VÀO FILE speed.py CỦA BẠN:")
    print("="*50)
    print(f"SOURCE = np.array({points})")
    print("="*50)
else:
    print("Không đọc được video!")