import cv2
import numpy as np

# ĐƯỜNG DẪN VIDEO
video_path = r"E:\Study\deeplearning\final\supervision\examples\speed_estimation\data\car.mp4"

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}]")
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Lay toa do', img)

cap = cv2.VideoCapture(video_path)
ret, img = cap.read()

if ret:
    img = cv2.resize(img, (1280, 720)) # Resize cho dễ nhìn nếu video 4k
    cv2.imshow('Lay toa do', img)
    cv2.setMouseCallback('Lay toa do', click_event)
    print("Hãy click chuột vào 4 góc của một làn đường hoặc vùng hình chữ nhật trên đường...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n--- COPY DÒNG DƯỚI ĐÂY VÀO FILE CODE CHÍNH ---")
    print(f"SOURCE = np.array({points})")
else:
    print("Không đọc được video!")