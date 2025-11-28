###BYTETRACK


# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# video_path = os.path.join('.', 'data', 'car.mp4')
# model_path = "yolo11n.pt"  # Dùng Nano hoặc Small

# # Tọa độ vạch đếm 
# LINE_START = (200, 500) 
# LINE_END = (1100, 500)

# cap = cv2.VideoCapture(video_path)
# model = YOLO(model_path)

# # Biến lưu trữ đếm
# total_count_up = 0
# total_count_down = 0
# previous_positions = {} # Lưu vị trí cũ: {track_id: (cx, cy)}

# # HÀM KIỂM TRA CẮT VẠCH (Giữ nguyên logic cũ)
# def ccw(A, B, C):
#     return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# def intersect(A, B, C, D):
#     return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# cv2.namedWindow('ByteTrack Counter', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ByteTrack Counter', 1280, 720)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # conf=0.25: Để thấp một chút để ByteTrack tự xử lý các xe ở xa
#     results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
#                           classes=[2, 3, 5, 7], conf=0.25, verbose=False)

#     # Vẽ vạch
#     cv2.line(frame, LINE_START, LINE_END, (255, 0, 0), 3)

#     # Lấy kết quả từ YOLO
#     if results[0].boxes.id is not None:
#         # Lấy các thông số: boxes (x,y,x,y), ids, class_ids
#         boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
#         # Duyệt qua từng xe
#         for box, track_id, class_id in zip(boxes, track_ids, class_ids):
#             x1, y1, x2, y2 = box
            
#             # Tính tâm hiện tại
#             cx = (x1 + x2) // 2
#             cy = (y1 + y2) // 2
#             current_center = (cx, cy)

#             # LOGIC ĐẾM XE 
#             if track_id in previous_positions:
#                 prev_center = previous_positions[track_id]

#                 if intersect(prev_center, current_center, LINE_START, LINE_END):
#                     if current_center[1] < prev_center[1]: 
#                         total_count_up += 1
#                     else:
#                         total_count_down += 1
                    
#                     cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 5)
#                     print(f"Xe #{track_id} cắt vạch!")

#             previous_positions[track_id] = current_center
            

#             # Vẽ hình
#             label = f"#{track_id} {model.names[class_id]}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#             cv2.circle(frame, current_center, 4, (0, 0, 255), -1)

#     # Hiển thị bảng kết quả
#     cv2.rectangle(frame, (20, 20), (250, 100), (0, 0, 0), -1)
#     cv2.putText(frame, f"Len: {total_count_up}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     cv2.putText(frame, f"Xuong: {total_count_down}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow('ByteTrack Counter', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()










### bytetrack + count + speed
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import supervision as sv
# from collections import defaultdict, deque

# VIDEO_PATH = r"E:\Study\deeplearning\final\supervision\examples\speed_estimation\data\car.mp4"
# MODEL_PATH = "yolo11n.pt"  # Dùng Nano cho mượt, máy khỏe thì đổi thành "yolo11s.pt"

# # TỌA ĐỘ VÙNG CHỌN
# # Dùng tool get_points.py để lấy 4 điểm này trên video CỦA BẠN.
# # Thứ tự: [Góc trên-trái, Góc trên-phải, Góc dưới-phải, Góc dưới-trái]
# SOURCE = np.array([
#     [1252, 787], 
#     [2298, 803], 
#     [5039, 2159], 
#     [-550, 2159]
# ])

# #Chiều dài, chiều rộng làn xe. Tùy thuộc vào từng video mà điều chỉnh
# #1 làn xe, thì chiều rộng thường là 3.5 mét.
# #2 làn xe, chiều rộng khoảng 7 - 8 mét.
# TARGET_WIDTH = 15
# #chiều dài của đoạn cần đo tốc độ
# TARGET_HEIGHT = 36

# TARGET = np.array([
#     [0, 0],
#     [TARGET_WIDTH - 1, 0],
#     [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
#     [0, TARGET_HEIGHT - 1],
# ])


# class ViewTransformer:
#     def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
#         source = source.astype(np.float32)
#         target = target.astype(np.float32)
#         self.m = cv2.getPerspectiveTransform(source, target)

#     def transform_points(self, points: np.ndarray) -> np.ndarray:
#         if points.size == 0:
#             return points
#         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
#         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
#         return transformed_points.reshape(-1, 2)

# def main():
#     # 1. Khởi tạo Model và Video Info
#     video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
#     model = YOLO(MODEL_PATH)

#     # 2. Khởi tạo ByteTrack
#     # track_activation_threshold=0.25: Giúp bắt xe ở xa/mờ tốt hơn
#     byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.25)

#     # 3. Các công cụ vẽ (Annotators)
#     thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
#     text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
#     box_annotator = sv.BoxAnnotator(thickness=thickness)
#     label_annotator = sv.LabelAnnotator(
#         text_scale=text_scale, 
#         text_thickness=thickness, 
#         text_position=sv.Position.BOTTOM_CENTER
#     )
#     trace_annotator = sv.TraceAnnotator(
#         thickness=thickness, 
#         trace_length=video_info.fps * 2, 
#         position=sv.Position.BOTTOM_CENTER
#     )

#     # 4. Công cụ tính toán tốc độ
#     frame_generator = sv.get_video_frames_generator(source_path=VIDEO_PATH)
#     polygon_zone = sv.PolygonZone(polygon=SOURCE)
#     view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
#     # Lưu trữ tọa độ để tính vận tốc
#     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

#     # Cài đặt cửa sổ hiển thị
#     cv2.namedWindow('ByteTrack', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('ByteTrack', 1280, 720)

#     print("Đang chạy... Nhấn 'q' để thoát.")

#     # 5. Vòng lặp chính
#     for frame in frame_generator:
#         # Nhận diện YOLO
#         result = model(frame, verbose=False)[0]
#         detections = sv.Detections.from_ultralytics(result)

#         # Lọc kết quả
#         detections = detections[detections.confidence > 0.3] # Chỉ lấy độ tin cậy > 0.3
        
#         # Chỉ xử lý xe nằm trong vùng SOURCE (Polygon)
#         # Nếu muốn đo cả xe ngoài vùng thì bỏ dòng này đi
#         detections = detections[polygon_zone.trigger(detections)] 
        
#         detections = detections.with_nms(threshold=0.7)
        
#         # CẬP NHẬT TRACKER 
#         detections = byte_track.update_with_detections(detections=detections)

#         # TÍNH TOÁN TỐC ĐỘ 
#         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
#         points = view_transformer.transform_points(points=points).astype(int)

#         for tracker_id, [_, y] in zip(detections.tracker_id, points):
#             coordinates[tracker_id].append(y)

#         labels = []
#         for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
#             class_name = model.names[class_id] # Lấy tên xe (car, bus...)
            
#             if len(coordinates[tracker_id]) < video_info.fps / 2:
#                 labels.append(f"#{tracker_id} {class_name}")
#             else:
#                 # Công thức tính vận tốc: v = s / t
#                 coordinate_start = coordinates[tracker_id][-1]
#                 coordinate_end = coordinates[tracker_id][0]
#                 distance = abs(coordinate_start - coordinate_end)
#                 time = len(coordinates[tracker_id]) / video_info.fps
#                 speed = distance / time * 3.6 # Đổi m/s sang km/h
                
#                 labels.append(f"#{tracker_id} {class_name} {int(speed)} km/h")

        
#         annotated_frame = frame.copy()
        
#         # Vẽ vùng chọn SOURCE để căn chỉnh
#         cv2.polylines(annotated_frame, [SOURCE.astype(np.int32)], True, (0, 0, 255), 2)

#         annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
#         annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
#         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

#         # Hiển thị
#         cv2.imshow("ByteTrack", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
    
    
    
    
    
    
    
    
### speed trau bo 
import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch


# 1. Tọa độ vùng chọn 
SOURCE = np.array([[160, 226], [550, 230], [614, 710], [5, 696]])

#Chiều dài, chiều rộng làn xe. Tùy thuộc vào từng video mà điều chỉnh
#1 làn xe, thì chiều rộng thường là 3.5 mét.
#2 làn xe, chiều rộng khoảng 7 - 8 mét.
TARGET_WIDTH = 15
#chiều dài của đoạn cần đo tốc độ
TARGET_HEIGHT = 36

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle Speed Estimation")
    parser.add_argument(
        "--source_video_path",
        default=r"E:\Study\deeplearning\final\track_name_deepsort\data\car6.mp4",
        help="Path to source video",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=r"E:\Study\deeplearning\final\track_name_deepsort\output_speed_gpu.mp4",
        help="Path to output video",
        type=str,
    )
    # Tăng confidence lên 0.4 vì model Large rất tự tin, lọc bớt nhiễu
    parser.add_argument("--confidence_threshold", default=0.4, type=float)
    parser.add_argument("--iou_threshold", default=0.7, type=float)

    return parser.parse_args()

if __name__ == "__main__":
    
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️ CẢNH BÁO: Đang chạy bằng CPU! Hãy cài lại PyTorch bản CUDA để tận dụng RTX 3050 Ti.")
        device = 'cpu'
    

    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    
    
    print("Đang tải model YOLO11 Large...")
    model = YOLO("yolo11l.pt") 
    model.to(device) # Đẩy model vào GPU

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, 
        track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    cv2.namedWindow("Speed Estimation GPU", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Speed Estimation GPU", 1280, 720)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            
            # Chạy nhận diện 
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Lọc xe cộ
            detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                class_name = model.names[class_id]
                
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} {class_name}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            cv2.polylines(annotated_frame, [SOURCE.astype(np.int32)], True, (0, 0, 255), 2)
            
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            sink.write_frame(annotated_frame)
            
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("Speed Estimation GPU", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()