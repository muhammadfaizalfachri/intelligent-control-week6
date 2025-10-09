from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan (gunakan raw string!)
detect_rail_lane(
    r"C:\Kuliah\Semester 7\Praktikum Kontrol Cerdas\intelligent-control-week6\Dataset\rail_segmentation\train\images\1000195092_0001-0_jpeg.rf.6ec5ff82bf82f12f153ab6c2de97cae9.jpg"
)