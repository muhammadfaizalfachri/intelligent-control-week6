import cv2
import numpy as np

def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("canny_result.jpg", edges)

    return "canny_result.jpg"

# Contoh penggunaan
canny_edge_detection(
    r"C:\Kuliah\Semester 7\Praktikum Kontrol Cerdas\intelligent-control-week6\Dataset\rail_segmentation\train\images\1000195092_0001-0_jpeg.rf.6ec5ff82bf82f12f153ab6c2de97cae9.jpg"
)
