import os
import cv2

from src.utils.visualize import save_processed_images

def process_image(img_path, save_dir):
    # 读取图片
    img = cv2.imread(img_path)

    if img is None:
        print(f"读取失败：{img_path}")
        return

    filename = os.path.basename(img_path)
    print("正在处理:", filename)

    #  灰度化
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # 局部直方图增强
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    #  高斯滤波, 去噪
    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    #  二值化
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 边缘检测
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    save_processed_images(
        save_dir,
        filename,
        gray,
        binary,
        edges
    )

    print(f"{filename} processing finished")
