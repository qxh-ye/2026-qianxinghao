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

    # 1. 灰度化
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # 2. 高斯滤波
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 3. 二值化
    _, binary = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY
    )

    # 4. 边缘检测
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
