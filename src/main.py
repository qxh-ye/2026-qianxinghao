import cv2
import os

from pathlib import Path

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

    # 保存
    cv2.imwrite(
        os.path.join(save_dir, "gray_" + filename),
        gray
    )

    cv2.imwrite(
        os.path.join(save_dir, "binary_" + filename),
        binary
    )

    cv2.imwrite(
        os.path.join(save_dir, "edge_" + filename),
        edges
    )

    print(f"{filename} processing finished")

def main():
    raw_dir = Path("data/raw")
    save_dir = Path("data/processed")

    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(raw_dir):
        if img_name.endswith((".jpg", ".png")):
            img_path = os.path.join(raw_dir, img_name)

            process_image(img_path, save_dir)

if __name__ == "__main__":
    main()








