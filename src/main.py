import os
from pathlib import Path

import cv2

from src.preprocess.enhancement import process_image
from src.detection.contour_detect import detect_contours
from src.utils.visualize import draw_contours


def main():
    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 数据路径
    raw_dir = BASE_DIR / "data" / "raw"
    save_dir = BASE_DIR / "data" / "processed"

    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(raw_dir):
        if img_name.endswith((".jpg", ".png")):
            img_path = os.path.join(raw_dir, img_name)

            result = process_image(img_path, save_dir)

            edges = result["edges"]

            contours = detect_contours(edges)

            print(
                img_name,
                "检测轮廓数量:",
                len(contours)
            )

            img = cv2.imread(img_path)

            result_img = draw_contours(img, contours)

            success = cv2.imwrite(
                str(save_dir / f"contour_{img_name}"),
                result_img
            )
            print("保存状态：", success)


if __name__ == "__main__":
    main()
