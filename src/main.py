import os
from pathlib import Path

import cv2

from src.preprocess.enhancement import process_image
from src.detection.contour_detect import detect_contours, filter_contours
from src.utils.visualize import draw_contours, draw_circle
from src.detection.circle_detect import detect_circle


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

            img = cv2.imread(img_path)

            filtered_contours = filter_contours(contours, img.shape)

            if len(filtered_contours) == 0:
                print(
                    img_name,
                    "没有检测到有效表盘"
                )
                continue

            circle = detect_circle(filtered_contours[0])

            print(
                img_name,
                "过滤前测轮廓数量:",
                len(contours),
                "过滤后测轮廓数量:",
                len(filtered_contours)
            )


            result_img = draw_contours(img, filtered_contours)  # draw_contours 检测轮廓
            # result_img = draw_circle(img, circle)       # 画圆

            success = cv2.imwrite(
                str(save_dir / f"contour_{img_name}"),
                result_img
            )
            print("保存状态：", success)


if __name__ == "__main__":
    main()
