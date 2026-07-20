import os
import cv2

from pathlib import Path

from src.preprocess.enhancement import process_image
from src.detection.contour_detect import detect_contours, filter_contours
from src.utils.visualize import draw_contours, draw_circle, draw_pointer_candidates, draw_selected_pointer_axis, draw_pointer_direction
from src.detection.circle_detect import detect_circle
from src.detection.pointer_detect import extract_dial_roi, detect_pointer_candidates, select_best_pointer_line, determine_pointer_tip
from src.calculation.angle_calculator import calculate_pointer_angle

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

            gray_image = result["gray"]

            dial_roi, dial_mask = extract_dial_roi(gray_image, circle, radius_scale=1.0)

            pointer_candidates, pointer_edges = detect_pointer_candidates(dial_roi, circle)

            pointer_candidate_image = draw_pointer_candidates(
                img,
                pointer_candidates,
                circle
            )

            best_pointer_line, best_pointer_score = select_best_pointer_line(
                gray_image,
                pointer_candidates,
                circle
            )

            pointer_tip = determine_pointer_tip(circle, best_pointer_line)

            pointer_angle = calculate_pointer_angle(circle, pointer_tip)

            pointer_direction_image = draw_pointer_direction(
                img,
                circle,
                pointer_tip,
                pointer_angle
            )

            cv2.imwrite(
                str(save_dir / f"pointer_direction_{img_name}"),
                pointer_direction_image
            )

            if pointer_angle is None:
                print(
                    img_name,
                    "没有检测到有效指针角度"
                )
            else:
                print(
                    img_name,
                    "指针角度：",
                    f"{pointer_angle:.1f}°"
                )

            selected_pointer_image = draw_selected_pointer_axis(
                img,
                best_pointer_line,
                circle
            )

            cv2.imwrite(
                str(save_dir / f"pointer_selected_{img_name}"),
                selected_pointer_image
            )

            print(
                img_name,
                "最佳指针轴暗度评分:", f"{best_pointer_score:.3f}"
            )

            cv2.imwrite(
                str(save_dir / f"pointer_edges_{img_name}"),
                pointer_edges
            )

            cv2.imwrite(
                str(save_dir / f"pointer_candidates_{img_name}"),
                pointer_candidate_image
            )
            print(
                img_name,
                "指针候选数量：", len(pointer_candidates)
            )

            roi_success = cv2.imwrite(
                str(save_dir / f"roi_{img_name}"),
                dial_roi
            )

            mask_success = cv2.imwrite(
                str(save_dir / f"mask_{img_name}"),
                dial_mask
            )

            print(
                img_name,
                "ROI保存状态：", roi_success,
                "Mask保存状态：", mask_success
            )

            print(
                img_name,
                "过滤前测轮廓数量:",
                len(contours),
                "过滤后测轮廓数量:",
                len(filtered_contours)
            )


            # result_img = draw_contours(img, filtered_contours)  # draw_contours 检测轮廓
            result_img = draw_circle(img, circle)       # 画圆

            success = cv2.imwrite(
                str(save_dir / f"contour_{img_name}"),
                result_img
            )
            print("保存状态：", success)


if __name__ == "__main__":
    main()
