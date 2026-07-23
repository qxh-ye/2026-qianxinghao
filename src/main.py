from pathlib import Path

import cv2

from configs.gauge_config import GAUGE_PROFILES
from src.calculation import (
    calculate_gauge_reading,
    calculate_pointer_angle,
    resolve_pointer_direction,
)
from src.detection.circle_detect import (
    detect_circle,
    detect_circle_by_hough,
)
from src.detection.contour_detect import (
    detect_contours,
    filter_contours,
)
from src.detection.pointer_detect import (
    create_pointer_tip_from_angle,
    detect_pointer_candidates,
    determine_pointer_tip,
    estimate_pointer_axis_angle,
    extract_dial_roi,
    select_best_pointer_line,
    select_pointer_direction_by_reach,
)
from src.preprocess.enhancement import process_image
from src.utils.visualize import (
    draw_circle,
    draw_meter_result,
    draw_pointer_candidates,
    draw_pointer_direction,
    draw_selected_pointer_axis,
)


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".png"}
GAUGE_PROFILE_NAME = "pressure_0_1_6_mpa"


def format_angle(angle):
    """将角度转换为便于阅读的日志文本"""
    if angle is None:
        return "None"

    return f"{angle:.1f}°"


def save_image(output_path, image, image_label, image_name):
    """保存结果图，并输出能够对应具体文件的保存状态"""
    success = cv2.imwrite(
        str(output_path),
        image
    )

    status = "成功" if success else "失败"

    print(
        image_name,
        f"{image_label}保存{status}：",
        output_path.name
    )

    return success


def main():
    # ==================== 1. 路径与配置 ====================
    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"
    output_root = base_dir / "data" / "processed"

    preprocess_dir = output_root / "preprocess"
    dial_dir = output_root / "dial"
    pointer_dir = output_root / "pointer"
    result_dir = output_root / "results"

    output_dirs = (
        preprocess_dir,
        dial_dir,
        pointer_dir,
        result_dir,
    )

    for output_dir in output_dirs:
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    gauge_profile = GAUGE_PROFILES[GAUGE_PROFILE_NAME]

    image_paths = sorted(
        path
        for path in raw_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    )

    # ==================== 2. 逐张处理图片 ====================
    for image_path in image_paths:
        image_name = image_path.name

        print()
        print("=" * 60)
        print("开始处理：", image_name)

        # -------------------- 2.1 图像读取与预处理 --------------------
        preprocess_result = process_image(
            str(image_path),
            preprocess_dir
        )

        if preprocess_result is None:
            print(image_name, "预处理失败，跳过当前图片")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(image_name, "原始图片读取失败，跳过当前图片")
            continue

        gray_image = preprocess_result["gray"]
        edges = preprocess_result["edges"]

        # -------------------- 2.2 表盘检测 --------------------
        contours = detect_contours(edges)
        filtered_contours = filter_contours(
            contours,
            image.shape
        )

        print(
            image_name,
            "轮廓数量：",
            f"过滤前={len(contours)}，",
            f"过滤后={len(filtered_contours)}"
        )

        if filtered_contours:
            circle = detect_circle(filtered_contours[0])
            detection_method = "contour"
        else:
            circle = detect_circle_by_hough(gray_image)
            detection_method = "hough"

        if circle is None:
            print(image_name, "没有检测到有效表盘")
            continue

        print(
            image_name,
            "表盘检测方法：", detection_method,
            "圆参数：", circle
        )

        dial_roi, dial_mask = extract_dial_roi(
            gray_image,
            circle,
            radius_scale=1.0
        )

        # draw_circle会直接修改传入图片，因此这里使用副本。
        circle_image = draw_circle(
            image.copy(),
            circle
        )

        save_image(
            dial_dir / f"circle_{image_name}",
            circle_image,
            "表盘圆结果",
            image_name
        )
        save_image(
            dial_dir / f"roi_{image_name}",
            dial_roi,
            "表盘ROI",
            image_name
        )
        save_image(
            dial_dir / f"mask_{image_name}",
            dial_mask,
            "表盘Mask",
            image_name
        )

        # -------------------- 2.3 指针候选线检测 --------------------
        pointer_candidates, pointer_edges = (
            detect_pointer_candidates(
                dial_roi,
                circle
            )
        )

        print(
            image_name,
            "指针候选数量：",
            len(pointer_candidates)
        )

        pointer_candidate_image = draw_pointer_candidates(
            image,
            pointer_candidates,
            circle
        )

        best_pointer_line, best_pointer_score = (
            select_best_pointer_line(
                gray_image,
                pointer_candidates,
                circle
            )
        )

        print(
            image_name,
            "最佳指针轴暗度评分：",
            f"{best_pointer_score:.3f}"
        )

        if best_pointer_line is None:
            print(
                image_name,
                "没有找到有效指针参考线，将继续保存诊断结果"
            )

        # -------------------- 2.4 中心轴与指针方向 --------------------
        raw_pointer_tip = determine_pointer_tip(
            circle,
            best_pointer_line
        )
        raw_pointer_angle = calculate_pointer_angle(
            circle,
            raw_pointer_tip
        )

        pointer_axis_angle = estimate_pointer_axis_angle(
            candidates=pointer_candidates,
            reference_line=best_pointer_line,
            angle_tolerance=5.0
        )

        (
            direction_angle,
            positive_reach,
            negative_reach,
            direction_confident,
        ) = select_pointer_direction_by_reach(
            circle=circle,
            candidates=pointer_candidates,
            axis_angle=pointer_axis_angle,
            reference_angle=raw_pointer_angle,
            direction_angle_tolerance=10.0,
            min_reach_difference=0.08
        )

        print(
            image_name,
            "径向延伸：",
            f"正方向={positive_reach:.3f}，",
            f"反方向={negative_reach:.3f}，",
            f"判断可靠={direction_confident}"
        )
        print(
            image_name,
            "方向分析角度：",
            f"原始端点={format_angle(raw_pointer_angle)}，",
            f"中心轴={format_angle(pointer_axis_angle)}，",
            f"径向初选={format_angle(direction_angle)}"
        )

        pointer_tip = create_pointer_tip_from_angle(
            circle=circle,
            pointer_angle=direction_angle,
            length_scale=0.5
        )

        # -------------------- 2.5 量程校正与读数计算 --------------------
        (
            pointer_tip,
            pointer_angle,
            direction_flipped,
        ) = resolve_pointer_direction(
            circle=circle,
            pointer_tip=pointer_tip,
            pointer_angle=direction_angle,
            start_angle=gauge_profile["start_angle"],
            end_angle=gauge_profile["end_angle"],
            direction=gauge_profile["direction"],
            angle_tolerance=gauge_profile["angle_tolerance"]
        )

        pointer_reading = calculate_gauge_reading(
            pointer_angle=pointer_angle,
            start_angle=gauge_profile["start_angle"],
            end_angle=gauge_profile["end_angle"],
            min_value=gauge_profile["min_value"],
            max_value=gauge_profile["max_value"],
            direction=gauge_profile["direction"],
            angle_tolerance=gauge_profile["angle_tolerance"]
        )

        if direction_flipped:
            print(
                image_name,
                "指针方向已根据有效刻度范围翻转180°"
            )

        if pointer_angle is None:
            print(image_name, "没有检测到有效指针角度")
        else:
            print(
                image_name,
                "最终指针角度：",
                format_angle(pointer_angle)
            )

        if pointer_reading is None:
            print(image_name, "没有得到有效仪表读数")
        else:
            print(
                image_name,
                "最终仪表读数：",
                f"{pointer_reading:.3f}",
                gauge_profile["unit"]
            )

        # -------------------- 2.6 绘制与保存指针结果 --------------------
        selected_pointer_image = draw_selected_pointer_axis(
            image,
            best_pointer_line,
            circle
        )
        pointer_direction_image = draw_pointer_direction(
            image,
            circle,
            pointer_tip,
            pointer_angle
        )
        meter_result_image = draw_meter_result(
            image,
            circle,
            pointer_tip,
            pointer_angle,
            pointer_reading,
            unit=gauge_profile["unit"]
        )

        save_image(
            pointer_dir / f"pointer_edges_{image_name}",
            pointer_edges,
            "指针边缘图",
            image_name
        )
        save_image(
            pointer_dir / f"pointer_candidates_{image_name}",
            pointer_candidate_image,
            "指针候选图",
            image_name
        )
        save_image(
            pointer_dir / f"pointer_selected_{image_name}",
            selected_pointer_image,
            "指针参考轴图",
            image_name
        )
        save_image(
            pointer_dir / f"pointer_direction_{image_name}",
            pointer_direction_image,
            "指针方向图",
            image_name
        )
        save_image(
            result_dir / f"result_{image_name}",
            meter_result_image,
            "最终结果图",
            image_name
        )

        print("处理完成：", image_name)


if __name__ == "__main__":
    main()
