import os
import cv2
import numpy as np


def save_processed_images(save_dir, filename, gray, binary, edges):
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

def draw_contours(image, filtered_contours):
    result = image.copy()
    cv2.drawContours(
        result,
        filtered_contours,
        -1,
        (0, 255, 0),
        2
    )
    return result

def draw_circle(img, circle):
    x, y, r = circle

    cv2.circle(
        img,
        (x, y),
        r,
        (0, 255, 0),
        3
    )

    return img

def draw_info_panel(image, lines, background_alpha=0.65):
    if len(lines) == 0:
        return image

    result = image.copy()

    image_height, image_width = result.shape[:2]
    short_side = min(image_height, image_width)

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = float(
        np.clip(
            short_side / 800.0,
            0.4,
            1.2
        )
    )

    text_thickness = max(
        1,
        int(round(font_scale * 2))
    )

    padding = max(
        5,
        int(round(short_side * 0.015))
    )

    line_gap = max(
        4,
        int(round(short_side * 0.01))
    )

    text_sizes = [
        cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness
        )
        for text in lines
    ]

    text_width = max(
        size[0][0]
        for size in text_sizes
    )

    text_height = max(
        size[0][1]
        for size in text_sizes
    )

    baseline = max(
        size[1]
        for size in text_sizes
    )

    line_height = text_height + baseline

    panel_width = (
        text_width + padding * 2
    )

    panel_height = (
        padding * 2 + line_height * len(lines) + line_gap * (len(lines) - 1)
    )

    panel_x = padding
    panel_y = padding

    overlay = result.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (
            min(image_width -1, panel_x + panel_width),
            min(image_height - 1, panel_y + panel_height)
        ),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        background_alpha,
        result,
        1.0 - background_alpha,
        0,
        result
    )

    text_x = panel_x + padding
    text_y = panel_y + padding + text_height

    for text in lines:
        cv2.putText(
            result,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

        text_y += line_height + line_gap

    return result


def draw_pointer_candidates(image, candidates, circle):
    """
    在原图上绘制所有指针候选直线
    :param image:
    :param candidates:
    :param circle:
    :return:
    """
    result = image.copy()

    center_x, center_y, _ = circle

    for line in candidates:
        x1, y1, x2, y2 = line

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

    cv2.circle(
        result,
        (center_x, center_y),
        5,
        (255, 0, 0),
        -1
    )
    return result

def draw_selected_pointer_axis(image, pointer_line, circle):
    """
    在原图上绘制最终选择的指针轴
    :param image:
    :param pointer_line:
    :param circle:
    :return:
    """

    result = image.copy()

    if pointer_line is None:
        return result

    center_x, center_y, radius = circle
    x1, y1, x2, y2 = pointer_line

    angle = np.arctan2(
        y2 - y1,
        x2 - x1
    )

    direction_x = np.cos(angle)
    direction_y = np.sin(angle)

    axis_length = radius * 0.85

    start_point = (
        int(center_x - direction_x * axis_length),
        int(center_y - direction_y * axis_length)
    )

    end_point = (
        int(center_x + direction_x * axis_length),
        int(center_y + direction_y * axis_length)
    )

    cv2.line(
        result,
        start_point,
        end_point,
        (0, 255, 0),
        3
    )

    cv2.circle(
        result,
        (center_x, center_y),
        5,
        (255, 0, 0),
        -1
    )

    return result

def draw_pointer_direction(image, circle, pointer_tip, pointer_angle, show_angle=True):
    """
    绘制最终指针方向和角度
    :param image:
    :param circle:
    :param pointer_tip:
    :param pointer_angle:
    :param show_angle
    :return:
    """
    result = image.copy()

    if pointer_tip is None or pointer_angle is None:
        return result

    center_x, center_y, radius = circle
    tip_x, tip_y = pointer_tip

    direction_x = tip_x - center_x
    direction_y = tip_y - center_y

    direction_length = np.hypot(
        direction_x,
        direction_y
    )

    if direction_length == 0:
        return result

    unit_x = direction_x / direction_length
    unit_y = direction_y / direction_length

    arrow_length = radius * 0.85

    arrow_tip = (
        int(center_x + unit_x * arrow_length),
        int(center_y + unit_y * arrow_length)
    )

    cv2.arrowedLine(
        result,
        (center_x, center_y),
        arrow_tip,
        (0, 255, 0),
        3,
        tipLength=0.08
    )

    cv2.circle(
        result,
        (center_x, center_y),
        5,
        (255, 0, 0),
        -1
    )

    if show_angle:
        result = draw_info_panel(
            result,
            [f"Angle: {pointer_angle:.1f} deg"]
        )

    return result


def draw_meter_result(image, circle, pointer_tip, pointer_angle, pointer_reading, unit):
    result = draw_pointer_direction(
        image,
        circle,
        pointer_tip,
        pointer_angle,
        show_angle=False
    )

    center_x, center_y, radius = circle

    cv2.circle(
        result,
        (center_x, center_y),
        radius,
        (0, 255, 0),
        3
    )

    if pointer_reading is not None:
        info_lines = []

        if pointer_angle is not None:
            info_lines.append(f"Angle: {pointer_angle:.1f} deg")

        if pointer_reading is not None:
            info_lines.append(f"Reading: {pointer_reading:.3f} {unit}")

        result = draw_info_panel(
            result,
            info_lines
        )

    return result

