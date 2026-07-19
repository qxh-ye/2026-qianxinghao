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
