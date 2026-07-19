import cv2
import numpy as np

def extract_dial_roi(gray_img, circle, radius_scale=1.0):
    """
    :param gray_img: 灰度图，类型为numpy数组
    :param circle:  表盘圆信息，格式为 (center_x, center_y, radius)
    :param radius_scale: ROI半径缩放比例
                         1.0  表示使用完整表盘半径
                         0.9  表示只保留90%的半径范围
    :return: roi_image：后续用于检测指针
             mask：用于检查圆形区域是否正确
    """
    center_x, center_y, radius = circle

    mask_radius = int(radius * radius_scale)

    mask = np.zeros_like(
        gray_img,
        dtype=np.uint8
    )

    cv2.circle(
        mask,
        (center_x, center_y),
        mask_radius,
        255,        # 白色
        -1          # 实心填充
    )

    # 表盘圆内保留原灰度图 圆外为黑色
    roi_image = cv2.bitwise_and(
        gray_img,
        gray_img,
        mask=mask
    )

    return roi_image, mask

def point_to_segment_distance(point, line):
    """
    计算一个点到直线的垂直距离
    :param point:  点坐标， 格式为 (point_x, point_y)
    :param line:   直线坐标， 格式为 (x1, y1, x2, y2)
    :return:  点到直线的垂直距离， 单位为像素
    """
    point_x, point_y = point
    x1, y1, x2, y2 = line

    line_x = x2 - x1
    line_y = y2 - y1

    line_length_squared = (
        line_x * line_x + line_y * line_y
    )

    if line_length_squared == 0:
        return np.hypot(
            point_x - x1,
            point_y - y1
        )

    projection = (
        (point_x - x1) * line_x + (point_y - y1) * line_y
    ) / line_length_squared

    projection = max(
        0.0,
        min(1.0, projection)
    )

    closest_x = x1 + projection * line_x
    closest_y = y1 + projection * line_y

    distance = np.hypot(
        point_x - closest_x,
        point_y - closest_y
    )

    return distance



def detect_pointer_candidates(dial_roi, circle):
    """

    :param dial_roi: 圆形表盘灰度ROI
    :param circle:  表盘圆信息，格式为：(center_x, center_y, radius)
    :return:
    """

    center_x, center_y, radius = circle

    pointer_edges = cv2.Canny(
        dial_roi,
        50,
        150
    )

    inner_mask = np.zeros_like(
        pointer_edges,
        dtype=np.uint8
    )

    cv2.circle(
        inner_mask,
        (center_x, center_y),
        int(radius * 0.90),
        255,
        -1
    )

    pointer_edges = cv2.bitwise_and(
        pointer_edges,
        pointer_edges,
        mask=inner_mask
    )

    min_line_length = max(
        10,
        int(radius * 0.25)
    )

    max_line_gap = max(
        5,
        int(radius * 0.10)
    )

    hough_threshold = max(
        15,
        int(radius * 0.12)
    )

    lines = cv2.HoughLinesP(
        pointer_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    candidates = []

    if lines is None:
        return candidates, pointer_edges

    center_distance_limit = max(
        3,
        radius * 0.08
    )

    for detected_line in lines:
        x1, y1, x2, y2 = detected_line[0]

        line = (x1, y1, x2, y2)

        line_length = np.hypot(x2 - x1, y2 - y1)

        center_distance = point_to_segment_distance(
            (center_x, center_y),
            line
        )

        if (line_length >= min_line_length and center_distance <= center_distance_limit):
            candidates.append(line)

    return candidates, pointer_edges
