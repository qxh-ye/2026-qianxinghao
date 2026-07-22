import cv2
import numpy as np
from numpy import arctan2


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


def calculate_axis_darkness(gray_img, circle, line):
    """
    计算候选直线方向上的暗度评分
    评分越高， 说明该方向包含的深色像素越多， 越可能是指针方向
    :param gray_img:    原始灰度图
    :param circle:      表盘圆信息 (center_x, center_y, radius)
    :param line:        Hough候选线段：(x1, y1, x2, y2)
    :return:
    """

    center_x, center_y, radius = circle
    x1, y1, x2, y2 = line

    angle = arctan2(
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

    axis_mask = np.zeros_like(
        gray_img,
        dtype=np.uint8
    )

    axis_width = max(
        2,
        int(radius * 0.025)
    )

    cv2.line(
        axis_mask,
        start_point,
        end_point,
        255,
        axis_width
    )

    cv2.circle(
        axis_mask,
        (center_x, center_y),
        int(radius * 0.10),
        0,
        -1
    )

    axis_pixels = gray_img[axis_mask > 0]

    if axis_pixels.size == 0:
        return 0.0

    darkness_score = np.mean(255.0 - axis_pixels) / 255.0

    return float(darkness_score)

def select_best_pointer_line(gray_img, candidates, circle):
    """
    根据候选方向的暗度评分， 选择最可能的指针轴
    :param gray_img:
    :param candidates:
    :param circle:
    :return:
    """
    if len(candidates) == 0:
        return None, 0.0

    best_line = None
    best_score = -1.0

    for line in candidates:
        score = calculate_axis_darkness(gray_img, circle, line)

        if score > best_score:
            best_score = score
            best_line = line
    return best_line, best_score

def determine_pointer_tip(circle, pointer_line):
    """
    根据线段两个端点到圆心的距离，判断真正的指针尖端
    :param circle:
    :param pointer_line:    最佳指针候选线 (x1, y1, x2, y2)
    :return:
    """

    if pointer_line is None:
        return None

    center_x, center_y, _ = circle
    x1, y1, x2, y2 = pointer_line

    distance_1 = np.hypot(
        x1 - center_x,
        y1 - center_y
    )

    distance_2 = np.hypot(
        x2 - center_x,
        y2 - center_y
    )

    if distance_1 >= distance_2:
        return x1, y1

    return x2, y2


def calculate_line_axis_angle(line):
    """
    计算Hough线段的轴方向  角度定义：正上方 = 0度  右侧 = 90度
    :param line: Hough线段 (x1, y1, x2, y2)
    :return:
    """
    x1, y1, x2, y2 = line

    delta_x = x2 - x1
    delta_y = y2 - y1

    if delta_x == 0 and delta_y == 0:
        return None

    angle = np.degrees(
        np.arctan2(delta_x, -delta_y)
    )

    axis_angle = angle % 180.0

    return float(axis_angle)


def calculate_axis_angle_difference(angle_1, angle_2):
    """
    计算两条无方向轴线之前的角度差
    :param angle_1:  第一条轴线角度
    :param angle_2:  第二条轴线角度
    :return:
    """

    difference = abs(angle_1 - angle_2) % 180.0

    return min(difference, 180.0 - difference)


def estimate_pointer_axis_angle(candidates, reference_line, angle_tolerance=10.0):
    """
    根据多条Hough候选线估计指针中心轴角度
    :param candidates:      所有通过圆心筛选的候选线
    :param reference_line:  暗度评分最佳的参考线
    :param angle_tolerance: 候选线允许的最大角度差
    :return:
    """

    if reference_line is None:
        return None

    reference_angle = calculate_line_axis_angle(reference_line)

    if reference_angle is None:
        return None

    selected_angles = []
    selected_weights = []

    for line in candidates:
        line_angle = calculate_line_axis_angle(line)

        if line_angle is None:
            continue

        angle_difference = calculate_axis_angle_difference(
            line_angle,
            reference_angle
        )

        if angle_difference > angle_tolerance:
            continue

        x1, y1, x2, y2 = line

        line_length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        selected_angles.append(line_angle)
        selected_weights.append(line_length)

    if len(selected_angles) == 0:
        return reference_angle

    double_angles = np.radians(
        np.array(selected_angles) * 2.0
    )

    weights = np.array(
        selected_weights,
        dtype=np.float64
    )

    mean_cos = np.sum(
        weights * np.cos(double_angles)
    )

    mean_sin = np.sum(
        weights * np.sin(double_angles)
    )

    mean_angle = 0.5 * np.degrees(
        arctan2(mean_sin, mean_cos)
    )

    mean_angle = mean_angle % 180.0

    return float(mean_angle)


def select_pointer_direction_by_reach(
        circle,
        candidates,
        axis_angle,
        reference_angle,
        direction_angle_tolerance=10.0,
        min_reach_difference=0.08
):
    """
    根据Hough候选线在中心轴两侧的径向延伸长度， 判断真实指针尖端方向
    :param circle:  表盘圆信息  (center_x, center_y, radius)
    :param candidates:  通过圆心距离筛选后的Hough候选线
    :param axis_angle:  无方向中心轴角度 一般为0~180度
    :param reference_angle:  原始线段端点得到的参考方向， 当两侧证据接近时作为备用方向
    :param direction_angle_tolerance:   计算方向长度时， 候选线与中心轴允许的最大角度差
    :param min_reach_difference:    两侧归一化延伸长度的最小有效差值
    :return:
            pointer_angle:  最终选择的方向角度
            positive_reach:  axis_angle方向的最大延伸比例
            negative_reach:  axis_angle + 180度方向的最大延伸比例
            direction_confident:    是否根据延伸长度得到明确方向
    """
    if axis_angle is None:
        return reference_angle, 0.0, 0.0, False

    center_x, center_y, radius = circle

    if radius <= 0:
        return reference_angle, 0.0, 0.0, False

    angle_radian = np.radians(axis_angle)

    unit_x = np.sin(angle_radian)
    unit_y = -np.cos(angle_radian)

    positive_reach = 0.0
    negative_reach = 0.0

    for line in candidates:
        line_axis_angle = calculate_line_axis_angle(
            line
        )

        if line_axis_angle is None:
            continue

        angle_difference = calculate_axis_angle_difference(
            line_axis_angle,
            axis_angle
        )

        if angle_difference > direction_angle_tolerance:
            continue

        x1, y1, x2, y2 = line

        endpoints = (
            (x1, y1),
            (x2, y2)
        )

        for point_x, point_y in endpoints:
            vector_x = point_x - center_x
            vector_y = point_y - center_y

            projection = (
                vector_x * unit_x
                + vector_y * unit_y
            )

            normalized_projection = (
                projection / radius
            )

            if normalized_projection >= 0:
                positive_reach = max(
                    positive_reach,
                    normalized_projection
                )
            else:
                negative_reach = max(
                    negative_reach,
                    -normalized_projection
                )

    reach_difference = abs(
        positive_reach - negative_reach
    )

    if reach_difference < min_reach_difference:
        fallback_angle = align_axis_angle_with_reference(
            axis_angle,
            reference_angle
        )

        return (
            fallback_angle,
            positive_reach,
            negative_reach,
            False
        )

    if positive_reach > negative_reach:
        pointer_angle = axis_angle % 360.0
    else:
        pointer_angle = (
            axis_angle + 180.0
        ) % 360.0

    return (
        float(pointer_angle),
        float(positive_reach),
        float(negative_reach),
        True
    )


def align_axis_angle_with_reference(axis_angle, reference_angle):
    """
    根据原始指针方向，为无方向轴角度选择正反方向
    :param axis_angle:      综合后的轴角度， 范围0~180度
    :param reference_angle: 原始尖端角度， 范围0~360度
    :return:                与原始方向更接近的角度
    """

    if axis_angle is None:
        return reference_angle

    if reference_angle is None:
        return axis_angle

    option_1 = axis_angle
    option_2 = (axis_angle + 180.0) % 360.0

    distance_1 = abs(
        (option_1 - reference_angle + 180.0) % 360.0 - 180.0
    )
    distance_2 = abs(
        (option_2 - reference_angle + 180.0) % 360.0 - 180.0
    )

    if distance_1 <= distance_2:
        return float(option_1)

    return float(option_2)


def create_pointer_tip_from_angle(circle, pointer_angle, length_scale=0.5):
    """
    根据角度构造一个用于表示方向的指针尖端
    :param circle:          表盘圆 (center_x, center_y, radius)
    :param pointer_angle:   指针方向角度
    :param length_scale:    构造点距离圆心的半径比例
    :return:                构造出的方向点
    """

    if pointer_angle is None:
        return None

    center_x, center_y, radius = circle

    angle_radian = np.radians(pointer_angle)

    pointer_length = (radius * length_scale)

    tip_x = center_x + (
        np.sin(angle_radian) * pointer_length
    )

    tip_y = center_y - (
        np.cos(angle_radian) * pointer_length
    )

    return int(tip_x), int(tip_y)