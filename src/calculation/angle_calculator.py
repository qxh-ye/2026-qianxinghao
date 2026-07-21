import numpy as np

def calculate_pointer_angle(circle, pointer_tip):
    """
    计算指针相对于表盘正上方的顺时针角度
    角度定义：
        正上方 = 0度
        右侧  = 90度
        正下方 = 180度
        左侧 = 270度
    :param circle:
    :param pointer_tip:  指针尖端坐标 (tip_x, tip_y)
    :return:
    """

    if pointer_tip is None:
        return None

    center_x, center_y, _ = circle
    tip_x, tip_y = pointer_tip

    delta_x = tip_x - center_x
    delta_y = tip_y - center_y

    angle_radian = np.arctan2(
        delta_x,
        -delta_y
    )

    angle_degree = np.degrees(angle_radian)

    angle_degree = angle_degree % 360.0

    return float(angle_degree)


def calculate_angle_offset(target_angle, start_angle, direction="clockwise"):
    """
    计算从起始角度到目标角度经过的角度
    :param target_angle:    目标角度
    :param start_angle:     起始角度
    :param direction:       clockwise 或 counterclockwise
    :return:
    """

    if direction == "clockwise":
        return (target_angle - start_angle) % 360.0

    if direction == "counterclockwise":
        return (start_angle - target_angle) % 360.0

    raise ValueError(
        "direction 必须是 clockwise 或 counterclockwise"
    )


def is_angle_in_scale(angle, start_angle, end_angle, direction="clockwise", angle_tolerance=5.0):
    """
    判断一个角度是否位于仪表有效刻度范围
    :param angle:           要检查的指针角度
    :param start_angle:     最小刻度角度
    :param end_angle:       最大刻度角度
    :param direction:       刻度增加方向
    :param angle_tolerance: 起止刻度允许的角度误差
    :return:
    """

    if angle is None:
        return False

    if angle_tolerance < 0:
        raise ValueError("angle_tolerance 不能小于 0")

    scale_sweep = calculate_angle_offset(
        end_angle,
        start_angle,
        direction
    )

    if scale_sweep == 0:
        return False

    pointer_sweep = calculate_angle_offset(
        angle,
        start_angle,
        direction
    )

    near_valid_range = (
        pointer_sweep <= scale_sweep + angle_tolerance
    )

    near_start_boundary = (
        pointer_sweep >= 360.0 - angle_tolerance
    )

    return near_valid_range or near_start_boundary


def resolve_pointer_direction(circle, pointer_tip, pointer_angle, start_angle, end_angle, direction="clockwise", angle_tolerance=5.0
):
    """
    根据有效刻度范围修正指针正反方向
    :param circle:          表盘圆信息 (center_x, center_y, radius)
    :param pointer_tip:     当前判断出的指针尖端
    :param pointer_angle:   当前指针角度
    :param start_angle:     最小刻度角度
    :param end_angle:       最大刻度角度
    :param direction:       刻度增加方向
    :param angle_tolerance  翻转检测时容错度数
    :return:
        corrected_tip：  修正后的尖端
        corrected_angle：修正后的角度
        flipped：        是否发生了180度翻转
    """

    if pointer_tip is None or pointer_angle is None:
        return None, None, False

    original_valid = is_angle_in_scale(
        pointer_angle,
        start_angle,
        end_angle,
        direction,
        angle_tolerance=angle_tolerance
    )

    opposite_angle = (pointer_angle + 180.0) % 360.0

    opposite_valid = is_angle_in_scale(
        opposite_angle,
        start_angle,
        end_angle,
        direction,
        angle_tolerance=angle_tolerance
    )

    if not original_valid and opposite_valid:
        center_x, center_y, _ = circle
        tip_x, tip_y = pointer_tip

        corrected_tip = (
            2 * center_x - tip_x,
            2 * center_y - tip_y
        )

        return corrected_tip, opposite_angle, True

    return pointer_tip, pointer_angle, False

