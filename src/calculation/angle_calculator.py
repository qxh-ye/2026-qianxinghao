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