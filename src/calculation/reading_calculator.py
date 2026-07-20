from .angle_calculator import calculate_angle_offset

def calculate_gauge_reading(pointer_angle, start_angle, end_angle, min_value, max_value, direction="clockwise"):
    """
    根据指针角度和仪表量程计算读数
    :param pointer_angle:   当前指针角度
    :param start_angle:     最小刻度对应角度
    :param end_angle:       最大刻度对应角度
    :param min_value:       仪表最小读数
    :param max_value:       仪表最大读数
    :return:
    """

    if pointer_angle is None:
        return None

    if max_value <= min_value:
        raise ValueError("max_value 必须大于 min_value")

    scale_sweep = calculate_angle_offset(
        end_angle,
        start_angle,
        direction
    )

    if scale_sweep == 0:
        raise ValueError("起始角度和结束角度不能相同")

    pointer_sweep = calculate_angle_offset(
        pointer_angle,
        start_angle,
        direction
    )

    if pointer_sweep > scale_sweep:
        return None

    scale_ratio = pointer_sweep / scale_sweep

    reading = min_value + scale_ratio * (max_value - min_value)

    return float(reading)