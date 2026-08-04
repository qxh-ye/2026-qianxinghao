"""提供二维像素到相机三维坐标的投影工具"""
import math
import numpy as np


def is_valid_depth(
        depth_m,
        min_depth_m=0.1,
        max_depth_m=10.0,
):
    """判断米制深度是否有限并位于有效范围内。"""
    try:
        min_depth_m = float(min_depth_m)
        max_depth_m = float(max_depth_m)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "深度范围必须是数值"
        ) from error

    if not math.isfinite(min_depth_m):
        raise ValueError(
            "min_depth_m 必须是有限数值"
        )

    if not math.isfinite(max_depth_m):
        raise ValueError(
            "max_depth_m 必须是有限数值"
        )

    if min_depth_m <= 0.0:
        raise ValueError(
            "min_depth_m 必须大于 0.0"
        )

    if max_depth_m <= min_depth_m:
        raise ValueError(
            "max_depth_m 必须大于 min_depth_m"
        )

    try:
        depth_m = float(depth_m)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(depth_m):
        return False

    return min_depth_m <= depth_m <= max_depth_m


def sample_valid_depth(
        depth_image,
        pixel_x,
        pixel_y,
        window_size=5,
):
    """从目标中心领域中获取可靠深度"""
    if depth_image is None or depth_image.size == 0:
        raise ValueError("depth_image 不能为空")

    if depth_image.ndim != 2:
        raise ValueError("depth_image 必须是单通道深度图")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size 必须是正奇数")

    pixel_x = int(pixel_x)
    pixel_y = int(pixel_y)

    height, width = depth_image.shape

    if not 0 <= pixel_x < width:
        raise ValueError("pixel_x 超出图像范围")

    if not 0 <= pixel_y < height:
        raise ValueError("pixel_y 超出图像范围")

    half_window = window_size // 2

    x_start = max(0, pixel_x - half_window)
    x_end = min(width, pixel_x + half_window + 1)

    y_start = max(0, pixel_y - half_window)
    y_end = min(height, pixel_y + half_window + 1)

    depth_window = depth_image[
        y_start:y_end,
        x_start:x_end,
    ]

    vaild_depths = depth_window[
        np.isfinite(depth_window) & (depth_window > 0.0)
    ]

    if vaild_depths.size == 0:
        raise ValueError("目标领域内没有有效深度")

    return float(np.median(vaild_depths))


def pixel_to_camera_point(
        pixel_x,
        pixel_y,
        depth,
        camera_matrix,
):
    """使用针孔相机模型将像素点投影到相机三维坐标系"""
    if camera_matrix is None or len(camera_matrix) != 9:
        raise ValueError("camera_matrix 必须包含 9 个元素")

    try:
        pixel_x = float(pixel_x)
        pixel_y = float(pixel_y)
        depth = float(depth)

        fx = float(camera_matrix[0])
        fy = float(camera_matrix[4])
        cx = float(camera_matrix[2])
        cy = float(camera_matrix[5])
    except (TypeError, ValueError) as error:
        raise ValueError("像素、深度和相机内参必须是数值") from error

    values = (
        pixel_x,
        pixel_y,
        depth,
        fx,
        fy,
        cx,
        cy,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("输入数据不能包含 NaN 或无穷大")

    if depth <= 0:
        raise ValueError("depth 必须大于 0")

    if fx <= 0 or fy <= 0:
        raise ValueError("相机焦距 fx 和 fy 必须大于 0")

    point_x = (pixel_x - cx) * depth / fx
    point_y = (pixel_y - cy) * depth / fy
    point_z = depth

    return point_x, point_y, point_z