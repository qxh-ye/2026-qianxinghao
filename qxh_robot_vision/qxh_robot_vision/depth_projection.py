"""提供二维像素到相机三维坐标的投影工具"""
import math


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