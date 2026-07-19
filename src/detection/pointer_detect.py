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
