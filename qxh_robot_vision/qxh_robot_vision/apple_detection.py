import math

import cv2


def create_color_masks(bgr_image):
    """根据 HSV 颜色范围创建红色和绿色的掩码"""
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("bgr_image 不能为空")

    if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
        raise ValueError("bgr_image 必须是三通道的 BGR 图像")

    hsv_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2HSV,
    )

    red_mask_1 = cv2.inRange(
        hsv_image,
        (0, 100, 80),
        (10, 255, 255),
    )

    red_mask_2 = cv2.inRange(
        hsv_image,
        (170, 100, 80),
        (179, 255, 255),
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    green_mask = cv2.inRange(
        hsv_image,
        (35, 80, 60),
        (85, 255, 255),
    )

    return {
        "ripe": red_mask,
        "unripe": green_mask,
    }


def clean_mask(mask, kernel_size=5):
    """去除颜色掩膜中的小噪声， 并且填补目标内部的小孔"""
    if mask is None or mask.size == 0:
        raise ValueError("mask 不能为空")

    if len(mask.shape) != 2:
        raise ValueError("mask 必须是单通道图像")

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是大于等于 3 的奇数")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    opened_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return cleaned_mask


def extract_detections(
        mask,
        maturity,
        min_area=500.0,
        min_circularity=0.6,
):
    """从二值掩膜中提取满足面积和圆度条件的目标"""
    if mask is None or mask.size == 0:
        raise ValueError("mask 不能为空")

    if len(mask.shape) != 2:
        raise ValueError("mask 必须是单通道图像")

    if not maturity:
        raise ValueError("maturity 不能为空")

    if min_area <= 0:
        raise ValueError("min_area 必须大于 0")

    if not 0.0 <= min_circularity <= 1.0:
        raise ValueError("min_circularity 必须位于 0 到 1 之间")

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter ** 2)

        if circularity < min_circularity:
            continue

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

        detections.append({
            "maturity": maturity,
            "center": (int(round(center_x)), int(round(center_y))),
            "radius": int(round(radius)),
            "area": area,
            "circularity": circularity,
        })

    detections.sort(
        key=lambda detection: detection["area"],
        reverse=True,
    )

    return detections


def detect_apples(
        bgr_image,
        kernel_size=5,
        min_area=500.0,
        min_circularity=0.6,
):
    """组合颜色分割、掩膜清理和轮廓筛选, 检测图像中的苹果"""
    masks = create_color_masks(bgr_image)

    detections = []

    for maturity, mask in masks.items():
        cleaned_mask = clean_mask(
            mask,
            kernel_size=kernel_size,
        )

        color_detections = extract_detections(
            cleaned_mask,
            maturity=maturity,
            min_area=min_area,
            min_circularity=min_circularity,
        )

        detections.extend(
            color_detections
        )

    detections.sort(
        key=lambda detection: detection["area"],
        reverse=True,
    )

    return detections