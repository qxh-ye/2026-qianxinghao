import cv2
import numpy as np


def detect_contours(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    return contours

def filter_contours(contours, edges):
    result = []

    img_area = (edges.shape[0] * edges.shape[1])

    for cnt in contours:
        # 轮廓面积
        area = cv2.contourArea(cnt)

        # 面积过滤
        if area < img_area * 0.01:
            continue

        # 周长
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area / perimeter**2)   # 判断圆形度

        if circularity < 0.5:
           continue

        result.append(cnt)

    return result

def get_bounding_box(contours):
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return boxes
