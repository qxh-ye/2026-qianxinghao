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

# def filter_contours(contours, edges):
#     result = []
#
#     img_area = (edges.shape[0] * edges.shape[1])
#
#     for cnt in contours:
#         # 轮廓面积
#         area = cv2.contourArea(cnt)
#
#         # 面积过滤
#         min_area = img_area * 0.01
#         max_area = img_area * 0.8
#
#
#         # 周长
#         perimeter = cv2.arcLength(cnt, True)
#
#         if perimeter == 0:
#             continue
#
#         circularity = (4 * np.pi * area / perimeter**2)   # 判断圆形度
#
#
#         if(min_area < area < max_area and circularity > 0.5):
#            continue
#
#         x, y, h, w = cv2.boundingRect(cnt)
#
#         ratio = w / h
#
#         if ratio < 0.7 or ratio > 1.3:
#             continue
#
#         result.append(cnt)
#
#     return result

def filter_contours(contours, img_shape):

    h,w = img_shape[:2]

    img_area = h*w

    candidates=[]


    for cnt in contours:

        area=cv2.contourArea(cnt)


        if area <=0:
            continue


        if area < img_area*0.03:
            continue


        if area > img_area*0.9:
            continue


        x,y,ww,hh=cv2.boundingRect(cnt)


        ratio=ww/hh


        if ratio <0.7 or ratio>1.3:
            continue


        perimeter=cv2.arcLength(cnt,True)

        if perimeter==0:
            continue


        circularity=4*np.pi*area/(perimeter*perimeter)


        if circularity<0.6:
            continue


        candidates.append(cnt)
        print(type(cnt), cnt.shape)


    if len(candidates)==0:
        return []


    candidates.sort(
        key=cv2.contourArea,
        reverse=True
    )


    return [candidates[0]]

def get_bounding_box(contours):
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return boxes
