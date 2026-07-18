import os
import cv2

def save_processed_images(save_dir, filename, gray, binary, edges):
    cv2.imwrite(
        os.path.join(save_dir, "gray_" + filename),
        gray
    )

    cv2.imwrite(
        os.path.join(save_dir, "binary_" + filename),
        binary
    )

    cv2.imwrite(
        os.path.join(save_dir, "edge_" + filename),
        edges
    )

def draw_contours(image, filtered_contours):
    result = image.copy()
    cv2.drawContours(
        result,
        filtered_contours,
        -1,
        (0, 255, 0),
        2
    )
    return result

def draw_circle(img, circle):
    x, y, r = circle

    cv2.circle(
        img,
        (x, y),
        r,
        (0, 255, 0),
        3
    )
    return img
