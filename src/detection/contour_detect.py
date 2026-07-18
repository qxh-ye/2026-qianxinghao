import cv2

def detect_contours(edges, min_area=1000):
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            filtered_contours.append(cnt)

    return filtered_contours