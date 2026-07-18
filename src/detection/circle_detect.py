import cv2

def detect_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)

    return (
        int(x),
        int(y),
        int(radius)
    )
