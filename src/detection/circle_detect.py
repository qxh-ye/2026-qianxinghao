import cv2

def detect_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)

    return (
        int(x),
        int(y),
        int(radius)
    )


def detect_circle_by_hough(
        gray_image,
        max_size=800,
        dp=1.2,
        min_dist_ratio=0.4,
        param1=100,
        param2=35,
        min_radius_ratio=0.20,
        max_radius_ratio=0.48
):
    """

    :param gray_image:
    :param max_size:
    :param dp:
    :param min_dist_ratio:
    :param param1:
    :param param2:
    :param min_radius_ratio:
    :param max_radius_ratio:
    :return:
    """

    if gray_image is None:
        return None

    height, width = gray_image.shape[:2]

    scale = min(
        1.0,
        max_size / max(height, width)
    )

    if scale < 1.0:
        resized = cv2.resize(
            gray_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = gray_image

    blurred = cv2.GaussianBlur(
        resized,
        (9, 9),
        2
    )

    short_side = min(blurred.shape[:2])

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=short_side * min_dist_ratio,
        param1=param1,
        param2=param2,
        minRadius=int(short_side * min_radius_ratio),
        maxRadius=int(short_side * max_radius_ratio)
    )

    if circles is None:
        return None

    x, y, radius = circles[0][0]

    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round((radius / scale)))
    )

