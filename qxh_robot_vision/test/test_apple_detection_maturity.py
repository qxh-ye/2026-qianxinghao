import cv2
import numpy as np
import pytest

from qxh_robot_vision.apple_detection import (
    classify_maturity,
    detect_apples,
    draw_detections,
)


@pytest.mark.parametrize(
    "red_ratio, expected_maturity",
    [
        (1.0, "ripe"),
        (0.80, "ripe"),
        (0.799, "half_ripe"),
        (0.40, "half_ripe"),
        (0.399, "unripe"),
        (0.0, "unripe"),
    ],
)
def test_classify_maturity_uses_red_ratio_thresholds(
        red_ratio,
        expected_maturity,
):
    assert classify_maturity(red_ratio) == expected_maturity


@pytest.mark.parametrize(
    "invalid_ratio",
    [-0.01, 1.01, float("nan"), float("inf"), "invalid"],
)
def test_classify_maturity_rejects_invalid_ratio(invalid_ratio):
    with pytest.raises(ValueError):
        classify_maturity(invalid_ratio)


def create_colored_apple(red_fraction):
    """创建用于颜色比例测试的红绿圆形苹果."""
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    apple_mask = np.zeros((180, 180), dtype=np.uint8)
    cv2.circle(apple_mask, (90, 90), 45, 255, -1)

    image[apple_mask > 0] = (0, 255, 0)

    if red_fraction >= 1.0:
        image[apple_mask > 0] = (0, 0, 255)
    elif red_fraction >= 0.5:
        red_region = np.zeros_like(apple_mask)
        red_region[:, :90] = apple_mask[:, :90]
        image[red_region > 0] = (0, 0, 255)

    return image


@pytest.mark.parametrize(
    "red_fraction, expected_maturity",
    [
        (1.0, "ripe"),
        (0.5, "half_ripe"),
        (0.0, "unripe"),
    ],
)
def test_detect_apples_classifies_mixed_color_regions(
        red_fraction,
        expected_maturity,
):
    detections = detect_apples(
        create_colored_apple(red_fraction),
        kernel_size=3,
        min_area=500.0,
        min_circularity=0.6,
    )

    assert len(detections) == 1
    assert detections[0]["maturity"] == expected_maturity
    assert 0.0 <= detections[0]["red_ratio"] <= 1.0


def test_draw_detections_uses_distinct_maturity_colors():
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    detections = [
        {"maturity": "ripe", "center": (40, 60), "radius": 20},
        {
            "maturity": "half_ripe",
            "center": (120, 60),
            "radius": 20,
        },
        {"maturity": "unripe", "center": (200, 60), "radius": 20},
    ]

    annotated = draw_detections(image, detections)

    assert np.any(np.all(annotated == (0, 0, 255), axis=2))
    assert np.any(np.all(annotated == (0, 165, 255), axis=2))
    assert np.any(np.all(annotated == (0, 255, 0), axis=2))
