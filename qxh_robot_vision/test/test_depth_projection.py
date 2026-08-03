import pytest
import numpy as np

from qxh_robot_vision.depth_projection import pixel_to_camera_point, sample_valid_depth


CAMERA_MATRIX = (
    277.19135641, 0.0, 160.0,
    0.0, 277.19135641, 120.0,
    0.0, 0.0, 1.0,
)


def test_center_pixel_projection_straight_forward():
    point = pixel_to_camera_point(
        160,
        120,
        2.0,
        CAMERA_MATRIX,
    )

    assert point == pytest.approx(
        (0.0, 0.0, 2.0)
    )


def test_right_pixel_has_positive_x():
    point = pixel_to_camera_point(
        180,
        120,
        2.0,
        CAMERA_MATRIX,
    )

    expected_x = 20.0 * 2.0 / 277.19135641

    assert point[0] == pytest.approx(expected_x)
    assert point[1] == pytest.approx(0.0)
    assert point[2] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "invalid_depth",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_depth_is_rejected(invalid_depth):
    with pytest.raises(ValueError):
        pixel_to_camera_point(
            160,
            120,
            invalid_depth,
            CAMERA_MATRIX,
        )


def test_sample_valid_depth_uses_median():
    depth_image = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.8, 2.0, 2.2, 0.0],
            [0.0, 2.0, np.nan, 2.0, 0.0],
            [0.0, 2.2, 2.0, np.inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    depth = sample_valid_depth(
        depth_image,
        2,
        2,
        window_size=3,
    )

    assert depth == pytest.approx(2.0)


def test_sample_valid_depth_handles_image_border():
    depth_image = np.array(
        [
            [1.5, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32, 
    )

    depth = sample_valid_depth(
        depth_image,
        0,
        0,
        window_size=3,
    )

    assert depth == pytest.approx(1.5)

def test_sample_valid_depth_rejects_empty_window():
    depth_image = np.zeros(
        (5, 5),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        sample_valid_depth(
            depth_image,
            2,
            2,
            window_size=3,
        )
