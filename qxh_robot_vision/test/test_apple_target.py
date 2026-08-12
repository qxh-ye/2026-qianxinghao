import pytest

from geometry_msgs.msg import PointStamped

from qxh_robot_vision.apple_target_node import (
    create_grasp_pose,
    create_pregrasp_pose,
)


@pytest.fixture
def apple_point():
    point = PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp.sec = 12
    point.header.stamp.nanosec = 345
    point.point.x = 0.778
    point.point.y = -0.167
    point.point.z = 0.550
    return point


def test_target_poses_form_forward_approach_sequence(apple_point):
    pregrasp_pose = create_pregrasp_pose(
        apple_point,
        approach_distance_m=0.25,
    )
    grasp_pose = create_grasp_pose(
        apple_point,
        grasp_offset_m=0.10,
    )

    assert pregrasp_pose.pose.position.x == pytest.approx(0.528)
    assert grasp_pose.pose.position.x == pytest.approx(0.678)
    assert (
        pregrasp_pose.pose.position.x
        < grasp_pose.pose.position.x
        < apple_point.point.x
    )
    assert grasp_pose.pose.position.y == pytest.approx(-0.167)
    assert grasp_pose.pose.position.z == pytest.approx(0.550)


def test_target_poses_keep_header_and_unit_orientation(apple_point):
    pregrasp_pose = create_pregrasp_pose(
        apple_point,
        approach_distance_m=0.25,
    )
    grasp_pose = create_grasp_pose(
        apple_point,
        grasp_offset_m=0.10,
    )

    for target_pose in (pregrasp_pose, grasp_pose):
        assert target_pose.header.frame_id == "base_link"
        assert target_pose.header.stamp.sec == 12
        assert target_pose.header.stamp.nanosec == 345
        assert target_pose.pose.orientation.x == 0.0
        assert target_pose.pose.orientation.y == 0.0
        assert target_pose.pose.orientation.z == 0.0
        assert target_pose.pose.orientation.w == 1.0


@pytest.mark.parametrize(
    "invalid_offset",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_grasp_pose_rejects_invalid_offset(
        apple_point,
        invalid_offset,
):
    with pytest.raises(ValueError):
        create_grasp_pose(
            apple_point,
            grasp_offset_m=invalid_offset,
        )


def test_grasp_pose_rejects_offset_past_apple(apple_point):
    with pytest.raises(ValueError):
        create_grasp_pose(
            apple_point,
            grasp_offset_m=apple_point.point.x,
        )
