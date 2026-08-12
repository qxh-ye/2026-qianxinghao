import pytest

from geometry_msgs.msg import PoseStamped
from qxh_robot_vision.apple_moveit_planner import (
    create_fixed_place_pose,
    get_grasp_result_action,
    get_next_motion_phase,
)


@pytest.mark.parametrize(
    "current_phase, expected_next_phase",
    [
        ("pregrasp", "grasp"),
        ("grasp", "retreat"),
        ("retreat", "place"),
        ("place", None),
    ],
)
def test_motion_phase_sequence(
        current_phase,
        expected_next_phase,
):
    assert get_next_motion_phase(
        current_phase
    ) == expected_next_phase


def test_motion_phase_sequence_rejects_unknown_phase():
    with pytest.raises(ValueError):
        get_next_motion_phase("unknown")


def test_fixed_place_pose_uses_configured_position_and_reference():
    reference_pose = PoseStamped()
    reference_pose.header.stamp.sec = 12
    reference_pose.header.stamp.nanosec = 34
    reference_pose.header.frame_id = "base_link"
    reference_pose.pose.orientation.x = 0.1
    reference_pose.pose.orientation.y = 0.2
    reference_pose.pose.orientation.z = 0.3
    reference_pose.pose.orientation.w = 0.9

    place_pose = create_fixed_place_pose(
        reference_pose=reference_pose,
        place_x_m=0.45,
        place_y_m=0.35,
        place_z_m=0.55,
    )

    assert place_pose.header.stamp == reference_pose.header.stamp
    assert place_pose.header.frame_id == "base_link"
    assert place_pose.pose.position.x == pytest.approx(0.45)
    assert place_pose.pose.position.y == pytest.approx(0.35)
    assert place_pose.pose.position.z == pytest.approx(0.55)
    assert (
        place_pose.pose.orientation
        == reference_pose.pose.orientation
    )


@pytest.mark.parametrize(
    "invalid_coordinate",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_fixed_place_pose_rejects_invalid_coordinates(
        invalid_coordinate,
):
    reference_pose = PoseStamped()
    reference_pose.header.frame_id = "base_link"
    reference_pose.pose.orientation.w = 1.0

    with pytest.raises(ValueError):
        create_fixed_place_pose(
            reference_pose=reference_pose,
            place_x_m=invalid_coordinate,
            place_y_m=0.35,
            place_z_m=0.55,
        )


def test_fixed_place_pose_rejects_missing_frame():
    with pytest.raises(ValueError):
        create_fixed_place_pose(
            reference_pose=PoseStamped(),
            place_x_m=0.45,
            place_y_m=0.35,
            place_z_m=0.55,
        )


@pytest.mark.parametrize(
    "grasp_succeeded, expected_action",
    [
        (True, "retreat"),
        (False, "retry"),
    ],
)
def test_grasp_result_selects_action(
        grasp_succeeded,
        expected_action,
):
    assert get_grasp_result_action(
        "grasp",
        True,
        grasp_succeeded,
    ) == expected_action


@pytest.mark.parametrize(
    "current_phase, waiting_for_grasp_result",
    [
        ("pregrasp", True),
        ("retreat", True),
        ("grasp", False),
    ],
)
def test_grasp_result_is_ignored_outside_waiting_state(
        current_phase,
        waiting_for_grasp_result,
):
    assert get_grasp_result_action(
        current_phase,
        waiting_for_grasp_result,
        True,
    ) is None


def test_grasp_result_rejects_non_boolean_value():
    with pytest.raises(TypeError):
        get_grasp_result_action(
            "grasp",
            True,
            "true",
        )
