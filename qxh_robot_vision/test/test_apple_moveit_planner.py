import math

import pytest

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from qxh_robot_vision.apple_moveit_planner import (
    create_cartesian_approach_request,
    create_ground_collision_objects,
    create_fixed_place_pose,
    create_pick_platform_collision_object,
    create_sorting_platform_collision_object,
    create_unripe_apple_collision_object,
    create_wrist_joint_constraints,
    get_grasp_result_action,
    get_next_motion_phase,
    get_phase_gate_action,
    get_release_result_action,
    parameterize_cartesian_trajectory,
)


def test_wrist_joint_constraints_avoid_multi_turn_ik():
    constraints = create_wrist_joint_constraints()

    assert [
        constraint.joint_name
        for constraint in constraints
    ] == [
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    for constraint in constraints:
        assert constraint.position == 0.0
        assert constraint.tolerance_above == pytest.approx(
            math.pi
        )
        assert constraint.tolerance_below == pytest.approx(
            math.pi
        )
        assert constraint.weight == 1.0


def test_cartesian_approach_request_keeps_fixed_grasp_pose():
    target_pose = PoseStamped()
    target_pose.header.frame_id = "base_link"
    target_pose.pose.position.x = 0.507
    target_pose.pose.position.y = -0.335
    target_pose.pose.position.z = 0.476
    target_pose.pose.orientation.w = 1.0

    request = create_cartesian_approach_request(
        target_pose=target_pose,
        planning_group="ur_manipulator",
        end_effector_link="tool0",
        max_step_m=0.01,
    )

    assert request.header.frame_id == "base_link"
    assert request.start_state.is_diff is True
    assert request.group_name == "ur_manipulator"
    assert request.link_name == "tool0"
    assert len(request.waypoints) == 1
    assert request.waypoints[0] == target_pose.pose
    assert request.max_step == pytest.approx(0.01)
    assert request.avoid_collisions is True


def test_cartesian_trajectory_receives_increasing_timestamps():
    trajectory = RobotTrajectory()
    for _ in range(3):
        trajectory.joint_trajectory.points.append(
            JointTrajectoryPoint()
        )

    parameterize_cartesian_trajectory(
        trajectory,
        duration_sec=6.0,
    )

    timestamps = [
        point.time_from_start.sec
        + point.time_from_start.nanosec / 1_000_000_000.0
        for point in trajectory.joint_trajectory.points
    ]
    assert timestamps == pytest.approx([2.0, 4.0, 6.0])


def test_cartesian_trajectory_rejects_empty_path():
    with pytest.raises(ValueError):
        parameterize_cartesian_trajectory(
            RobotTrajectory(),
            duration_sec=6.0,
        )


def test_ground_collision_objects_cover_floor_around_base():
    ground_objects = create_ground_collision_objects(
        "base_link"
    )

    assert len(ground_objects) == 4
    assert {
        collision_object.id
        for collision_object in ground_objects
    } == {
        "simulation_ground_positive_x",
        "simulation_ground_negative_x",
        "simulation_ground_positive_y",
        "simulation_ground_negative_y",
    }

    for collision_object in ground_objects:
        assert collision_object.header.frame_id == "base_link"
        assert collision_object.operation == collision_object.ADD
        assert len(collision_object.primitives) == 1
        assert len(collision_object.primitive_poses) == 1
        assert collision_object.primitives[0].type == (
            collision_object.primitives[0].BOX
        )
        assert collision_object.primitives[0].dimensions[2] == 0.10
        assert (
            collision_object.primitive_poses[0].position.z
            == pytest.approx(-0.05)
        )
        assert collision_object.primitive_poses[0].orientation.w == 1.0


def test_ground_collision_objects_reject_invalid_frame():
    with pytest.raises(TypeError):
        create_ground_collision_objects(None)

    with pytest.raises(ValueError):
        create_ground_collision_objects("")


def test_pick_platform_collision_matches_gazebo_scene():
    collision_object = create_pick_platform_collision_object(
        "base_link"
    )

    assert collision_object.header.frame_id == "base_link"
    assert collision_object.id == "platform_a_pick_area"
    assert collision_object.operation == collision_object.ADD

    primitive = collision_object.primitives[0]
    pose = collision_object.primitive_poses[0]
    assert primitive.type == primitive.BOX
    assert primitive.dimensions == pytest.approx(
        [0.44, 0.82, 0.12]
    )
    assert pose.position.x == pytest.approx(0.68)
    assert pose.position.y == pytest.approx(0.0)
    assert pose.position.z == pytest.approx(0.36)
    assert pose.orientation.w == 1.0


def test_sorting_platform_collision_matches_gazebo_scene():
    collision_object = create_sorting_platform_collision_object(
        "base_link"
    )

    assert collision_object.header.frame_id == "base_link"
    assert collision_object.id == "platform_b_sorting_area"
    assert collision_object.operation == collision_object.ADD

    primitive = collision_object.primitives[0]
    pose = collision_object.primitive_poses[0]
    assert primitive.type == primitive.BOX
    assert primitive.dimensions == pytest.approx(
        [0.30, 0.70, 0.12]
    )
    assert pose.position.x == pytest.approx(0.32)
    assert pose.position.y == pytest.approx(0.30)
    assert pose.position.z == pytest.approx(0.36)
    assert pose.orientation.w == 1.0


def test_unripe_apple_collision_includes_gripper_clearance():
    collision_object = create_unripe_apple_collision_object(
        "base_link"
    )

    assert collision_object.header.frame_id == "base_link"
    assert collision_object.id == "simulation_unripe_apple"
    assert collision_object.operation == collision_object.ADD
    assert len(collision_object.primitives) == 1
    assert len(collision_object.primitive_poses) == 1

    primitive = collision_object.primitives[0]
    pose = collision_object.primitive_poses[0]
    assert primitive.type == primitive.SPHERE
    assert primitive.dimensions[0] == pytest.approx(0.20)
    assert pose.position.x == pytest.approx(0.70)
    assert pose.position.y == pytest.approx(0.18)
    assert pose.position.z == pytest.approx(0.55)
    assert pose.orientation.w == 1.0


def test_unripe_apple_collision_rejects_invalid_frame():
    with pytest.raises(TypeError):
        create_unripe_apple_collision_object(None)

    with pytest.raises(ValueError):
        create_unripe_apple_collision_object("")


@pytest.mark.parametrize(
    "current_phase, expected_next_phase",
    [
        ("pregrasp", "grasp"),
        ("grasp", "retreat"),
        ("retreat", "place"),
        ("place", "return"),
        ("return", None),
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


def test_fixed_place_pose_offsets_second_apple_along_y_axis():
    reference_pose = PoseStamped()
    reference_pose.header.frame_id = "base_link"
    reference_pose.pose.orientation.w = 1.0

    place_pose = create_fixed_place_pose(
        reference_pose=reference_pose,
        place_x_m=0.45,
        place_y_m=0.35,
        place_z_m=0.55,
        apple_id=2,
        place_spacing_y_m=0.20,
    )

    assert place_pose.pose.position.x == pytest.approx(0.45)
    assert place_pose.pose.position.y == pytest.approx(0.15)
    assert place_pose.pose.position.z == pytest.approx(0.55)


@pytest.mark.parametrize("invalid_apple_id", [0, -1])
def test_fixed_place_pose_rejects_invalid_apple_id(
        invalid_apple_id,
):
    reference_pose = PoseStamped()
    reference_pose.header.frame_id = "base_link"

    with pytest.raises(ValueError):
        create_fixed_place_pose(
            reference_pose=reference_pose,
            place_x_m=0.45,
            place_y_m=0.35,
            place_z_m=0.55,
            apple_id=invalid_apple_id,
            place_spacing_y_m=0.20,
        )


def test_fixed_place_pose_rejects_negative_spacing():
    reference_pose = PoseStamped()
    reference_pose.header.frame_id = "base_link"

    with pytest.raises(ValueError):
        create_fixed_place_pose(
            reference_pose=reference_pose,
            place_x_m=0.45,
            place_y_m=0.35,
            place_z_m=0.55,
            apple_id=2,
            place_spacing_y_m=-0.20,
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


@pytest.mark.parametrize(
    "release_succeeded, expected_action",
    [
        (True, "return"),
        (False, "failed"),
    ],
)
def test_release_result_selects_final_action(
        release_succeeded,
        expected_action,
):
    assert get_release_result_action(
        "place",
        True,
        release_succeeded,
    ) == expected_action


@pytest.mark.parametrize(
    "current_phase, waiting_for_release_result",
    [
        ("retreat", True),
        ("place", False),
    ],
)
def test_release_result_is_ignored_outside_waiting_state(
        current_phase,
        waiting_for_release_result,
):
    assert get_release_result_action(
        current_phase,
        waiting_for_release_result,
        True,
    ) is None


def test_release_result_rejects_non_boolean_value():
    with pytest.raises(TypeError):
        get_release_result_action(
            "place",
            True,
            "true",
        )


@pytest.mark.parametrize(
    "completed_phase, execute_plan, expected_action",
    [
        ("pregrasp", False, "advance"),
        ("grasp", False, "advance_to_retreat"),
        ("grasp", True, "wait_grasp_result"),
        ("retreat", False, "advance"),
        ("place", False, "advance_to_return"),
        ("place", True, "wait_release_result"),
        ("return", False, "finish_plan_only"),
        ("return", True, "finish_sequence"),
    ],
)
def test_phase_gate_separates_plan_and_execute_modes(
        completed_phase,
        execute_plan,
        expected_action,
):
    assert get_phase_gate_action(
        completed_phase,
        execute_plan,
    ) == expected_action


def test_phase_gate_rejects_non_boolean_execute_plan():
    with pytest.raises(TypeError):
        get_phase_gate_action(
            "grasp",
            "false",
        )
