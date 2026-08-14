import pytest

from geometry_msgs.msg import PointStamped

from qxh_robot_vision.apple_target_node import (
    AppleCandidate,
    create_grasp_pose,
    create_pregrasp_pose,
    find_matching_candidate,
    get_picker_terminal_result,
    point_distance_m,
    select_nearest_candidate,
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


def make_candidate(apple_id, x, y, z):
    point = PointStamped()
    point.header.frame_id = "base_link"
    point.point.x = x
    point.point.y = y
    point.point.z = z
    return AppleCandidate(
        apple_id=apple_id,
        point=point,
        distance_m=point_distance_m(point),
    )


def test_candidate_matching_deduplicates_nearby_measurements():
    candidate = make_candidate(1, 0.70, -0.18, 0.55)
    repeated_point = PointStamped()
    repeated_point.header.frame_id = "base_link"
    repeated_point.point.x = 0.705
    repeated_point.point.y = -0.175
    repeated_point.point.z = 0.552

    assert find_matching_candidate(
        [candidate],
        repeated_point,
        match_distance_m=0.10,
    ) is candidate


def test_candidate_matching_keeps_separated_apples():
    candidate = make_candidate(1, 0.70, -0.18, 0.55)
    other_point = PointStamped()
    other_point.header.frame_id = "base_link"
    other_point.point.x = 0.70
    other_point.point.y = 0.18
    other_point.point.z = 0.55

    assert find_matching_candidate(
        [candidate],
        other_point,
        match_distance_m=0.10,
    ) is None


def test_nearest_unprocessed_candidate_is_selected():
    candidates = [
        make_candidate(1, 0.80, 0.20, 0.55),
        make_candidate(2, 0.60, 0.10, 0.45),
        make_candidate(3, 0.70, -0.10, 0.50),
    ]

    first = select_nearest_candidate(candidates, set())
    second = select_nearest_candidate(candidates, {2})

    assert first.apple_id == 2
    assert second.apple_id == 3
    assert select_nearest_candidate(candidates, {1, 2, 3}) is None


@pytest.mark.parametrize(
    "status_text, expected_result",
    [
        ("SUCCEEDED: complete", True),
        ("PLAN_ONLY_SUCCEEDED: complete", True),
        ("FAILED: planning error", False),
        ("PLANNING: phase=pregrasp", None),
        ("WAITING_TARGET", None),
    ],
)
def test_picker_terminal_status_is_classified(
        status_text,
        expected_result,
):
    assert get_picker_terminal_result(status_text) is expected_result
