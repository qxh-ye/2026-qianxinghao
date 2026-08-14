import pytest

from geometry_msgs.msg import PointStamped

from qxh_robot_vision.apple_target_node import (
    AppleCandidate,
    HarvestRecord,
    create_harvest_record,
    create_grasp_pose,
    create_pregrasp_pose,
    finish_harvest_record,
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


def test_target_poses_add_tool_clearance_above_platform(apple_point):
    pregrasp_pose = create_pregrasp_pose(
        apple_point,
        approach_distance_m=0.25,
        height_offset_m=0.06,
    )
    grasp_pose = create_grasp_pose(
        apple_point,
        grasp_offset_m=0.10,
        height_offset_m=0.06,
    )

    assert pregrasp_pose.pose.position.z == pytest.approx(0.610)
    assert grasp_pose.pose.position.z == pytest.approx(0.610)


def test_grasp_pose_matches_demo_fingertip_length(apple_point):
    grasp_pose = create_grasp_pose(
        apple_point,
        grasp_offset_m=0.18,
        height_offset_m=0.06,
    )

    assert grasp_pose.pose.position.x == pytest.approx(0.598)
    assert grasp_pose.pose.position.z == pytest.approx(0.610)


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
        detection_time="2026-08-14T01:02:03+00:00",
    )


def test_harvest_record_starts_with_detected_candidate_data():
    candidate = make_candidate(2, 0.70, -0.18, 0.55)

    record = create_harvest_record(candidate)

    assert record == HarvestRecord(
        apple_id=2,
        detection_time="2026-08-14T01:02:03+00:00",
        maturity="ripe",
        camera_3d_position=None,
        base_3d_position=(0.70, -0.18, 0.55),
    )


def test_successful_harvest_record_saves_result_and_duration():
    record = create_harvest_record(
        make_candidate(1, 0.70, -0.18, 0.55)
    )
    record.grasp_attempts = 1

    result = finish_harvest_record(
        record,
        terminal_status="SUCCEEDED: apple placed",
        duration_sec=12.5,
    )

    assert result is record
    assert record.grasp_success is True
    assert record.place_success is True
    assert record.failure_reason == ""
    assert record.duration == pytest.approx(12.5)


def test_failed_harvest_record_preserves_failure_reason():
    record = create_harvest_record(
        make_candidate(1, 0.70, -0.18, 0.55)
    )

    finish_harvest_record(
        record,
        terminal_status="FAILED: MoveIt error code -4",
        duration_sec=3.25,
    )

    assert record.grasp_success is None
    assert record.place_success is False
    assert record.failure_reason == "MoveIt error code -4"
    assert record.duration == pytest.approx(3.25)


def test_plan_only_record_does_not_claim_physical_success():
    record = create_harvest_record(
        make_candidate(1, 0.70, -0.18, 0.55)
    )

    finish_harvest_record(
        record,
        terminal_status="PLAN_ONLY_SUCCEEDED: complete",
        duration_sec=1.0,
    )

    assert record.grasp_success is None
    assert record.place_success is None
    assert record.failure_reason == ""


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
