import pytest

from qxh_robot_vision.apple_moveit_planner import (
    get_grasp_result_action,
    get_next_motion_phase,
)


@pytest.mark.parametrize(
    "current_phase, expected_next_phase",
    [
        ("pregrasp", "grasp"),
        ("grasp", "retreat"),
        ("retreat", None),
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
