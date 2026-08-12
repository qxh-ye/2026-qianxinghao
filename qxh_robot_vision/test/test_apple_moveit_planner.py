import pytest

from qxh_robot_vision.apple_moveit_planner import (
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
