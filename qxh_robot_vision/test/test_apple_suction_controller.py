import pytest

from qxh_robot_vision.apple_suction_controller import (
    normalize_suction_state,
)


@pytest.mark.parametrize(
    "raw_state, expected_state",
    [
        ("attached", "attached"),
        ("Detached", "detached"),
        (" attached ", "attached"),
    ],
)
def test_normalize_suction_state_accepts_known_states(
        raw_state,
        expected_state,
):
    assert normalize_suction_state(raw_state) == expected_state


@pytest.mark.parametrize(
    "raw_state",
    [
        "",
        "unknown",
        "attaching",
    ],
)
def test_normalize_suction_state_rejects_unknown_states(
        raw_state,
):
    with pytest.raises(ValueError):
        normalize_suction_state(raw_state)
