import pytest

from qxh_robot_vision.apple_suction_controller import (
    get_suction_state_action,
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


@pytest.mark.parametrize(
    "pending_command, state, expected_action",
    [
        ("initialize_detach", "detached", "initialized"),
        ("attach", "attached", "grasp_succeeded"),
        ("detach", "detached", "release_succeeded"),
    ],
)
def test_suction_state_selects_confirmed_action(
        pending_command,
        state,
        expected_action,
):
    assert get_suction_state_action(
        pending_command,
        state,
    ) == expected_action


@pytest.mark.parametrize(
    "pending_command, state",
    [
        (None, "detached"),
        ("attach", "detached"),
        ("detach", "attached"),
    ],
)
def test_suction_state_ignores_unconfirmed_transition(
        pending_command,
        state,
):
    assert get_suction_state_action(
        pending_command,
        state,
    ) is None
