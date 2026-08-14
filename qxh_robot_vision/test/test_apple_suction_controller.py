import pytest

from qxh_robot_vision.apple_suction_controller import (
    create_suction_topic,
    get_suction_state_action,
    normalize_suction_state,
    validate_apple_id,
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


@pytest.mark.parametrize("apple_id", [1, 2])
@pytest.mark.parametrize("endpoint", ["attach", "detach", "state"])
def test_suction_topic_addresses_selected_apple(apple_id, endpoint):
    assert create_suction_topic(
        apple_id,
        apple_count=2,
        endpoint=endpoint,
    ) == f"/apple_picker/suction/apple_{apple_id}/{endpoint}"


@pytest.mark.parametrize("invalid_apple_id", [0, 3, -1])
def test_apple_id_rejects_unconfigured_channel(invalid_apple_id):
    with pytest.raises(ValueError):
        validate_apple_id(
            invalid_apple_id,
            apple_count=2,
        )


def test_suction_topic_rejects_unknown_endpoint():
    with pytest.raises(ValueError):
        create_suction_topic(
            apple_id=1,
            apple_count=2,
            endpoint="toggle",
        )
