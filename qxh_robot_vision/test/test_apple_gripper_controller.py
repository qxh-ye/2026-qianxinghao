import math

import pytest

from action_msgs.msg import GoalStatus
from qxh_robot_vision.apple_gripper_controller import (
    create_close_goal,
    grasp_result_succeeded,
)


def test_create_close_goal_sets_command_values():
    goal = create_close_goal(
        closed_position_m=0.01,
        max_effort_n=40.0,
    )

    assert goal.command.position == pytest.approx(0.01)
    assert goal.command.max_effort == pytest.approx(40.0)


@pytest.mark.parametrize(
    "closed_position_m, max_effort_n",
    [
        (-0.01, 40.0),
        (math.nan, 40.0),
        (0.0, 0.0),
        (0.0, math.inf),
    ],
)
def test_create_close_goal_rejects_invalid_parameters(
        closed_position_m,
        max_effort_n,
):
    with pytest.raises(ValueError):
        create_close_goal(
            closed_position_m,
            max_effort_n,
        )


@pytest.mark.parametrize(
    "reached_goal, stalled",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_grasp_result_accepts_reached_or_stalled(
        reached_goal,
        stalled,
):
    assert grasp_result_succeeded(
        GoalStatus.STATUS_SUCCEEDED,
        reached_goal,
        stalled,
    )


def test_grasp_result_rejects_unsuccessful_action_status():
    assert not grasp_result_succeeded(
        GoalStatus.STATUS_ABORTED,
        reached_goal=True,
        stalled=True,
    )


def test_grasp_result_rejects_missing_completion_flags():
    assert not grasp_result_succeeded(
        GoalStatus.STATUS_SUCCEEDED,
        reached_goal=False,
        stalled=False,
    )
