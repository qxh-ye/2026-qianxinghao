import pytest

from qxh_robot_vision.depth_history import (
    DepthHistory,
)


def test_history_returns_recent_nearby_depth():
    history = DepthHistory(
        max_age_frames=15,
        max_match_distance_px=20.0,
    )

    history.update(
        maturity="ripe",
        center=(100, 80),
        depth_m=2.5,
        frame_index=100,
    )

    depth_m = history.get(
        maturity="ripe",
        center=(108, 85),
        frame_index=110,
    )

    assert depth_m == pytest.approx(2.5)


def test_history_rejects_different_maturity():
    history = DepthHistory(
        max_age_frames=15,
        max_match_distance_px=20.0,
    )

    history.update(
        maturity="ripe",
        center=(100, 80),
        depth_m=2.5,
        frame_index=100,
    )

    depth_m = history.get(
        maturity="unripe",
        center=(100, 80),
        frame_index=105,
    )

    assert depth_m is None


def test_history_rejects_distant_target():
    history = DepthHistory(
        max_age_frames=15,
        max_match_distance_px=20.0,
    )

    history.update(
        maturity="ripe",
        center=(100, 80),
        depth_m=2.5,
        frame_index=100,
    )

    depth_m = history.get(
        maturity="ripe",
        center=(150, 80),
        frame_index=105,
    )

    assert depth_m is None


def test_history_expires_old_depth():
    history = DepthHistory(
        max_age_frames=15,
        max_match_distance_px=20.0,
    )

    history.update(
        maturity="ripe",
        center=(100, 80),
        depth_m=2.5,
        frame_index=100,
    )

    assert history.get(
        maturity="ripe",
        center=(100, 80),
        frame_index=115,
    ) == pytest.approx(2.5)

    assert history.get(
        maturity="ripe",
        center=(100, 80),
        frame_index=116,
    ) is None


def test_history_rejects_invalid_depth():
    history = DepthHistory()

    with pytest.raises(ValueError):
        history.update(
            maturity="ripe",
            center=(100, 80),
            depth_m=float("nan"),
            frame_index=100,
        )

def test_history_selects_nearest_same_maturity():
    history = DepthHistory(
        max_age_frames=15,
        max_match_distance_px=20.0,
    )

    history.update(
        maturity="ripe",
        center=(100, 80),
        depth_m=2.0,
        frame_index=100,
    )

    history.update(
        maturity="ripe",
        center=(130, 80),
        depth_m=3.0,
        frame_index=100,
    )

    depth_m = history.get(
        maturity="ripe",
        center=(118, 80),
        frame_index=105,
    )

    assert depth_m == pytest.approx(3.0)