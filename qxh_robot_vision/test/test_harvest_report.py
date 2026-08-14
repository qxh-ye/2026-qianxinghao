import csv
import json

import pytest

from qxh_robot_vision.apple_target_node import HarvestRecord
from qxh_robot_vision.harvest_report import (
    build_harvest_summary,
    export_harvest_results,
    harvest_record_to_dict,
)


def make_record(
        apple_id,
        grasp_attempts,
        place_success,
        duration,
):
    return HarvestRecord(
        apple_id=apple_id,
        detection_time="2026-08-14T01:02:03+00:00",
        maturity="ripe",
        camera_3d_position=None,
        base_3d_position=(0.70, -0.18, 0.55),
        grasp_attempts=grasp_attempts,
        grasp_success=(
            True if place_success is True else None
        ),
        place_success=place_success,
        place_position=(0.45, 0.35, 0.55),
        failure_reason=(
            "" if place_success is True else "planning failed"
        ),
        duration=duration,
    )


def test_harvest_record_conversion_keeps_report_schema():
    record_data = harvest_record_to_dict(
        make_record(1, 1, True, 10.0)
    )

    assert record_data["apple_id"] == 1
    assert record_data["maturity"] == "ripe"
    assert record_data["base_3d_position"] == (
        0.70,
        -0.18,
        0.55,
    )


def test_harvest_summary_counts_success_failure_and_average():
    records = [
        make_record(1, 1, True, 10.0),
        make_record(2, 1, False, 20.0),
        make_record(3, 0, None, None),
    ]

    summary = build_harvest_summary(records)

    assert summary["total_apples"] == 3
    assert summary["ripe_apples"] == 3
    assert summary["attempted"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["success_rate_percent"] == pytest.approx(50.0)
    assert summary["average_duration_sec"] == pytest.approx(15.0)


def test_export_harvest_results_writes_json_and_csv(tmp_path):
    records = [
        make_record(2, 1, False, 20.0),
        make_record(1, 1, True, 10.0),
    ]

    json_path, csv_path, summary = export_harvest_results(
        records,
        tmp_path,
    )

    assert json_path.name == "harvest_results.json"
    assert csv_path.name == "harvest_results.csv"
    assert summary["success_rate_percent"] == pytest.approx(50.0)

    json_document = json.loads(
        json_path.read_text(encoding="utf-8")
    )
    assert json_document["summary"]["succeeded"] == 1
    assert [
        apple["apple_id"]
        for apple in json_document["apples"]
    ] == [1, 2]

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert [row["apple_id"] for row in csv_rows] == ["1", "2"]
    assert csv_rows[0]["base_3d_position"] == (
        "[0.7, -0.18, 0.55]"
    )
