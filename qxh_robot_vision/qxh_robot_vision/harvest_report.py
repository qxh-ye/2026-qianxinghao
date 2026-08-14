import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


REPORT_FIELDS = (
    "apple_id",
    "detection_time",
    "maturity",
    "camera_3d_position",
    "base_3d_position",
    "grasp_attempts",
    "grasp_success",
    "place_success",
    "place_position",
    "failure_reason",
    "duration",
)


def harvest_record_to_dict(record):
    """把采摘记录转换成可由JSON和CSV处理的字典."""
    if not is_dataclass(record):
        raise TypeError("record 必须是 dataclass 实例")

    record_data = asdict(record)
    missing_fields = [
        field_name
        for field_name in REPORT_FIELDS
        if field_name not in record_data
    ]
    if missing_fields:
        raise ValueError(
            "采摘记录缺少字段: "
            + ", ".join(missing_fields)
        )

    return {
        field_name: record_data[field_name]
        for field_name in REPORT_FIELDS
    }


def build_harvest_summary(records):
    """计算汇报所需的采摘数量、成功率和平均耗时."""
    record_data = [
        harvest_record_to_dict(record)
        for record in records
    ]

    total_apples = len(record_data)
    ripe_apples = sum(
        data["maturity"] == "ripe"
        for data in record_data
    )
    attempted = sum(
        data["grasp_attempts"] > 0
        for data in record_data
    )
    succeeded = sum(
        data["place_success"] is True
        for data in record_data
    )
    failed = sum(
        data["place_success"] is False
        for data in record_data
    )
    durations = [
        float(data["duration"])
        for data in record_data
        if data["duration"] is not None
        and math.isfinite(float(data["duration"]))
    ]

    success_rate = (
        succeeded / attempted * 100.0
        if attempted > 0
        else 0.0
    )
    average_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    return {
        "total_apples": total_apples,
        "ripe_apples": ripe_apples,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate_percent": success_rate,
        "average_duration_sec": average_duration,
    }


def export_harvest_results(records, output_dir):
    """把采摘记录和汇总写入固定名称的JSON及CSV文件."""
    output_path = Path(output_dir).expanduser()
    if not str(output_path).strip():
        raise ValueError("output_dir 不能为空")

    output_path.mkdir(parents=True, exist_ok=True)

    ordered_records = sorted(
        records,
        key=lambda record: record.apple_id,
    )
    record_data = [
        harvest_record_to_dict(record)
        for record in ordered_records
    ]
    summary = build_harvest_summary(ordered_records)

    json_path = output_path / "harvest_results.json"
    csv_path = output_path / "harvest_results.csv"

    json_document = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": summary,
        "apples": record_data,
    }
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            json_document,
            json_file,
            ensure_ascii=False,
            indent=2,
        )
        json_file.write("\n")

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=REPORT_FIELDS,
        )
        writer.writeheader()
        for data in record_data:
            csv_row = dict(data)
            for position_field in (
                "camera_3d_position",
                "base_3d_position",
                "place_position",
            ):
                position = csv_row[position_field]
                csv_row[position_field] = (
                    json.dumps(position)
                    if position is not None
                    else ""
                )
            writer.writerow(csv_row)

    return json_path, csv_path, summary
