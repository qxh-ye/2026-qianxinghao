import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import rclpy

from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Int32, String

from qxh_robot_vision.harvest_report import export_harvest_results


@dataclass
class AppleCandidate:
    """保存一个去重后的成熟苹果候选."""

    apple_id: int
    point: PointStamped
    distance_m: float
    detection_time: str = ""


@dataclass
class HarvestRecord:
    """保存一颗苹果从检测到采摘终态的结构化结果."""

    apple_id: int
    detection_time: str
    maturity: str
    camera_3d_position: object
    base_3d_position: tuple
    grasp_attempts: int = 0
    grasp_success: object = None
    place_success: object = None
    place_position: object = None
    failure_reason: str = ""
    duration: object = None


def create_harvest_record(candidate):
    """根据稳定编号后的成熟候选创建初始记录."""
    if not isinstance(candidate, AppleCandidate):
        raise TypeError("candidate 必须是 AppleCandidate")

    if candidate.apple_id < 1:
        raise ValueError("candidate.apple_id 必须大于等于 1")

    detection_time = str(candidate.detection_time).strip()
    if not detection_time:
        raise ValueError("candidate.detection_time 不能为空")

    return HarvestRecord(
        apple_id=candidate.apple_id,
        detection_time=detection_time,
        maturity="ripe",
        camera_3d_position=None,
        base_3d_position=(
            float(candidate.point.point.x),
            float(candidate.point.point.y),
            float(candidate.point.point.z),
        ),
    )


def finish_harvest_record(
        record,
        terminal_status,
        duration_sec,
):
    """使用采摘终态更新记录并保留失败原因."""
    if not isinstance(record, HarvestRecord):
        raise TypeError("record 必须是 HarvestRecord")

    duration_sec = float(duration_sec)
    if not math.isfinite(duration_sec) or duration_sec < 0.0:
        raise ValueError("duration_sec 必须是非负有限值")

    terminal_status = str(terminal_status).strip()
    terminal_result = get_picker_terminal_result(terminal_status)
    if terminal_result is None:
        raise ValueError("terminal_status 不是采摘终态")

    record.duration = duration_sec

    if terminal_status.startswith("PLAN_ONLY_SUCCEEDED:"):
        record.grasp_success = None
        record.place_success = None
        record.failure_reason = ""
        return record

    if terminal_result:
        record.grasp_success = True
        record.place_success = True
        record.failure_reason = ""
        return record

    record.place_success = False
    record.failure_reason = terminal_status.partition(":")[2].strip()
    return record


def point_distance_m(point):
    """计算PointStamped中的点到坐标系原点的距离."""
    if not isinstance(point, PointStamped):
        raise TypeError("point 必须是 PointStamped")

    coordinates = (
        float(point.point.x),
        float(point.point.y),
        float(point.point.z),
    )
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("苹果坐标必须是有限值")

    return math.sqrt(sum(value ** 2 for value in coordinates))


def find_matching_candidate(candidates, point, match_distance_m):
    """按三维空间距离查找同一苹果的已有候选."""
    match_distance_m = float(match_distance_m)
    if (
        not math.isfinite(match_distance_m)
        or match_distance_m <= 0.0
    ):
        raise ValueError("match_distance_m 必须是大于0的有限值")

    for candidate in candidates:
        if candidate.point.header.frame_id != point.header.frame_id:
            continue

        delta_x = candidate.point.point.x - point.point.x
        delta_y = candidate.point.point.y - point.point.y
        delta_z = candidate.point.point.z - point.point.z
        distance_m = math.sqrt(
            delta_x ** 2 + delta_y ** 2 + delta_z ** 2
        )
        if distance_m <= match_distance_m:
            return candidate

    return None


def select_nearest_candidate(candidates, processed_apple_ids):
    """从未处理候选中选择距基座原点最近的苹果."""
    pending_candidates = [
        candidate
        for candidate in candidates
        if candidate.apple_id not in processed_apple_ids
    ]
    if not pending_candidates:
        return None

    return min(
        pending_candidates,
        key=lambda candidate: candidate.distance_m,
    )


def get_picker_terminal_result(status_text):
    """把采摘状态转换成成功、失败或非终态."""
    status_text = str(status_text).strip()
    if (
        status_text.startswith("SUCCEEDED:")
        or status_text.startswith("PLAN_ONLY_SUCCEEDED:")
    ):
        return True
    if status_text.startswith("FAILED:"):
        return False
    return None


def create_offset_pose(
        apple_point,
        offset_distance_m,
        distance_name,
        height_offset_m=0.0,
):
    """根据苹果基座坐标和偏移距离生成末端目标位姿。"""
    if not isinstance(apple_point, PointStamped):
        raise TypeError(
            "apple_point 必须是 PointStamped"
        )

    if not apple_point.header.frame_id:
        raise ValueError(
            "apple_point.header.frame_id 不能为空"
        )

    try:
        offset_distance_m = float(
            offset_distance_m
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{distance_name} 必须是数值"
        ) from error

    if not math.isfinite(offset_distance_m):
        raise ValueError(
            f"{distance_name} 必须是有限值"
        )

    if offset_distance_m <= 0.0:
        raise ValueError(
            f"{distance_name} 必须大于0"
        )

    height_offset_m = float(height_offset_m)
    if (
        not math.isfinite(height_offset_m)
        or height_offset_m < 0.0
    ):
        raise ValueError(
            "末端高度偏移必须是大于等于0的有限值"
        )

    point_x = float(apple_point.point.x)
    point_y = float(apple_point.point.y)
    point_z = float(apple_point.point.z)

    if not all(
        math.isfinite(value)
        for value in (point_x, point_y, point_z)
    ):
        raise ValueError(
            "苹果坐标必须是有限值"
        )

    if point_x <= offset_distance_m:
        raise ValueError(
            f"苹果X坐标小于或等于{distance_name}"
        )

    target_pose = PoseStamped()

    target_pose.header.stamp = (
        apple_point.header.stamp
    )
    target_pose.header.frame_id = (
        apple_point.header.frame_id
    )

    target_pose.pose.position.x = (
        point_x - offset_distance_m
    )
    target_pose.pose.position.y = point_y
    target_pose.pose.position.z = point_z + height_offset_m

    target_pose.pose.orientation.x = 0.0
    target_pose.pose.orientation.y = 0.0
    target_pose.pose.orientation.z = 0.0
    target_pose.pose.orientation.w = 1.0

    return target_pose


def create_pregrasp_pose(
        apple_point,
        approach_distance_m,
        height_offset_m=0.0,
        lateral_offset_m=0.0,
):
    """根据苹果基座坐标生成预抓取位姿。"""
    lateral_offset_m = float(lateral_offset_m)
    if not math.isfinite(lateral_offset_m):
        raise ValueError("预抓取侧向偏移必须是有限值")

    target_pose = create_offset_pose(
        apple_point=apple_point,
        offset_distance_m=approach_distance_m,
        distance_name="预抓取距离",
        height_offset_m=height_offset_m,
    )
    target_pose.pose.position.y += lateral_offset_m
    return target_pose


def create_grasp_pose(
        apple_point,
        grasp_offset_m,
        height_offset_m=0.0,
):
    """根据苹果基座坐标生成抓取位姿。"""
    return create_offset_pose(
        apple_point=apple_point,
        offset_distance_m=grasp_offset_m,
        distance_name="抓取偏移距离",
        height_offset_m=height_offset_m,
    )


class AppleTargetNode(Node):
    """接收成熟苹果坐标并生成预抓取和抓取位姿。"""

    def __init__(self):
        super().__init__("apple_target_node")

        self.declare_parameter(
            "approach_distance_m",
            0.25,
        )

        self.declare_parameter(
            "grasp_offset_m",
            0.10,
        )

        self.declare_parameter(
            "tool_height_offset_m",
            0.06,
        )

        self.declare_parameter(
            "queued_pregrasp_lateral_offset_m",
            0.0,
        )

        self.declare_parameter(
            "queued_pregrasp_height_clearance_m",
            0.0,
        )

        self.declare_parameter(
            "expected_frame",
            "base_link",
        )

        self.declare_parameter(
            "publish_once",
            False,
        )

        self.declare_parameter(
            "candidate_collection_sec",
            1.0,
        )

        self.declare_parameter(
            "candidate_match_distance_m",
            0.10,
        )

        self.declare_parameter(
            "harvest_report_output_dir",
            "~/.ros/qxh_robot_vision",
        )

        self.approach_distance_m = float(
            self.get_parameter(
                "approach_distance_m"
            ).value
        )

        self.grasp_offset_m = float(
            self.get_parameter(
                "grasp_offset_m"
            ).value
        )

        self.tool_height_offset_m = float(
            self.get_parameter(
                "tool_height_offset_m"
            ).value
        )

        self.queued_pregrasp_lateral_offset_m = float(
            self.get_parameter(
                "queued_pregrasp_lateral_offset_m"
            ).value
        )

        self.queued_pregrasp_height_clearance_m = float(
            self.get_parameter(
                "queued_pregrasp_height_clearance_m"
            ).value
        )

        if (
            not math.isfinite(self.tool_height_offset_m)
            or self.tool_height_offset_m < 0.0
        ):
            raise ValueError(
                "tool_height_offset_m "
                "必须是大于等于0的有限值"
            )

        if not math.isfinite(
            self.queued_pregrasp_lateral_offset_m
        ):
            raise ValueError(
                "queued_pregrasp_lateral_offset_m 必须是有限值"
            )

        if (
            not math.isfinite(
                self.queued_pregrasp_height_clearance_m
            )
            or self.queued_pregrasp_height_clearance_m < 0.0
        ):
            raise ValueError(
                "queued_pregrasp_height_clearance_m "
                "必须是大于等于0的有限值"
            )

        if self.grasp_offset_m >= self.approach_distance_m:
            raise ValueError(
                "grasp_offset_m 必须小于 approach_distance_m"
            )

        self.expected_frame = str(
            self.get_parameter(
                "expected_frame"
            ).value
        )

        self.publish_once = bool(
            self.get_parameter(
                "publish_once"
            ).value
        )

        self.candidate_collection_sec = float(
            self.get_parameter(
                "candidate_collection_sec"
            ).value
        )
        self.candidate_match_distance_m = float(
            self.get_parameter(
                "candidate_match_distance_m"
            ).value
        )
        self.harvest_report_output_dir = str(
            self.get_parameter(
                "harvest_report_output_dir"
            ).value
        ).strip()

        if (
            not math.isfinite(self.candidate_collection_sec)
            or self.candidate_collection_sec <= 0.0
        ):
            raise ValueError(
                "candidate_collection_sec 必须是大于0的有限值"
            )

        if (
            not math.isfinite(self.candidate_match_distance_m)
            or self.candidate_match_distance_m <= 0.0
        ):
            raise ValueError(
                "candidate_match_distance_m 必须是大于0的有限值"
            )

        if not self.harvest_report_output_dir:
            raise ValueError(
                "harvest_report_output_dir 不能为空"
            )

        self.pregrasp_publisher = self.create_publisher(
            PoseStamped,
            "/apple_picker/pregrasp_pose",
            10,
        )

        self.grasp_publisher = self.create_publisher(
            PoseStamped,
            "/apple_picker/grasp_pose",
            10,
        )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.current_apple_id_publisher = self.create_publisher(
            Int32,
            "/apple_picker/current_apple_id",
            status_qos,
        )
        self.candidate_status_publisher = self.create_publisher(
            String,
            "/apple_picker/candidate_status",
            status_qos,
        )

        self.ripe_point_subscription = (
            self.create_subscription(
                PointStamped,
                "/apple_detector/ripe_point_base",
                self.ripe_point_callback,
                10,
            )
        )

        self.picker_status_subscription = self.create_subscription(
            String,
            "/apple_picker/status",
            self.picker_status_callback,
            status_qos,
        )

        self.published_count = 0
        self.candidates = []
        self.processed_apple_ids = set()
        self.failed_apple_ids = set()
        self.harvest_records = {}
        self.harvest_started_at = {}
        self.active_candidate = None
        self.next_apple_id = 1
        self.collection_started_at = None
        self.collection_closed = False
        self.all_processed_reported = False
        self.dispatch_timer = self.create_timer(
            0.10,
            self.dispatch_timer_callback,
        )

        self.get_logger().info(
            "Apple target node started. "
            f"approach_distance_m="
            f"{self.approach_distance_m:.3f}, "
            f"grasp_offset_m={self.grasp_offset_m:.3f}, "
            "tool_height_offset_m="
            f"{self.tool_height_offset_m:.3f}, "
            "queued_pregrasp_lateral_offset_m="
            f"{self.queued_pregrasp_lateral_offset_m:.3f}, "
            "queued_pregrasp_height_clearance_m="
            f"{self.queued_pregrasp_height_clearance_m:.3f}, "
            f"expected_frame={self.expected_frame}, "
            f"publish_once={self.publish_once}, "
            "candidate_collection_sec="
            f"{self.candidate_collection_sec:.1f}, "
            "candidate_match_distance_m="
            f"{self.candidate_match_distance_m:.3f}, "
            "harvest_report_output_dir="
            f"{self.harvest_report_output_dir}"
        )

    def ripe_point_callback(self, message):
        """在初始采集窗口内保存并去重成熟苹果坐标."""
        if self.collection_closed:
            return

        if message.header.frame_id != self.expected_frame:
            self.get_logger().warning(
                "Ripe apple frame mismatch: "
                f"expected={self.expected_frame}, "
                f"received={message.header.frame_id}"
            )
            return

        try:
            distance_m = point_distance_m(message)
            matching_candidate = find_matching_candidate(
                self.candidates,
                message,
                self.candidate_match_distance_m,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f"Invalid ripe apple candidate: {error}"
            )
            return

        if matching_candidate is not None:
            matching_candidate.point = message
            matching_candidate.distance_m = distance_m
            return

        candidate = AppleCandidate(
            apple_id=self.next_apple_id,
            point=message,
            distance_m=distance_m,
            detection_time=datetime.now(
                timezone.utc
            ).isoformat(),
        )
        self.candidates.append(candidate)
        self.next_apple_id += 1

        if self.collection_started_at is None:
            self.collection_started_at = time.monotonic()

        self.get_logger().info(
            "Ripe apple candidate registered: "
            f"apple_id={candidate.apple_id}, "
            f"position=("
            f"{message.point.x:.3f}, "
            f"{message.point.y:.3f}, "
            f"{message.point.z:.3f}), "
            f"distance={distance_m:.3f}m"
        )

    def dispatch_timer_callback(self):
        """采集结束后按最近优先规则逐个发布目标."""
        if not self.collection_closed:
            if self.collection_started_at is None:
                return
            elapsed_sec = time.monotonic() - self.collection_started_at
            if elapsed_sec < self.candidate_collection_sec:
                return

            self.collection_closed = True
            self.candidates.sort(
                key=lambda candidate: candidate.distance_m
            )
            for apple_id, candidate in enumerate(
                self.candidates,
                start=1,
            ):
                candidate.apple_id = apple_id
                self.harvest_records[apple_id] = (
                    create_harvest_record(candidate)
                )

            self.get_logger().info(
                "Ripe apple candidate collection completed: "
                f"total={len(self.candidates)}"
            )

        if self.active_candidate is not None:
            return

        if self.publish_once and self.processed_apple_ids:
            self.publish_all_processed_status()
            return

        candidate = select_nearest_candidate(
            self.candidates,
            self.processed_apple_ids,
        )
        if candidate is None:
            self.publish_all_processed_status()
            return

        self.publish_candidate_target(candidate)

    def publish_candidate_target(self, candidate):
        """为一个候选生成位姿并交给MoveIt流程."""
        message = candidate.point
        is_queued_candidate = candidate.apple_id > 1
        pregrasp_lateral_offset_m = 0.0
        pregrasp_height_clearance_m = 0.0
        if is_queued_candidate:
            pregrasp_lateral_offset_m = (
                self.queued_pregrasp_lateral_offset_m
            )
            pregrasp_height_clearance_m = (
                self.queued_pregrasp_height_clearance_m
            )

        try:
            pregrasp_pose = create_pregrasp_pose(
                apple_point=message,
                approach_distance_m=(
                    self.approach_distance_m
                ),
                height_offset_m=(
                    self.tool_height_offset_m
                    + pregrasp_height_clearance_m
                ),
                lateral_offset_m=pregrasp_lateral_offset_m,
            )
            grasp_pose = create_grasp_pose(
                apple_point=message,
                grasp_offset_m=self.grasp_offset_m,
                height_offset_m=(
                    self.tool_height_offset_m
                ),
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                "Failed to create apple target poses: "
                f"apple_id={candidate.apple_id}, error={error}"
            )
            self.processed_apple_ids.add(candidate.apple_id)
            self.failed_apple_ids.add(candidate.apple_id)
            return

        self.active_candidate = candidate
        record = self.harvest_records[candidate.apple_id]
        record.grasp_attempts += 1
        self.harvest_started_at[candidate.apple_id] = (
            time.monotonic()
        )

        apple_id_message = Int32()
        apple_id_message.data = candidate.apple_id
        self.current_apple_id_publisher.publish(apple_id_message)

        self.pregrasp_publisher.publish(
            pregrasp_pose
        )
        self.grasp_publisher.publish(
            grasp_pose
        )

        self.published_count += 1

        candidate_status = String()
        candidate_status.data = (
            f"ACTIVE: apple_id={candidate.apple_id}, "
            f"distance={candidate.distance_m:.3f}m"
        )
        self.candidate_status_publisher.publish(candidate_status)

        self.get_logger().info(
            "Apple target poses published: "
            f"apple_id={candidate.apple_id}, "
            f"frame={pregrasp_pose.header.frame_id}, "
            f"pregrasp=("
            f"{pregrasp_pose.pose.position.x:.3f}, "
            f"{pregrasp_pose.pose.position.y:.3f}, "
            f"{pregrasp_pose.pose.position.z:.3f}), "
            f"grasp=("
            f"{grasp_pose.pose.position.x:.3f}, "
            f"{grasp_pose.pose.position.y:.3f}, "
            f"{grasp_pose.pose.position.z:.3f})"
        )

    def picker_status_callback(self, message):
        """收到本轮终态后标记苹果并允许调度下一候选."""
        terminal_result = get_picker_terminal_result(message.data)
        if terminal_result is None or self.active_candidate is None:
            return

        apple_id = self.active_candidate.apple_id
        self.processed_apple_ids.add(apple_id)
        if not terminal_result:
            self.failed_apple_ids.add(apple_id)

        started_at = self.harvest_started_at.pop(
            apple_id,
            time.monotonic(),
        )
        finish_harvest_record(
            self.harvest_records[apple_id],
            terminal_status=message.data,
            duration_sec=(
                time.monotonic() - started_at
            ),
        )

        result_name = "SUCCEEDED" if terminal_result else "FAILED"
        status_message = String()
        status_message.data = (
            f"{result_name}: apple_id={apple_id}, "
            f"processed={len(self.processed_apple_ids)}/"
            f"{len(self.candidates)}"
        )
        self.candidate_status_publisher.publish(status_message)
        self.get_logger().info(status_message.data)

        self.active_candidate = None

    def publish_all_processed_status(self):
        """所有初始成熟候选处理完毕后发布一次汇总状态."""
        if self.all_processed_reported:
            return

        total = len(self.candidates)
        failed = len(self.failed_apple_ids)
        succeeded = len(self.processed_apple_ids) - failed
        status_message = String()
        status_message.data = (
            "ALL_RIPE_APPLES_PROCESSED: "
            f"total={total}, succeeded={succeeded}, failed={failed}"
        )
        self.candidate_status_publisher.publish(status_message)
        self.get_logger().info(status_message.data)

        ordered_records = [
            self.harvest_records[apple_id]
            for apple_id in sorted(self.harvest_records)
        ]
        try:
            json_path, csv_path, summary = (
                export_harvest_results(
                    ordered_records,
                    self.harvest_report_output_dir,
                )
            )
        except (OSError, TypeError, ValueError) as error:
            self.get_logger().error(
                f"Failed to export harvest report: {error}"
            )
        else:
            self.get_logger().info(
                "Harvest summary: "
                f"total={summary['total_apples']}, "
                f"ripe={summary['ripe_apples']}, "
                f"attempted={summary['attempted']}, "
                f"succeeded={summary['succeeded']}, "
                f"failed={summary['failed']}, "
                "success_rate="
                f"{summary['success_rate_percent']:.1f}%, "
                "average_duration="
                f"{summary['average_duration_sec']:.2f}s"
            )
            self.get_logger().info(
                "Harvest reports written: "
                f"json={json_path}, csv={csv_path}"
            )

        self.all_processed_reported = True


def main(args=None):
    rclpy.init(args=args)

    node = AppleTargetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
