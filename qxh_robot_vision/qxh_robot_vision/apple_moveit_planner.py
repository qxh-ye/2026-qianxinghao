import math

import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class AppleMoveItPlanner(Node):
    """依次请求MoveIt运动到预抓取和抓取位姿。"""

    def __init__(self):
        super().__init__("apple_moveit_planner")

        self.declare_parameter(
            "planning_group",
            "ur_manipulator",
        )
        self.declare_parameter(
            "end_effector_link",
            "tool0",
        )
        self.declare_parameter(
            "expected_frame",
            "base_link",
        )
        self.declare_parameter(
            "planning_time_sec",
            5.0,
        )
        self.declare_parameter(
            "planning_attempts",
            10,
        )
        self.declare_parameter(
            "position_tolerance_m",
            0.02,
        )
        self.declare_parameter(
            "orientation_tolerance_rad",
            0.20,
        )
        self.declare_parameter(
            "velocity_scaling",
            0.10,
        )
        self.declare_parameter(
            "acceleration_scaling",
            0.10,
        )
        self.declare_parameter(
            "execute_plan",
            False,
        )
        self.declare_parameter(
            "max_retries",
            1,
        )
        self.declare_parameter(
            "retry_delay_sec",
            1.0,
        )

        self.planning_group = str(
            self.get_parameter(
                "planning_group"
            ).value
        )
        self.end_effector_link = str(
            self.get_parameter(
                "end_effector_link"
            ).value
        )
        self.expected_frame = str(
            self.get_parameter(
                "expected_frame"
            ).value
        )
        self.planning_time_sec = float(
            self.get_parameter(
                "planning_time_sec"
            ).value
        )
        self.planning_attempts = int(
            self.get_parameter(
                "planning_attempts"
            ).value
        )
        self.position_tolerance_m = float(
            self.get_parameter(
                "position_tolerance_m"
            ).value
        )
        self.orientation_tolerance_rad = float(
            self.get_parameter(
                "orientation_tolerance_rad"
            ).value
        )
        self.velocity_scaling = float(
            self.get_parameter(
                "velocity_scaling"
            ).value
        )
        self.acceleration_scaling = float(
            self.get_parameter(
                "acceleration_scaling"
            ).value
        )
        self.execute_plan = bool(
            self.get_parameter(
                "execute_plan"
            ).value
        )
        self.max_retries = int(
            self.get_parameter(
                "max_retries"
            ).value
        )
        self.retry_delay_sec = float(
            self.get_parameter(
                "retry_delay_sec"
            ).value
        )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries 不能小于 0"
            )

        if (
            not math.isfinite(self.retry_delay_sec)
            or self.retry_delay_sec <= 0.0
        ):
            raise ValueError(
                "retry_delay_sec 必须是大于 0 的有限数值"
            )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.status_publisher = self.create_publisher(
            String,
            "/apple_picker/status",
            status_qos,
        )

        self.last_status_message = ""

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self.pregrasp_subscription = (
            self.create_subscription(
                PoseStamped,
                "/apple_picker/pregrasp_pose",
                self.pregrasp_pose_callback,
                10,
            )
        )

        self.grasp_subscription = self.create_subscription(
            PoseStamped,
            "/apple_picker/grasp_pose",
            self.grasp_pose_callback,
            10,
        )

        self.goal_sent = False
        self.server_warning_logged = False

        self.latest_pregrasp_pose = None
        self.latest_grasp_pose = None
        self.latest_target_pose = None
        self.current_phase = ""
        self.retry_count = 0
        self.retry_timer = None

        self.get_logger().info(
            "Apple MoveIt planner started. "
            f"group={self.planning_group}, "
            f"end_effector={self.end_effector_link}, "
            f"execute_plan={self.execute_plan}, "
            f"plan_only={not self.execute_plan}, "
            f"max_retries={self.max_retries}, "
            f"retry_delay_sec={self.retry_delay_sec:.1f}"
        )
        self.publish_status(
            "WAITING_TARGET"
        )

    def publish_status(
            self,
            state,
            detail="",
    ):
        """发布苹果采摘流程的当前状态。"""
        state = str(state).strip()
        detail = str(detail).strip()

        if not state:
            raise ValueError(
                "state 不能为空"
            )

        if detail:
            status_text = (
                f"{state}: {detail}"
            )
        else:
            status_text = state

        if status_text == self.last_status_message:
            return

        status_message = String()
        status_message.data = status_text

        self.status_publisher.publish(
            status_message
        )

        self.last_status_message = status_text

        self.get_logger().info(
            f"Picker status: {status_text}"
        )

    def create_plan_goal(self, target_pose):
        """根据当前阶段的目标位姿创建MoveGroup规划目标。"""
        if not isinstance(target_pose, PoseStamped):
            raise TypeError(
                "target_pose 必须是PoseStamped"
            )

        if target_pose.header.frame_id != self.expected_frame:
            raise ValueError(
                "目标坐标系不正确："
                f"expected={self.expected_frame}, "
                f"received={target_pose.header.frame_id}"
            )

        quaternion = target_pose.pose.orientation

        quaternion_norm = math.sqrt(
            quaternion.x ** 2
            + quaternion.y ** 2
            + quaternion.z ** 2
            + quaternion.w ** 2
        )

        if not math.isfinite(quaternion_norm):
            raise ValueError(
                "目标姿态四元数包含无效值"
            )

        if abs(quaternion_norm - 1.0) > 0.01:
            raise ValueError(
                "目标姿态四元数必须是单位四元数"
            )

        position_region = SolidPrimitive()
        position_region.type = SolidPrimitive.SPHERE
        position_region.dimensions = [
            self.position_tolerance_m
        ]

        region_pose = Pose()
        region_pose.position.x = (
            target_pose.pose.position.x
        )
        region_pose.position.y = (
            target_pose.pose.position.y
        )
        region_pose.position.z = (
            target_pose.pose.position.z
        )
        region_pose.orientation.w = 1.0

        position_constraint = PositionConstraint()
        position_constraint.header.stamp = (
            target_pose.header.stamp
        )
        position_constraint.header.frame_id = (
            target_pose.header.frame_id
        )
        position_constraint.link_name = (
            self.end_effector_link
        )
        position_constraint.constraint_region.primitives.append(
            position_region
        )
        position_constraint.constraint_region.primitive_poses.append(
            region_pose
        )
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.stamp = (
            target_pose.header.stamp
        )
        orientation_constraint.header.frame_id = (
            target_pose.header.frame_id
        )
        orientation_constraint.link_name = (
            self.end_effector_link
        )
        orientation_constraint.orientation.x = (
            quaternion.x
        )
        orientation_constraint.orientation.y = (
            quaternion.y
        )
        orientation_constraint.orientation.z = (
            quaternion.z
        )
        orientation_constraint.orientation.w = (
            quaternion.w
        )
        orientation_constraint.absolute_x_axis_tolerance = (
            self.orientation_tolerance_rad
        )
        orientation_constraint.absolute_y_axis_tolerance = (
            self.orientation_tolerance_rad
        )
        orientation_constraint.absolute_z_axis_tolerance = (
            self.orientation_tolerance_rad
        )
        orientation_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.name = (
            f"apple_{self.current_phase}"
        )
        goal_constraints.position_constraints.append(
            position_constraint
        )
        goal_constraints.orientation_constraints.append(
            orientation_constraint
        )

        goal = MoveGroup.Goal()

        goal.request.pipeline_id = ""
        goal.request.group_name = (
            self.planning_group
        )
        goal.request.num_planning_attempts = (
            self.planning_attempts
        )
        goal.request.allowed_planning_time = (
            self.planning_time_sec
        )
        goal.request.max_velocity_scaling_factor = (
            self.velocity_scaling
        )
        goal.request.max_acceleration_scaling_factor = (
            self.acceleration_scaling
        )
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(
            goal_constraints
        )

        goal.planning_options.plan_only = (
            not self.execute_plan
        )
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = (
            True
        )
        goal.planning_options.planning_scene_diff.robot_state.is_diff = (
            True
        )

        return goal

    def pregrasp_pose_callback(self, message):
        """缓存本轮预抓取位姿，并尝试启动两阶段运动。"""
        if self.goal_sent:
            return

        self.latest_pregrasp_pose = message
        self.try_start_sequence()

    def grasp_pose_callback(self, message):
        """缓存本轮抓取位姿，并尝试启动两阶段运动。"""
        if self.goal_sent:
            return

        self.latest_grasp_pose = message
        self.try_start_sequence()

    def try_start_sequence(self):
        """两条同帧目标位姿均可用时启动预抓取阶段。"""
        if self.goal_sent:
            return

        if (
            self.latest_pregrasp_pose is None
            or self.latest_grasp_pose is None
        ):
            return

        pregrasp_stamp = self.latest_pregrasp_pose.header.stamp
        grasp_stamp = self.latest_grasp_pose.header.stamp

        if (
            pregrasp_stamp.sec != grasp_stamp.sec
            or pregrasp_stamp.nanosec != grasp_stamp.nanosec
        ):
            return

        if not self.move_group_client.server_is_ready():
            if not self.server_warning_logged:
                self.get_logger().warning(
                    "MoveIt /move_action server is not ready"
                )
                self.server_warning_logged = True
            return

        self.server_warning_logged = False
        self.current_phase = "pregrasp"
        self.latest_target_pose = self.latest_pregrasp_pose
        self.retry_count = 0
        self.goal_sent = True

        self.send_moveit_request(
            self.latest_target_pose
        )

    def send_moveit_request(self, target_pose):
        """根据目标位姿发送一次MoveIt请求。"""
        if not self.move_group_client.server_is_ready():
            self.get_logger().error(
                "MoveIt /move_action server became unavailable"
            )
            self.schedule_retry(
                "MoveIt action server is unavailable"
            )
            return

        try:
            goal = self.create_plan_goal(target_pose)
        except (TypeError, ValueError) as error:
            self.get_logger().error(
                f"Invalid {self.current_phase} pose: {error}"
            )
            self.publish_status(
                "FAILED",
                f"invalid target: {error}",
            )
            self.goal_sent = False
            return

        attempt_number = self.retry_count + 1
        total_attempts = self.max_retries + 1

        self.publish_status(
            "PLANNING",
            (
                f"phase={self.current_phase}, "
                f"attempt {attempt_number}/{total_attempts}"
            ),
        )

        if self.execute_plan:
            request_mode = "plan_and_execute"
        else:
            request_mode = "plan_only"

        self.get_logger().info(
            "Sending MoveIt request: "
            f"mode={request_mode}, "
            f"phase={self.current_phase}, "
            f"attempt={attempt_number}/{total_attempts}, "
            f"frame={target_pose.header.frame_id}, "
            f"position=("
            f"{target_pose.pose.position.x:.3f}, "
            f"{target_pose.pose.position.y:.3f}, "
            f"{target_pose.pose.position.z:.3f})"
        )

        try:
            goal_future = (
                self.move_group_client.send_goal_async(
                    goal,
                    feedback_callback=(
                        self.moveit_feedback_callback
                    ),
                )
            )
        except Exception as error:
            self.get_logger().error(
                f"Failed to start MoveIt request: {error}"
            )
            self.schedule_retry(
                f"request start error: {error}"
            )
            return

        goal_future.add_done_callback(
            self.goal_response_callback
        )

    def schedule_retry(self, failure_reason):
        """失败后安排有限次数的延迟重试。"""
        if self.retry_timer is not None:
            return

        if self.retry_count >= self.max_retries:
            self.publish_status(
                "FAILED",
                (
                    f"{failure_reason}; "
                    f"retries exhausted "
                    f"({self.retry_count}/"
                    f"{self.max_retries})"
                ),
            )
            return

        self.retry_count += 1

        self.get_logger().warning(
            f"{failure_reason}; "
            f"retry {self.retry_count}/"
            f"{self.max_retries} will start in "
            f"{self.retry_delay_sec:.1f}s"
        )

        self.publish_status(
            "RETRYING",
            (
                f"retry {self.retry_count}/"
                f"{self.max_retries} in "
                f"{self.retry_delay_sec:.1f}s"
            ),
        )

        self.retry_timer = self.create_timer(
            self.retry_delay_sec,
            self.retry_timer_callback,
        )

    def retry_timer_callback(self):
        """定时器到期后重新发送缓存的目标。"""
        timer = self.retry_timer
        self.retry_timer = None

        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

        if self.latest_target_pose is None:
            self.publish_status(
                "FAILED",
                "no cached target for retry",
            )
            return

        self.send_moveit_request(
            self.latest_target_pose
        )

    def moveit_feedback_callback(
            self,
            feedback_message,
    ):
        """根据MoveIt反馈更新规划和执行状态。"""
        moveit_state = str(
            feedback_message.feedback.state
        ).upper()

        if "PLANNING" in moveit_state:
            self.publish_status(
                "PLANNING",
                f"phase={self.current_phase}",
            )
            return

        if (
            self.execute_plan
            and (
                "MONITOR" in moveit_state
                or "EXECUT" in moveit_state
            )
        ):
            self.publish_status(
                "EXECUTING",
                f"phase={self.current_phase}",
            )

    def goal_response_callback(self, future):
        """处理MoveIt是否接受本次请求。"""
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f"Failed to send MoveIt goal: {error}"
            )
            self.schedule_retry(
                f"goal send error: {error}"
            )
            return

        if not goal_handle.accepted:
            self.get_logger().error(
                "MoveIt rejected the request"
            )
            self.schedule_retry(
                "MoveIt rejected the request"
            )
            return

        self.get_logger().info(
            "MoveIt accepted the request"
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.plan_result_callback
        )

    def plan_result_callback(self, future):
        """处理MoveIt规划或规划执行的最终结果。"""
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().error(
                f"Failed to receive MoveIt result: {error}"
            )
            self.schedule_retry(
                f"result error: {error}"
            )
            return

        result = wrapped_result.result
        error_code = result.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                "MoveIt request failed: "
                f"error_code={error_code}"
            )
            self.schedule_retry(
                f"MoveIt error code {error_code}"
            )
            return

        trajectory_points = (
            result.planned_trajectory
            .joint_trajectory
            .points
        )

        point_count = len(trajectory_points)

        if point_count > 0:
            final_duration = (
                trajectory_points[-1]
                .time_from_start
                .sec
                + trajectory_points[-1]
                .time_from_start
                .nanosec
                / 1_000_000_000.0
            )
        else:
            final_duration = 0.0

        if self.execute_plan:
            result_description = (
                "planning and execution succeeded"
            )
        else:
            result_description = (
                "planning succeeded; "
                "trajectory was NOT executed"
            )

        self.get_logger().info(
            "MoveIt request succeeded; "
            f"phase={self.current_phase}, "
            f"planned_points={point_count}, "
            f"planning_time="
            f"{result.planning_time:.3f}s, "
            f"trajectory_duration="
            f"{final_duration:.3f}s, "
            f"{result_description}"
        )

        if self.current_phase == "pregrasp":
            self.publish_status(
                "PREGRASP_SUCCEEDED",
                (
                    "execution completed"
                    if self.execute_plan
                    else "planning completed"
                ),
            )

            self.current_phase = "grasp"
            self.latest_target_pose = self.latest_grasp_pose
            self.retry_count = 0

            self.send_moveit_request(
                self.latest_target_pose
            )
            return

        self.publish_status(
            "SUCCEEDED",
            (
                "grasp approach execution completed"
                if self.execute_plan
                else "grasp approach planning completed"
            ),
        )


def main(args=None):
    rclpy.init(args=args)

    node = AppleMoveItPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
