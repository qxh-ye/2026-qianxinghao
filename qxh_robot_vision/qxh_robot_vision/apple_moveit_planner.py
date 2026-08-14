import math

import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
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
from std_msgs.msg import Bool, Empty, Int32, String
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def create_ground_collision_objects(frame_id):
    """创建避开机械臂底座的四块仿真地面碰撞体。"""
    if not isinstance(frame_id, str):
        raise TypeError("frame_id 必须是 str")

    if not frame_id:
        raise ValueError("frame_id 不能为空")

    ground_size_m = 10.0
    base_clearance_m = 0.25
    ground_thickness_m = 0.10
    outer_half_m = ground_size_m / 2.0
    strip_length_m = outer_half_m - base_clearance_m
    strip_center_m = (
        outer_half_m + base_clearance_m
    ) / 2.0

    strip_specs = (
        (
            "positive_x",
            strip_center_m,
            0.0,
            strip_length_m,
            ground_size_m,
        ),
        (
            "negative_x",
            -strip_center_m,
            0.0,
            strip_length_m,
            ground_size_m,
        ),
        (
            "positive_y",
            0.0,
            strip_center_m,
            2.0 * base_clearance_m,
            strip_length_m,
        ),
        (
            "negative_y",
            0.0,
            -strip_center_m,
            2.0 * base_clearance_m,
            strip_length_m,
        ),
    )

    collision_objects = []

    for object_suffix, center_x, center_y, size_x, size_y in (
            strip_specs
    ):
        ground_box = SolidPrimitive()
        ground_box.type = SolidPrimitive.BOX
        ground_box.dimensions = [
            size_x,
            size_y,
            ground_thickness_m,
        ]

        ground_pose = Pose()
        ground_pose.position.x = center_x
        ground_pose.position.y = center_y
        ground_pose.position.z = -ground_thickness_m / 2.0
        ground_pose.orientation.w = 1.0

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = (
            f"simulation_ground_{object_suffix}"
        )
        collision_object.primitives.append(ground_box)
        collision_object.primitive_poses.append(ground_pose)
        collision_object.operation = CollisionObject.ADD

        collision_objects.append(collision_object)

    return collision_objects


def create_unripe_apple_collision_object(frame_id):
    """创建包含夹爪安全余量的未成熟苹果障碍物。"""
    if not isinstance(frame_id, str):
        raise TypeError("frame_id 必须是 str")

    if not frame_id:
        raise ValueError("frame_id 不能为空")

    apple_sphere = SolidPrimitive()
    apple_sphere.type = SolidPrimitive.SPHERE
    apple_sphere.dimensions = [0.20]

    apple_pose = Pose()
    apple_pose.position.x = 0.70
    apple_pose.position.y = 0.18
    apple_pose.position.z = 0.55
    apple_pose.orientation.w = 1.0

    collision_object = CollisionObject()
    collision_object.header.frame_id = frame_id
    collision_object.id = "simulation_unripe_apple"
    collision_object.primitives.append(apple_sphere)
    collision_object.primitive_poses.append(apple_pose)
    collision_object.operation = CollisionObject.ADD

    return collision_object


def get_next_motion_phase(current_phase):
    """返回抓取动作序列中的下一个阶段。"""
    phase_transitions = {
        "pregrasp": "grasp",
        "grasp": "retreat",
        "retreat": "place",
        "place": "return",
        "return": None,
    }

    if current_phase not in phase_transitions:
        raise ValueError(
            f"未知的运动阶段: {current_phase}"
        )

    return phase_transitions[current_phase]


def create_fixed_place_pose(
        reference_pose,
        place_x_m,
        place_y_m,
        place_z_m,
        apple_id=1,
        place_spacing_y_m=0.0,
):
    """按苹果编号生成互不重叠的固定放置目标位姿."""
    if not isinstance(reference_pose, PoseStamped):
        raise TypeError(
            "reference_pose 必须是 PoseStamped"
        )

    if not reference_pose.header.frame_id:
        raise ValueError(
            "reference_pose.header.frame_id 不能为空"
        )

    if isinstance(apple_id, bool) or not isinstance(apple_id, int):
        raise TypeError(
            "apple_id 必须是 int"
        )

    if apple_id < 1:
        raise ValueError(
            "apple_id 必须大于等于 1"
        )

    place_spacing_y_m = float(place_spacing_y_m)
    if (
        not math.isfinite(place_spacing_y_m)
        or place_spacing_y_m < 0.0
    ):
        raise ValueError(
            "place_spacing_y_m 必须是大于等于 0 的有限数值"
        )

    indexed_place_y_m = (
        float(place_y_m)
        - (apple_id - 1) * place_spacing_y_m
    )

    place_coordinates = (
        float(place_x_m),
        indexed_place_y_m,
        float(place_z_m),
    )

    if not all(
        math.isfinite(value)
        for value in place_coordinates
    ):
        raise ValueError(
            "固定放置坐标必须是有限数值"
        )

    place_pose = PoseStamped()
    place_pose.header.stamp = reference_pose.header.stamp
    place_pose.header.frame_id = (
        reference_pose.header.frame_id
    )
    place_pose.pose.position.x = place_coordinates[0]
    place_pose.pose.position.y = place_coordinates[1]
    place_pose.pose.position.z = place_coordinates[2]
    place_pose.pose.orientation = (
        reference_pose.pose.orientation
    )

    return place_pose


def get_grasp_result_action(
        current_phase,
        waiting_for_grasp_result,
        grasp_succeeded,
):
    """根据当前状态和抓取结果决定继续撤退或重试抓取。"""
    if not isinstance(grasp_succeeded, bool):
        raise TypeError(
            "grasp_succeeded 必须是 bool"
        )

    if (
        current_phase != "grasp"
        or not waiting_for_grasp_result
    ):
        return None

    if grasp_succeeded:
        return "retreat"

    return "retry"


def get_release_result_action(
        current_phase,
        waiting_for_release_result,
        release_succeeded,
):
    """根据放置阶段和解除吸附结果决定最终状态."""
    if not isinstance(release_succeeded, bool):
        raise TypeError(
            "release_succeeded 必须是 bool"
        )

    if (
        current_phase != "place"
        or not waiting_for_release_result
    ):
        return None

    if release_succeeded:
        return "return"

    return "failed"


def get_phase_gate_action(
        completed_phase,
        execute_plan,
):
    """根据运行模式决定抓取和放置阶段后的动作。"""
    if not isinstance(execute_plan, bool):
        raise TypeError(
            "execute_plan 必须是 bool"
        )

    if completed_phase == "grasp":
        if execute_plan:
            return "wait_grasp_result"
        return "advance_to_retreat"

    if completed_phase == "place":
        if execute_plan:
            return "wait_release_result"
        return "advance_to_return"

    if completed_phase == "return":
        if execute_plan:
            return "finish_sequence"
        return "finish_plan_only"

    return "advance"


class AppleMoveItPlanner(Node):
    """请求 MoveIt 完成抓取、放置和返回运动。"""

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
        self.declare_parameter(
            "grasp_result_timeout_sec",
            3.0,
        )
        self.declare_parameter(
            "release_result_timeout_sec",
            3.0,
        )
        self.declare_parameter(
            "place_x_m",
            0.45,
        )
        self.declare_parameter(
            "place_y_m",
            0.35,
        )
        self.declare_parameter(
            "place_z_m",
            0.55,
        )
        self.declare_parameter(
            "place_spacing_y_m",
            0.20,
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
        self.grasp_result_timeout_sec = float(
            self.get_parameter(
                "grasp_result_timeout_sec"
            ).value
        )
        self.release_result_timeout_sec = float(
            self.get_parameter(
                "release_result_timeout_sec"
            ).value
        )
        self.place_x_m = float(
            self.get_parameter(
                "place_x_m"
            ).value
        )
        self.place_y_m = float(
            self.get_parameter(
                "place_y_m"
            ).value
        )
        self.place_z_m = float(
            self.get_parameter(
                "place_z_m"
            ).value
        )
        self.place_spacing_y_m = float(
            self.get_parameter(
                "place_spacing_y_m"
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

        if (
            not math.isfinite(
                self.grasp_result_timeout_sec
            )
            or self.grasp_result_timeout_sec <= 0.0
        ):
            raise ValueError(
                "grasp_result_timeout_sec "
                "必须是大于 0 的有限数值"
            )

        if (
            not math.isfinite(
                self.release_result_timeout_sec
            )
            or self.release_result_timeout_sec <= 0.0
        ):
            raise ValueError(
                "release_result_timeout_sec "
                "必须是大于 0 的有限数值"
            )

        if not all(
            math.isfinite(value)
            for value in (
                self.place_x_m,
                self.place_y_m,
                self.place_z_m,
                self.place_spacing_y_m,
            )
        ):
            raise ValueError(
                "固定放置坐标必须是有限数值"
            )

        if self.place_spacing_y_m < 0.0:
            raise ValueError(
                "place_spacing_y_m 不能小于 0"
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

        self.gripper_close_publisher = self.create_publisher(
            Empty,
            "/apple_picker/gripper_close",
            10,
        )
        self.gripper_open_publisher = self.create_publisher(
            Empty,
            "/apple_picker/gripper_open",
            10,
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

        self.grasp_result_subscription = (
            self.create_subscription(
                Bool,
                "/apple_picker/grasp_result",
                self.grasp_result_callback,
                10,
            )
        )
        self.release_result_subscription = (
            self.create_subscription(
                Bool,
                "/apple_picker/release_result",
                self.release_result_callback,
                10,
            )
        )
        self.current_apple_id_subscription = (
            self.create_subscription(
                Int32,
                "/apple_picker/current_apple_id",
                self.current_apple_id_callback,
                status_qos,
            )
        )

        self.goal_sent = False
        self.server_warning_logged = False

        self.latest_pregrasp_pose = None
        self.latest_grasp_pose = None
        self.latest_target_pose = None
        self.current_phase = ""
        self.retry_count = 0
        self.retry_timer = None
        self.waiting_for_grasp_result = False
        self.grasp_result_timeout_timer = None
        self.waiting_for_release_result = False
        self.release_result_timeout_timer = None
        self.current_apple_id = 1
        self.active_apple_id = 1

        self.get_logger().info(
            "Apple MoveIt planner started. "
            f"group={self.planning_group}, "
            f"end_effector={self.end_effector_link}, "
            f"execute_plan={self.execute_plan}, "
            f"plan_only={not self.execute_plan}, "
            f"max_retries={self.max_retries}, "
            f"retry_delay_sec={self.retry_delay_sec:.1f}, "
            "grasp_result_timeout_sec="
            f"{self.grasp_result_timeout_sec:.1f}, "
            "release_result_timeout_sec="
            f"{self.release_result_timeout_sec:.1f}, "
            "place_position=("
            f"{self.place_x_m:.3f}, "
            f"{self.place_y_m:.3f}, "
            f"{self.place_z_m:.3f}), "
            "place_spacing_y_m="
            f"{self.place_spacing_y_m:.3f}"
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

    def reset_sequence_state(self):
        """清理终态数据，使节点能够接收下一颗苹果目标."""
        self.cancel_grasp_result_timeout()
        self.cancel_release_result_timeout()

        retry_timer = self.retry_timer
        self.retry_timer = None
        if retry_timer is not None:
            retry_timer.cancel()
            self.destroy_timer(retry_timer)

        self.goal_sent = False
        self.latest_pregrasp_pose = None
        self.latest_grasp_pose = None
        self.latest_target_pose = None
        self.current_phase = ""
        self.retry_count = 0
        self.waiting_for_grasp_result = False
        self.waiting_for_release_result = False

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
        planning_world = (
            goal.planning_options.planning_scene_diff.world
        )
        planning_world.collision_objects.extend(
            create_ground_collision_objects(
                self.expected_frame
            )
        )
        planning_world.collision_objects.append(
            create_unripe_apple_collision_object(
                self.expected_frame
            )
        )

        return goal

    def pregrasp_pose_callback(self, message):
        """缓存本轮预抓取位姿，并尝试启动两阶段运动。"""
        if self.goal_sent:
            return

        self.latest_pregrasp_pose = message
        self.try_start_sequence()

    def current_apple_id_callback(self, message):
        """缓存目标节点为下一轮选中的苹果编号。"""
        apple_id = int(message.data)
        if apple_id < 1:
            self.get_logger().warning(
                f"Ignoring invalid apple_id={apple_id}"
            )
            return

        self.current_apple_id = apple_id

    def grasp_pose_callback(self, message):
        """缓存本轮抓取位姿，并尝试启动两阶段运动。"""
        if self.goal_sent:
            return

        self.latest_grasp_pose = message
        self.try_start_sequence()

    def grasp_result_callback(self, message):
        """处理抓取执行器返回的成功或失败结果。"""
        action = get_grasp_result_action(
            self.current_phase,
            self.waiting_for_grasp_result,
            message.data,
        )

        if action is None:
            self.get_logger().warning(
                "Ignoring grasp result because the planner "
                "is not waiting for it"
            )
            return

        self.waiting_for_grasp_result = False
        self.cancel_grasp_result_timeout()

        if action == "retry":
            self.schedule_retry(
                "grasp result reported failure"
            )
            return

        self.publish_status(
            "GRASP_SUCCEEDED",
            "grasp result confirmed",
        )
        self.current_phase = "retreat"
        self.latest_target_pose = (
            self.latest_pregrasp_pose
        )
        self.retry_count = 0

        self.send_moveit_request(
            self.latest_target_pose
        )

    def start_grasp_result_timeout(self):
        """进入抓取结果等待状态并启动超时定时器。"""
        self.cancel_grasp_result_timeout()
        self.waiting_for_grasp_result = True

        self.publish_status(
            "WAITING_GRASP_RESULT",
            (
                "timeout="
                f"{self.grasp_result_timeout_sec:.1f}s"
            ),
        )

        self.grasp_result_timeout_timer = (
            self.create_timer(
                self.grasp_result_timeout_sec,
                self.grasp_result_timeout_callback,
            )
        )

        self.gripper_close_publisher.publish(
            Empty()
        )

    def cancel_grasp_result_timeout(self):
        """取消并销毁当前抓取结果超时定时器。"""
        timer = self.grasp_result_timeout_timer
        self.grasp_result_timeout_timer = None

        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

    def grasp_result_timeout_callback(self):
        """抓取结果超时后按现有有限重试策略重试。"""
        if not self.waiting_for_grasp_result:
            self.cancel_grasp_result_timeout()
            return

        self.waiting_for_grasp_result = False
        self.cancel_grasp_result_timeout()
        self.schedule_retry(
            "grasp result timeout"
        )

    def release_result_callback(self, message):
        """根据解除吸附结果完成或终止本轮放置流程。"""
        action = get_release_result_action(
            self.current_phase,
            self.waiting_for_release_result,
            message.data,
        )

        if action is None:
            self.get_logger().warning(
                "Ignoring release result because the planner "
                "is not waiting for it"
            )
            return

        self.waiting_for_release_result = False
        self.cancel_release_result_timeout()

        if action == "failed":
            self.publish_status(
                "FAILED",
                "apple release reported failure",
            )
            self.reset_sequence_state()
            return

        self.publish_status(
            "RELEASE_SUCCEEDED",
            "apple placed and suction detached",
        )
        self.current_phase = "return"
        self.latest_target_pose = (
            self.latest_pregrasp_pose
        )
        self.retry_count = 0

        self.send_moveit_request(
            self.latest_target_pose
        )

    def start_release_result_timeout(self):
        """发送打开命令并等待解除吸附结果。"""
        self.cancel_release_result_timeout()
        self.waiting_for_release_result = True

        self.publish_status(
            "WAITING_RELEASE_RESULT",
            (
                "timeout="
                f"{self.release_result_timeout_sec:.1f}s"
            ),
        )

        self.release_result_timeout_timer = (
            self.create_timer(
                self.release_result_timeout_sec,
                self.release_result_timeout_callback,
            )
        )

        self.gripper_open_publisher.publish(
            Empty()
        )

    def cancel_release_result_timeout(self):
        """取消并销毁当前解除吸附结果超时定时器。"""
        timer = self.release_result_timeout_timer
        self.release_result_timeout_timer = None

        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

    def release_result_timeout_callback(self):
        """解除吸附结果超时后终止本轮放置流程。"""
        if not self.waiting_for_release_result:
            self.cancel_release_result_timeout()
            return

        self.waiting_for_release_result = False
        self.cancel_release_result_timeout()
        self.publish_status(
            "FAILED",
            "apple release result timeout",
        )
        self.reset_sequence_state()

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
        self.active_apple_id = self.current_apple_id
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
            self.reset_sequence_state()
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
            self.reset_sequence_state()
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
            self.reset_sequence_state()
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

        completed_phase = self.current_phase

        phase_gate_action = get_phase_gate_action(
            completed_phase,
            self.execute_plan,
        )

        try:
            next_phase = get_next_motion_phase(
                completed_phase
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            self.publish_status(
                "FAILED",
                str(error),
            )
            self.reset_sequence_state()
            return

        if phase_gate_action == "wait_grasp_result":
            self.start_grasp_result_timeout()
            return

        if phase_gate_action == "wait_release_result":
            self.publish_status(
                "PLACE_REACHED",
                (
                    "place execution completed; "
                    "apple remains attached"
                ),
            )
            self.start_release_result_timeout()
            return

        if phase_gate_action == "finish_plan_only":
            self.publish_status(
                "PLAN_ONLY_SUCCEEDED",
                (
                    "all motion phases planned; "
                    "no motion or suction was executed"
                ),
            )
            self.reset_sequence_state()
            return

        if phase_gate_action == "finish_sequence":
            self.publish_status(
                "SUCCEEDED",
                (
                    "apple placed, suction detached, "
                    "and robot returned to safe pose"
                ),
            )
            self.reset_sequence_state()
            return

        if next_phase is not None:
            if completed_phase == "pregrasp":
                completed_status = "PREGRASP_SUCCEEDED"
                next_target_pose = self.latest_grasp_pose
            elif completed_phase == "grasp":
                completed_status = "GRASP_PLAN_SUCCEEDED"
                next_target_pose = self.latest_pregrasp_pose
            elif completed_phase == "retreat":
                completed_status = "RETREAT_SUCCEEDED"

                try:
                    next_target_pose = create_fixed_place_pose(
                        reference_pose=(
                            self.latest_pregrasp_pose
                        ),
                        place_x_m=self.place_x_m,
                        place_y_m=self.place_y_m,
                        place_z_m=self.place_z_m,
                        apple_id=self.active_apple_id,
                        place_spacing_y_m=(
                            self.place_spacing_y_m
                        ),
                    )
                except (TypeError, ValueError) as error:
                    self.get_logger().error(
                        f"Invalid place pose: {error}"
                    )
                    self.publish_status(
                        "FAILED",
                        f"invalid place pose: {error}",
                    )
                    self.reset_sequence_state()
                    return
            elif completed_phase == "place":
                completed_status = "PLACE_PLAN_SUCCEEDED"
                next_target_pose = self.latest_pregrasp_pose
            else:
                self.publish_status(
                    "FAILED",
                    (
                        "unsupported phase transition: "
                        f"{completed_phase} -> {next_phase}"
                    ),
                )
                self.reset_sequence_state()
                return

            self.publish_status(
                completed_status,
                (
                    "execution completed"
                    if self.execute_plan
                    else "planning completed"
                ),
            )

            self.current_phase = next_phase
            self.latest_target_pose = next_target_pose
            self.retry_count = 0

            self.send_moveit_request(
                self.latest_target_pose
            )
            return

        self.publish_status(
            "FAILED",
            f"unsupported terminal phase: {completed_phase}",
        )
        self.reset_sequence_state()


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
