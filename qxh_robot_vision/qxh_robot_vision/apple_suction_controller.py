import math
import time

import rclpy

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Empty, Int32, String


VALID_SUCTION_STATES = {
    "attached",
    "detached",
}


def normalize_suction_state(state):
    """将Gazebo吸附状态转换成统一的小写字符串。"""
    normalized_state = str(state).strip().lower()

    if normalized_state not in VALID_SUCTION_STATES:
        raise ValueError(
            f"未知的吸附状态: {state}"
        )

    return normalized_state


def get_suction_state_action(pending_command, state):
    """将当前待执行命令和Gazebo状态映射成确认动作."""
    normalized_state = normalize_suction_state(state)
    confirmed_actions = {
        ("initialize_detach", "detached"): "initialized",
        ("attach", "attached"): "grasp_succeeded",
        ("detach", "detached"): "release_succeeded",
    }

    return confirmed_actions.get(
        (pending_command, normalized_state)
    )


def validate_apple_id(apple_id, apple_count):
    """校验苹果编号是否对应一个已配置的吸附通道."""
    if isinstance(apple_id, bool) or not isinstance(apple_id, int):
        raise TypeError("apple_id 必须是 int")
    if isinstance(apple_count, bool) or not isinstance(apple_count, int):
        raise TypeError("apple_count 必须是 int")
    if apple_count <= 0:
        raise ValueError("apple_count 必须大于0")
    if apple_id < 1 or apple_id > apple_count:
        raise ValueError(
            f"apple_id={apple_id} 超出范围 1..{apple_count}"
        )
    return apple_id


def create_suction_topic(apple_id, apple_count, endpoint):
    """生成指定苹果的Gazebo吸附话题名称."""
    validate_apple_id(apple_id, apple_count)
    if endpoint not in {"attach", "detach", "state"}:
        raise ValueError(f"未知的吸附话题端点: {endpoint}")
    return f"/apple_picker/suction/apple_{apple_id}/{endpoint}"


class AppleSuctionController(Node):
    """将苹果抓取命令转换成Gazebo吸附命令。"""

    def __init__(self):
        super().__init__("apple_suction_controller")

        self.declare_parameter(
            "command_period_sec",
            0.25,
        )
        self.declare_parameter(
            "attach_timeout_sec",
            3.0,
        )
        self.declare_parameter(
            "detach_timeout_sec",
            3.0,
        )
        self.declare_parameter(
            "apple_count",
            2,
        )

        self.command_period_sec = float(
            self.get_parameter(
                "command_period_sec"
            ).value
        )
        self.attach_timeout_sec = float(
            self.get_parameter(
                "attach_timeout_sec"
            ).value
        )
        self.detach_timeout_sec = float(
            self.get_parameter(
                "detach_timeout_sec"
            ).value
        )
        self.apple_count = int(
            self.get_parameter(
                "apple_count"
            ).value
        )

        if (
            not math.isfinite(self.command_period_sec)
            or self.command_period_sec <= 0.0
        ):
            raise ValueError(
                "command_period_sec 必须是大于 0 的有限数值"
            )

        if (
            not math.isfinite(self.attach_timeout_sec)
            or self.attach_timeout_sec <= 0.0
        ):
            raise ValueError(
                "attach_timeout_sec 必须是大于 0 的有限数值"
            )

        if (
            not math.isfinite(self.detach_timeout_sec)
            or self.detach_timeout_sec <= 0.0
        ):
            raise ValueError(
                "detach_timeout_sec 必须是大于 0 的有限数值"
            )

        if self.apple_count <= 0:
            raise ValueError("apple_count 必须大于0")

        self.attach_publishers = {}
        self.detach_publishers = {}
        self.state_subscriptions = []
        self.suction_states = {}

        for apple_id in range(1, self.apple_count + 1):
            self.attach_publishers[apple_id] = self.create_publisher(
                Empty,
                create_suction_topic(
                    apple_id,
                    self.apple_count,
                    "attach",
                ),
                10,
            )
            self.detach_publishers[apple_id] = self.create_publisher(
                Empty,
                create_suction_topic(
                    apple_id,
                    self.apple_count,
                    "detach",
                ),
                10,
            )
            self.state_subscriptions.append(
                self.create_subscription(
                    String,
                    create_suction_topic(
                        apple_id,
                        self.apple_count,
                        "state",
                    ),
                    lambda message, selected_id=apple_id: (
                        self.state_callback(selected_id, message)
                    ),
                    10,
                )
            )
            self.suction_states[apple_id] = "unknown"

        self.result_publisher = self.create_publisher(
            Bool,
            "/apple_picker/grasp_result",
            10,
        )
        self.release_result_publisher = self.create_publisher(
            Bool,
            "/apple_picker/release_result",
            10,
        )

        self.close_subscription = self.create_subscription(
            Empty,
            "/apple_picker/gripper_close",
            self.close_callback,
            10,
        )
        self.open_subscription = self.create_subscription(
            Empty,
            "/apple_picker/gripper_open",
            self.open_callback,
            10,
        )

        apple_id_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.apple_id_subscription = self.create_subscription(
            Int32,
            "/apple_picker/current_apple_id",
            self.apple_id_callback,
            apple_id_qos,
        )

        self.initialized = False
        self.selected_apple_id = None
        self.command_apple_id = None
        self.attached_apple_id = None
        self.pending_command = "initialize_detach"
        self.attach_started_at = None
        self.detach_started_at = None
        self.last_initialization_warning_at = time.monotonic()

        self.command_timer = self.create_timer(
            self.command_period_sec,
            self.command_timer_callback,
        )

        self.get_logger().info(
            "Apple suction controller started. "
            "Initializing the simulation in detached state. "
            f"command_period_sec={self.command_period_sec:.2f}, "
            f"attach_timeout_sec={self.attach_timeout_sec:.1f}, "
            f"detach_timeout_sec={self.detach_timeout_sec:.1f}, "
            f"apple_count={self.apple_count}"
        )

    def apple_id_callback(self, message):
        """保存目标节点为当前抓取轮次选择的苹果编号."""
        try:
            apple_id = validate_apple_id(
                message.data,
                self.apple_count,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().error(str(error))
            return

        self.selected_apple_id = apple_id
        self.get_logger().info(
            f"Suction target selected: apple_id={apple_id}"
        )

    def publish_grasp_result(self, succeeded):
        """发布Gazebo吸附操作的抓取结果。"""
        message = Bool()
        message.data = bool(succeeded)
        self.result_publisher.publish(message)

        self.get_logger().info(
            "Suction grasp result published: "
            f"success={message.data}"
        )

    def publish_release_result(self, succeeded):
        """发布Gazebo解除吸附操作的放置结果。"""
        message = Bool()
        message.data = bool(succeeded)
        self.release_result_publisher.publish(message)

        self.get_logger().info(
            "Suction release result published: "
            f"success={message.data}"
        )

    def close_callback(self, _message):
        """收到抓取命令后开始等待Gazebo确认吸附。"""
        if not self.initialized:
            self.get_logger().error(
                "Suction is not initialized in detached state"
            )
            self.publish_grasp_result(False)
            return

        if self.pending_command is not None:
            self.get_logger().warning(
                "Ignoring close command because another "
                "suction operation is in progress"
            )
            return

        if self.selected_apple_id is None:
            self.get_logger().error(
                "Cannot attach because no apple_id was selected"
            )
            self.publish_grasp_result(False)
            return

        if self.suction_states[self.selected_apple_id] != "detached":
            self.get_logger().error(
                "Cannot attach because selected suction joint "
                "is not detached: "
                f"apple_id={self.selected_apple_id}, "
                "state="
                f"{self.suction_states[self.selected_apple_id]}"
            )
            self.publish_grasp_result(False)
            return

        self.pending_command = "attach"
        self.command_apple_id = self.selected_apple_id
        self.attach_started_at = time.monotonic()
        self.attach_publishers[
            self.command_apple_id
        ].publish(Empty())

        self.get_logger().info(
            "Suction attach requested: "
            f"apple_id={self.command_apple_id}"
        )

    def open_callback(self, _message):
        """收到放置命令后开始等待Gazebo确认解除吸附。"""
        if not self.initialized:
            self.get_logger().error(
                "Suction is not initialized in detached state"
            )
            self.publish_release_result(False)
            return

        if self.pending_command is not None:
            self.get_logger().warning(
                "Ignoring open command because another "
                "suction operation is in progress"
            )
            return

        if self.attached_apple_id is None:
            self.get_logger().error(
                "Cannot release apple because suction is not attached"
            )
            self.publish_release_result(False)
            return

        self.pending_command = "detach"
        self.command_apple_id = self.attached_apple_id
        self.detach_started_at = time.monotonic()
        self.detach_publishers[
            self.command_apple_id
        ].publish(Empty())

        self.get_logger().info(
            "Suction detach requested: "
            f"apple_id={self.command_apple_id}"
        )

    def state_callback(self, apple_id, message):
        """根据Gazebo固定关节状态确认初始化或抓取结果。"""
        try:
            state = normalize_suction_state(message.data)
        except ValueError as error:
            self.get_logger().warning(str(error))
            return

        self.suction_states[apple_id] = state

        if self.pending_command == "initialize_detach":
            if all(
                suction_state == "detached"
                for suction_state in self.suction_states.values()
            ):
                self.pending_command = None
                self.initialized = True
                self.get_logger().info(
                    "All suction channels initialized: detached"
                )
            return

        if apple_id != self.command_apple_id:
            return

        action = get_suction_state_action(
            self.pending_command,
            state,
        )

        if action == "grasp_succeeded":
            self.pending_command = None
            self.attach_started_at = None
            self.attached_apple_id = apple_id
            self.command_apple_id = None
            self.publish_grasp_result(True)
            return

        if action == "release_succeeded":
            self.pending_command = None
            self.detach_started_at = None
            self.attached_apple_id = None
            self.command_apple_id = None
            self.publish_release_result(True)

    def command_timer_callback(self):
        """重复发送命令，避免Gazebo Transport发现阶段丢包。"""
        if self.pending_command == "initialize_detach":
            for apple_id, state in self.suction_states.items():
                if state != "detached":
                    self.detach_publishers[apple_id].publish(Empty())

            now = time.monotonic()
            if (
                now - self.last_initialization_warning_at
                >= self.detach_timeout_sec
            ):
                self.get_logger().warning(
                    "Still waiting for Gazebo to confirm "
                    "all detached suction states"
                )
                self.last_initialization_warning_at = now
            return

        if self.pending_command == "detach":
            elapsed_sec = (
                time.monotonic()
                - self.detach_started_at
            )

            if elapsed_sec >= self.detach_timeout_sec:
                self.pending_command = None
                self.detach_started_at = None
                self.command_apple_id = None
                self.get_logger().error(
                    "Suction detach timed out after "
                    f"{elapsed_sec:.1f}s"
                )
                self.publish_release_result(False)
                return

            self.detach_publishers[
                self.command_apple_id
            ].publish(Empty())
            return

        if self.pending_command != "attach":
            return

        elapsed_sec = (
            time.monotonic()
            - self.attach_started_at
        )

        if elapsed_sec >= self.attach_timeout_sec:
            self.pending_command = None
            self.attach_started_at = None
            self.command_apple_id = None
            self.get_logger().error(
                "Suction attach timed out after "
                f"{elapsed_sec:.1f}s"
            )
            self.publish_grasp_result(False)
            return

        self.attach_publishers[
            self.command_apple_id
        ].publish(Empty())


def main(args=None):
    rclpy.init(args=args)

    node = AppleSuctionController()

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
