import math
import time

import rclpy

from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String


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

        self.attach_publisher = self.create_publisher(
            Empty,
            "/apple_picker/suction/attach",
            10,
        )
        self.detach_publisher = self.create_publisher(
            Empty,
            "/apple_picker/suction/detach",
            10,
        )
        self.result_publisher = self.create_publisher(
            Bool,
            "/apple_picker/grasp_result",
            10,
        )

        self.close_subscription = self.create_subscription(
            Empty,
            "/apple_picker/gripper_close",
            self.close_callback,
            10,
        )
        self.state_subscription = self.create_subscription(
            String,
            "/apple_picker/suction/state",
            self.state_callback,
            10,
        )

        self.initialized = False
        self.current_state = "unknown"
        self.pending_command = "detach"
        self.attach_started_at = None
        self.last_initialization_warning_at = time.monotonic()

        self.command_timer = self.create_timer(
            self.command_period_sec,
            self.command_timer_callback,
        )

        self.get_logger().info(
            "Apple suction controller started. "
            "Initializing the simulation in detached state. "
            f"command_period_sec={self.command_period_sec:.2f}, "
            f"attach_timeout_sec={self.attach_timeout_sec:.1f}"
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

    def close_callback(self, _message):
        """收到抓取命令后开始等待Gazebo确认吸附。"""
        if not self.initialized:
            self.get_logger().error(
                "Suction is not initialized in detached state"
            )
            self.publish_grasp_result(False)
            return

        if self.pending_command == "attach":
            self.get_logger().warning(
                "Ignoring close command because an attach "
                "operation is already in progress"
            )
            return

        self.pending_command = "attach"
        self.attach_started_at = time.monotonic()
        self.attach_publisher.publish(Empty())

        self.get_logger().info(
            "Suction attach requested"
        )

    def state_callback(self, message):
        """根据Gazebo固定关节状态确认初始化或抓取结果。"""
        try:
            state = normalize_suction_state(message.data)
        except ValueError as error:
            self.get_logger().warning(str(error))
            return

        self.current_state = state

        if (
            self.pending_command == "detach"
            and state == "detached"
        ):
            self.pending_command = None
            self.initialized = True
            self.get_logger().info(
                "Suction initialized: apple is detached"
            )
            return

        if (
            self.pending_command == "attach"
            and state == "attached"
        ):
            self.pending_command = None
            self.attach_started_at = None
            self.publish_grasp_result(True)

    def command_timer_callback(self):
        """重复发送命令，避免Gazebo Transport发现阶段丢包。"""
        if self.pending_command == "detach":
            self.detach_publisher.publish(Empty())

            now = time.monotonic()
            if (
                now - self.last_initialization_warning_at
                >= self.attach_timeout_sec
            ):
                self.get_logger().warning(
                    "Still waiting for Gazebo to confirm "
                    "detached suction state"
                )
                self.last_initialization_warning_at = now
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
            self.get_logger().error(
                "Suction attach timed out after "
                f"{elapsed_sec:.1f}s"
            )
            self.publish_grasp_result(False)
            return

        self.attach_publisher.publish(Empty())


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
