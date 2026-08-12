import math

import rclpy

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


def create_close_goal(
        closed_position_m,
        max_effort_n,
):
    """根据闭合位置和最大力创建标准夹爪动作目标。"""
    closed_position_m = float(closed_position_m)
    max_effort_n = float(max_effort_n)

    if (
        not math.isfinite(closed_position_m)
        or closed_position_m < 0.0
    ):
        raise ValueError(
            "closed_position_m 必须是大于或等于 0 的有限数值"
        )

    if (
        not math.isfinite(max_effort_n)
        or max_effort_n <= 0.0
    ):
        raise ValueError(
            "max_effort_n 必须是大于 0 的有限数值"
        )

    goal = GripperCommand.Goal()
    goal.command.position = closed_position_m
    goal.command.max_effort = max_effort_n

    return goal


def grasp_result_succeeded(
        action_status,
        reached_goal,
        stalled,
):
    """判断夹爪动作是否成功到位或因夹住物体而停止。"""
    if action_status != GoalStatus.STATUS_SUCCEEDED:
        return False

    return bool(reached_goal or stalled)


class AppleGripperController(Node):
    """接收苹果夹爪命令并调用标准GripperCommand动作。"""

    def __init__(self):
        super().__init__("apple_gripper_controller")

        self.declare_parameter(
            "gripper_action_name",
            "/gripper_controller/gripper_cmd",
        )
        self.declare_parameter(
            "closed_position_m",
            0.0,
        )
        self.declare_parameter(
            "max_effort_n",
            40.0,
        )

        self.gripper_action_name = str(
            self.get_parameter(
                "gripper_action_name"
            ).value
        )
        self.closed_position_m = float(
            self.get_parameter(
                "closed_position_m"
            ).value
        )
        self.max_effort_n = float(
            self.get_parameter(
                "max_effort_n"
            ).value
        )

        self.close_goal = create_close_goal(
            self.closed_position_m,
            self.max_effort_n,
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            self.gripper_action_name,
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

        self.goal_in_progress = False

        self.get_logger().info(
            "Apple gripper controller started. "
            f"action={self.gripper_action_name}, "
            f"closed_position_m={self.closed_position_m:.3f}, "
            f"max_effort_n={self.max_effort_n:.1f}"
        )

    def publish_grasp_result(self, succeeded):
        """向抓取状态机发布夹爪动作结果。"""
        message = Bool()
        message.data = bool(succeeded)
        self.result_publisher.publish(message)

        self.get_logger().info(
            "Grasp result published: "
            f"success={message.data}"
        )

    def close_callback(self, _message):
        """收到闭合命令后向夹爪控制器发送动作目标。"""
        if self.goal_in_progress:
            self.get_logger().warning(
                "Ignoring close command because a gripper "
                "goal is already in progress"
            )
            return

        if not self.gripper_client.server_is_ready():
            self.get_logger().error(
                "Gripper action server is not ready: "
                f"{self.gripper_action_name}"
            )
            self.publish_grasp_result(False)
            return

        self.goal_in_progress = True

        self.get_logger().info(
            "Sending gripper close goal: "
            f"position={self.closed_position_m:.3f} m, "
            f"max_effort={self.max_effort_n:.1f} N"
        )

        try:
            goal_future = self.gripper_client.send_goal_async(
                self.close_goal
            )
        except Exception as error:
            self.goal_in_progress = False
            self.get_logger().error(
                f"Failed to send gripper goal: {error}"
            )
            self.publish_grasp_result(False)
            return

        goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):
        """处理夹爪控制器是否接受闭合目标。"""
        try:
            goal_handle = future.result()
        except Exception as error:
            self.goal_in_progress = False
            self.get_logger().error(
                f"Failed to receive gripper goal response: {error}"
            )
            self.publish_grasp_result(False)
            return

        if not goal_handle.accepted:
            self.goal_in_progress = False
            self.get_logger().error(
                "Gripper controller rejected the close goal"
            )
            self.publish_grasp_result(False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.result_callback
        )

    def result_callback(self, future):
        """读取夹爪最终状态并转换为抓取成功布尔值。"""
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.goal_in_progress = False
            self.get_logger().error(
                f"Failed to receive gripper result: {error}"
            )
            self.publish_grasp_result(False)
            return

        result = wrapped_result.result
        succeeded = grasp_result_succeeded(
            action_status=wrapped_result.status,
            reached_goal=result.reached_goal,
            stalled=result.stalled,
        )

        self.goal_in_progress = False

        self.get_logger().info(
            "Gripper action completed: "
            f"status={wrapped_result.status}, "
            f"position={result.position:.3f} m, "
            f"effort={result.effort:.1f} N, "
            f"reached_goal={result.reached_goal}, "
            f"stalled={result.stalled}"
        )

        self.publish_grasp_result(succeeded)


def main(args=None):
    rclpy.init(args=args)

    node = AppleGripperController()

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
