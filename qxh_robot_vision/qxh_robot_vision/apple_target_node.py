import math

import rclpy

from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node


def create_pregrasp_pose(
        apple_point,
        approach_distance_m,
):
    """根据苹果基座坐标生成预抓取位姿。"""
    if not isinstance(apple_point, PointStamped):
        raise TypeError(
            "apple_point 必须是 PointStamped"
        )

    if not apple_point.header.frame_id:
        raise ValueError(
            "apple_point.header.frame_id 不能为空"
        )

    try:
        approach_distance_m = float(
            approach_distance_m
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "approach_distance_m 必须是数值"
        ) from error

    if not math.isfinite(approach_distance_m):
        raise ValueError(
            "approach_distance_m 必须是有限值"
        )

    if approach_distance_m <= 0.0:
        raise ValueError(
            "approach_distance_m 必须大于0"
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

    if point_x <= approach_distance_m:
        raise ValueError(
            "苹果X坐标小于或等于预抓取距离"
        )

    pregrasp_pose = PoseStamped()

    pregrasp_pose.header.stamp = (
        apple_point.header.stamp
    )
    pregrasp_pose.header.frame_id = (
        apple_point.header.frame_id
    )

    pregrasp_pose.pose.position.x = (
        point_x - approach_distance_m
    )
    pregrasp_pose.pose.position.y = point_y
    pregrasp_pose.pose.position.z = point_z

    pregrasp_pose.pose.orientation.x = 0.0
    pregrasp_pose.pose.orientation.y = 0.0
    pregrasp_pose.pose.orientation.z = 0.0
    pregrasp_pose.pose.orientation.w = 1.0

    return pregrasp_pose


class AppleTargetNode(Node):
    """接收成熟苹果坐标并生成预抓取位姿。"""

    def __init__(self):
        super().__init__("apple_target_node")

        self.declare_parameter(
            "approach_distance_m",
            0.25,
        )

        self.declare_parameter(
            "expected_frame",
            "base_link",
        )

        self.declare_parameter(
            "publish_once",
            False,
        )

        self.approach_distance_m = float(
            self.get_parameter(
                "approach_distance_m"
            ).value
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

        self.pregrasp_publisher = self.create_publisher(
            PoseStamped,
            "/apple_picker/pregrasp_pose",
            10,
        )

        self.ripe_point_subscription = (
            self.create_subscription(
                PointStamped,
                "/apple_detector/ripe_point_base",
                self.ripe_point_callback,
                10,
            )
        )

        self.published_count = 0

        self.get_logger().info(
            "Apple target node started. "
            f"approach_distance_m="
            f"{self.approach_distance_m:.3f}, "
            f"expected_frame={self.expected_frame}, "
            f"publish_once={self.publish_once}"
        )

    def ripe_point_callback(self, message):
        """接收成熟苹果坐标并发布预抓取位姿。"""
        if self.publish_once and self.published_count > 0:
            return

        if message.header.frame_id != self.expected_frame:
            self.get_logger().warning(
                "Ripe apple frame mismatch: "
                f"expected={self.expected_frame}, "
                f"received={message.header.frame_id}"
            )
            return

        try:
            pregrasp_pose = create_pregrasp_pose(
                apple_point=message,
                approach_distance_m=(
                    self.approach_distance_m
                ),
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f"Failed to create pregrasp pose: {error}"
            )
            return

        self.pregrasp_publisher.publish(
            pregrasp_pose
        )

        self.published_count += 1

        if (
            self.published_count == 1
            or self.published_count % 30 == 0
        ):
            self.get_logger().info(
                "Pregrasp pose published: "
                f"frame={pregrasp_pose.header.frame_id}, "
                f"position=("
                f"{pregrasp_pose.pose.position.x:.3f}, "
                f"{pregrasp_pose.pose.position.y:.3f}, "
                f"{pregrasp_pose.pose.position.z:.3f}), "
                f"count={self.published_count}"
            )


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