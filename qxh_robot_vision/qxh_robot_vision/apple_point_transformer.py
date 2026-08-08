import rclpy

from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class ApplePointTransformer(Node):
    """将苹果三维点转换到机械臂基座坐标系。"""

    def __init__(self):
        super().__init__("apple_point_transformer")

        self.declare_parameter(
            "target_frame",
            "base_link",
        )

        self.target_frame = str(
            self.get_parameter(
                "target_frame"
            ).value
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.ripe_base_publisher = self.create_publisher(
            PointStamped,
            "/apple_detector/ripe_point_base",
            10,
        )

        self.unripe_base_publisher = self.create_publisher(
            PointStamped,
            "/apple_detector/unripe_point_base",
            10,
        )

        self.ripe_subscription = self.create_subscription(
            PointStamped,
            "/apple_detector/ripe_point",
            self.ripe_point_callback,
            10,
        )

        self.unripe_subscription = self.create_subscription(
            PointStamped,
            "/apple_detector/unripe_point",
            self.unripe_point_callback,
            10,
        )

        self.transform_count = 0
        self.transform_failure_count = 0

        self.get_logger().info(
            "Apple point transformer started. "
            f"Target frame: {self.target_frame}"
        )

    def transform_and_publish(
            self,
            message,
            publisher,
            maturity,
    ):
        """转换一个苹果点，并发布转换结果。"""
        if not message.header.frame_id:
            self.get_logger().warning(
                f"{maturity} point has no frame_id; ignored"
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                message.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.2),
            )

            transformed_point = do_transform_point(
                message,
                transform,
            )

        except TransformException as error:
            self.transform_failure_count += 1

            if (
                self.transform_failure_count == 1
                or self.transform_failure_count % 30 == 0
            ):
                self.get_logger().warning(
                    f"Failed to transform {maturity} point "
                    f"from {message.header.frame_id} "
                    f"to {self.target_frame}: {error}"
                )

            return

        transformed_point.header.stamp = (
            message.header.stamp
        )
        transformed_point.header.frame_id = (
            self.target_frame
        )

        publisher.publish(
            transformed_point
        )

        self.transform_count += 1

        if (
            self.transform_count == 1
            or self.transform_count % 30 == 0
        ):
            self.get_logger().info(
                f"{maturity} point transformed: "
                f"source_frame={message.header.frame_id}, "
                f"target_frame={self.target_frame}, "
                f"point=("
                f"{transformed_point.point.x:.3f}, "
                f"{transformed_point.point.y:.3f}, "
                f"{transformed_point.point.z:.3f}) m"
            )

    def ripe_point_callback(self, message):
        """接收并转换成熟苹果坐标。"""
        self.transform_and_publish(
            message=message,
            publisher=self.ripe_base_publisher,
            maturity="ripe",
        )

    def unripe_point_callback(self, message):
        """接收并转换未成熟苹果坐标。"""
        self.transform_and_publish(
            message=message,
            publisher=self.unripe_base_publisher,
            maturity="unripe",
        )


def main(args=None):
    rclpy.init(args=args)

    node = ApplePointTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()