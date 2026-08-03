import cv2
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from qxh_robot_vision.apple_detection import (
    detect_apples,
    draw_detections,
)


class AppleDetectorNode(Node):
    """
    接收 ROS2 图像， 转换为 OpenCv 图像并发布标注结果
    """

    def __init__(self):
        super().__init__("apple_detector_node")

        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.annotated_image_publisher = self.create_publisher(
            Image,
            "/apple_detector/image_annotated",
            10,
        )

        self.frame_count = 0

        self.get_logger().info(
            "Apple detector node started. "
            "Waiting for images on /camera/image_raw"
        )


    def image_callback(self, message):
        """处理一帧 ROS2 图像， 检测苹果并发布标注结果"""
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f"Failed to convert ROS2 image: {error}"
            )
            return

        self.frame_count += 1

        try:
            detections = detect_apples(image)

            annotated_image = draw_detections(image, detections)

        except (ValueError, cv2.error) as error:
            self.get_logger().error(
                f"Failed to detect apples: {error}"
            )
            return
        try:
            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding="bgr8"
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f"Failed to create ROS image: {error}"
            )
            return

        annotated_message.header = message.header

        self.annotated_image_publisher.publish(
            annotated_message
        )

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Processed {self.frame_count} images; "
                f"detected {len(detections)} apples"
            )


def main(args=None):
    rclpy.init(args=args)

    node = AppleDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()