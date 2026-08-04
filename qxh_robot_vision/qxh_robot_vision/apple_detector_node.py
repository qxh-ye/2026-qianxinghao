import cv2
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from qxh_robot_vision.apple_detection import (
    detect_apples,
    draw_detections,
)
from qxh_robot_vision.depth_projection import (
    is_valid_depth,
)


class AppleDetectorNode(Node):
    """
    接收 ROS2 图像， 转换为 OpenCv 图像并发布标注结果
    """

    def __init__(self):
        super().__init__("apple_detector_node")

        self.bridge = CvBridge()

        # 尚未收到深度图时使用 None
        self.latest_depth_image = None
        self.latest_depth_encoding = ""
        self.latest_depth_frame_id = ""

        # 尚未收到 CameraInfo 时使用 None
        self.camera_matrix = None
        self.camera_info_width = 0
        self.camera_info_height = 0
        self.camera_frame_id = ""

        self.frame_count = 0
        self.depth_message_count = 0
        self.camera_info_message_count = 0

        # RGB 图像订阅
        self.image_subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        # 深度图订阅
        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            qos_profile_sensor_data,
        )

        # 相机内参订阅
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.annotated_image_publisher = self.create_publisher(
            Image,
            "/apple_detector/image_annotated",
            10,
        )

        self.get_logger().info(
            "Apple detector node started. "
            "Waiting for RGB, depth and CameraInfo messages."
        )


    def depth_callback(self, message):
        """接收深度图，并缓存为 Numpy 二维数组"""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f"Failed to convert depth image: {error}"
            )
            return

        if depth_image.ndim != 2:
            self.get_logger().error(
                "Depth image must be single-channel;"
                f"received shape={depth_image.shape}"
            )
            return

        self.latest_depth_image = depth_image.copy()
        self.latest_depth_encoding = message.encoding
        self.latest_depth_frame_id = message.header.frame_id

        self.depth_message_count += 1

        if(self.depth_message_count == 1 or self.depth_message_count % 30 == 0):
            self.get_logger().info(
                "Depth subscription active: "
                f"count={self.depth_message_count}, "
                f"encoding={self.latest_depth_encoding}, "
                f"shape={self.latest_depth_image.shape}, "
                f"frame_id={self.latest_depth_frame_id}"
            )

    def camera_info_callback(self, message):
        """接收 CameraInfo, 并缓存相机内参矩阵。"""
        if len(message.k) != 9:
            self.get_logger().error(
                "CameraInfo.k must contain 9 values; "
                f"received {len(message.k)}"
            )
            return

        self.camera_matrix = tuple(message.k)
        self.camera_info_width = int(message.width)
        self.camera_info_height = int(message.height)
        self.camera_frame_id = message.header.frame_id

        self.camera_info_message_count += 1

        if(
            self.camera_info_message_count == 1
            or self.camera_info_message_count % 30 == 0
        ):
            self.get_logger().info(
                "CameraInfo subscription active: "
                f"count={self.camera_info_message_count}, "
                f"size=({self.camera_info_height}, "
                f"{self.camera_info_width}), "
                f"fx={self.camera_matrix[0]:.3f}, "
                f"fy={self.camera_matrix[4]:.3f}, "
                f"cx={self.camera_matrix[2]:.3f}, "
                f"cy={self.camera_matrix[5]:.3f}, "
                f"frame_id={self.camera_frame_id}"
            )


    def read_depth_at_pixel(self, pixel_x, pixel_y):
        """读取指定像素的深度，转换为米并过滤无效值。"""
        if self.latest_depth_image is None:
            return None

        try:
            pixel_x = int(pixel_x)
            pixel_y = int(pixel_y)
        except (TypeError, ValueError):
            return None

        height, width = self.latest_depth_image.shape

        if not 0 <= pixel_x < width:
            return None

        if not 0 <= pixel_y < height:
            return None

        raw_depth = float(
            self.latest_depth_image[pixel_y, pixel_x]
        )

        encoding = self.latest_depth_encoding.upper()

        if encoding == "32FC1":
            depth_m = raw_depth
        elif encoding == "16UC1":
            depth_m = raw_depth / 1000.0
        else:
            return None

        if not is_valid_depth(
            depth_m,
            min_depth_m=0.1,
            max_depth_m=10.0,
        ):
            return None

        return depth_m


    def image_callback(self, message):
        """检测苹果，并读取每个苹果中心像素的有效深度。"""
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
        except (ValueError, cv2.error) as error:
            self.get_logger().error(
                f"Failed to detect apples: {error}"
            )
            return

        depth_summaries = []

        for detection in detections:
            center_x, center_y = detection["center"]

            depth_m = self.read_depth_at_pixel(
                center_x,
                center_y,
            )

            detection["depth_m"] = depth_m

            if depth_m is None:
                depth_summaries.append(
                    f"{detection['maturity']} "
                    f"center=({center_x}, {center_y}) "
                    "depth unavailable"
                )
            else:
                depth_summaries.append(
                    f"{detection['maturity']} "
                    f"center=({center_x}, {center_y}) "
                    f"valid depth={depth_m:.3f} m"
                )

        try:
            annotated_image = draw_detections(
                image,
                detections,
            )

            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding="bgr8",
            )
        except (ValueError, cv2.error, CvBridgeError) as error:
            self.get_logger().error(
                f"Failed to create annotated image: {error}"
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

            for summary in depth_summaries:
                self.get_logger().info(summary)


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