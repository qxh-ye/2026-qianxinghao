from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """启动单苹果视觉定位、运动规划和吸附放置流程。"""
    package_share = FindPackageShare("qxh_robot_vision")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    execute_plan = LaunchConfiguration("execute_plan")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    package_share,
                    "launch",
                    "ur5e_apple_gripper.launch.py",
                ]
            )
        ),
        launch_arguments={
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
        }.items(),
    )

    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="apple_camera_bridge",
        arguments=[
            (
                "/rgbd_camera/image"
                "@sensor_msgs/msg/Image[gz.msgs.Image"
            ),
            (
                "/rgbd_camera/depth_image"
                "@sensor_msgs/msg/Image[gz.msgs.Image"
            ),
            (
                "/rgbd_camera/camera_info"
                "@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
            ),
        ],
        output="screen",
    )

    camera_transform = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="apple_camera_static_transform",
        arguments=[
            "--x", "1.50",
            "--y", "0.0",
            "--z", "0.55",
            "--qx", "-0.5",
            "--qy", "-0.5",
            "--qz", "0.5",
            "--qw", "0.5",
            "--frame-id", "world",
            "--child-frame-id",
            "rgbd_camera/camera_link/rgbd_sensor",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    apple_detector = Node(
        package="qxh_robot_vision",
        executable="apple_detector_node",
        parameters=[{"use_sim_time": True}],
        remappings=[
            ("/camera/image_raw", "/rgbd_camera/image"),
            (
                "/camera/depth/image_raw",
                "/rgbd_camera/depth_image",
            ),
            (
                "/camera/camera_info",
                "/rgbd_camera/camera_info",
            ),
        ],
        output="screen",
    )

    point_transformer = Node(
        package="qxh_robot_vision",
        executable="apple_point_transformer",
        parameters=[
            {
                "use_sim_time": True,
                "target_frame": "base_link",
            }
        ],
        output="screen",
    )

    target_generator = Node(
        package="qxh_robot_vision",
        executable="apple_target_node",
        parameters=[
            {
                "use_sim_time": True,
                "approach_distance_m": 0.25,
                "grasp_offset_m": 0.10,
                "expected_frame": "base_link",
                "publish_once": False,
            }
        ],
        output="screen",
    )

    moveit_planner = Node(
        package="qxh_robot_vision",
        executable="apple_moveit_planner",
        parameters=[
            {
                "use_sim_time": True,
                "execute_plan": execute_plan,
                "planning_group": "ur_manipulator",
                "end_effector_link": "tool0",
                "expected_frame": "base_link",
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Launch RViz with MoveIt",
            ),
            DeclareLaunchArgument(
                "gazebo_gui",
                default_value="true",
                description="Launch Gazebo GUI",
            ),
            DeclareLaunchArgument(
                "execute_plan",
                default_value="false",
                description=(
                    "Execute planned robot motions when true; "
                    "plan only when false"
                ),
            ),
            simulation,
            camera_bridge,
            camera_transform,
            apple_detector,
            point_transformer,
            target_generator,
            moveit_planner,
        ]
    )
