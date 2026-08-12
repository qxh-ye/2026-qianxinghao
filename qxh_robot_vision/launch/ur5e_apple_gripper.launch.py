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
    """启动UR5e、MoveIt和可控制的简单双指夹爪。"""
    package_share = FindPackageShare("qxh_robot_vision")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur_simulation_gz"),
                    "launch",
                    "ur_sim_control.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": "ur5e",
            "runtime_config_package": "qxh_robot_vision",
            "controllers_file": "ur5e_gripper_controllers.yaml",
            "description_package": "qxh_robot_vision",
            "description_file": "ur5e_with_gripper.urdf.xacro",
            "initial_joint_controller": "joint_trajectory_controller",
            "launch_rviz": "false",
            "gazebo_gui": gazebo_gui,
            "world_file": world_file,
        }.items(),
    )

    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur_moveit_config"),
                    "launch",
                    "ur_moveit.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": "ur5e",
            "description_package": "ur_description",
            "description_file": "ur.urdf.xacro",
            "use_sim_time": "true",
            "launch_rviz": launch_rviz,
        }.items(),
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    suction_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            (
                "/apple_picker/suction/attach"
                "@std_msgs/msg/Empty]ignition.msgs.Empty"
            ),
            (
                "/apple_picker/suction/detach"
                "@std_msgs/msg/Empty]ignition.msgs.Empty"
            ),
            (
                "/apple_picker/suction/state"
                "@std_msgs/msg/String[ignition.msgs.StringMsg"
            ),
        ],
        output="screen",
    )

    suction_adapter = Node(
        package="qxh_robot_vision",
        executable="apple_suction_controller",
        parameters=[
            {
                "use_sim_time": True,
                "command_period_sec": 0.25,
                "attach_timeout_sec": 3.0,
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
                "world_file",
                default_value=PathJoinSubstitution(
                    [
                        package_share,
                        "worlds",
                        "ur5e_apple_rgbd_world.sdf",
                    ]
                ),
                description="Gazebo world containing the apple camera scene",
            ),
            ur_control,
            ur_moveit,
            gripper_controller_spawner,
            suction_bridge,
            suction_adapter,
        ]
    )
