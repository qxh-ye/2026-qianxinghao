import importlib.util
from pathlib import Path

from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch"
    / "apple_pick_demo.launch.py"
)


def load_launch_module():
    """从源码路径加载单苹果演示launch模块。"""
    spec = importlib.util.spec_from_file_location(
        "apple_pick_demo_launch",
        LAUNCH_FILE,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_launch_entities():
    """生成launch描述并返回顶层实体。"""
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    return list(launch_description.entities)


def test_pick_demo_launch_declares_safe_execution_default():
    entities = get_launch_entities()
    arguments = {
        entity.name: entity
        for entity in entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert "execute_plan" in arguments
    default_value = arguments[
        "execute_plan"
    ].default_value
    assert len(default_value) == 1
    assert default_value[0].text == "false"


def test_pick_demo_launch_starts_pipeline_nodes():
    entities = get_launch_entities()
    executables = {
        entity.node_executable
        for entity in entities
        if isinstance(entity, Node)
    }

    assert {
        "parameter_bridge",
        "static_transform_publisher",
        "apple_detector_node",
        "apple_point_transformer",
        "apple_target_node",
        "apple_moveit_planner",
    }.issubset(executables)


def test_pick_demo_launch_contains_required_data_connections():
    source = LAUNCH_FILE.read_text(encoding="utf-8")

    required_connections = {
        "/rgbd_camera/image",
        "/rgbd_camera/depth_image",
        "/rgbd_camera/camera_info",
        "/camera/image_raw",
        "/camera/depth/image_raw",
        "/camera/camera_info",
        "rgbd_camera/camera_link/rgbd_sensor",
        "base_link",
    }

    for connection in required_connections:
        assert connection in source


def test_pick_demo_launch_places_ripe_apple_on_platform_b():
    source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert '"place_x_m": 0.24' in source
    assert '"place_y_m": 0.35' in source
    assert '"place_z_m": 0.60' in source
    assert (
        '"queued_pregrasp_lateral_offset_m": -0.25'
        in source
    )
    assert (
        '"queued_pregrasp_height_clearance_m": 0.06'
        in source
    )
