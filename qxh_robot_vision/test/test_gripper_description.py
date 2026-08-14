from pathlib import Path
import math
import subprocess
import xml.etree.ElementTree as ElementTree

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def load_sorting_world():
    world_file = (
        PACKAGE_ROOT
        / "worlds"
        / "ur5e_apple_rgbd_world.sdf"
    )
    return ElementTree.parse(world_file).getroot().find("world")


def generate_robot_description():
    result = subprocess.run(
        [
            "xacro",
            str(
                PACKAGE_ROOT
                / "urdf"
                / "ur5e_with_gripper.urdf.xacro"
            ),
            "name:=ur",
            "ur_type:=ur5e",
            "sim_ignition:=true",
            "simulation_controllers:=controllers.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return ElementTree.fromstring(result.stdout)


def test_gripper_description_contains_controlled_joint():
    robot = generate_robot_description()

    joint = robot.find("./joint[@name='gripper_finger_joint']")
    control_joint = robot.find(
        "./ros2_control[@name='gripper_system']"
        "/joint[@name='gripper_finger_joint']"
    )

    assert joint is not None
    assert joint.attrib["type"] == "prismatic"
    assert joint.find("limit").attrib["lower"] == "0.0"
    assert joint.find("limit").attrib["upper"] == "0.18"
    assert control_joint is not None

    initial_position = control_joint.find(
        "./state_interface[@name='position']"
        "/param[@name='initial_value']"
    )
    assert initial_position.text == "0.18"


def test_gripper_controller_matches_description():
    robot = generate_robot_description()
    joint_names = {
        joint.attrib["name"]
        for joint in robot.findall("./joint")
    }

    controller_file = (
        PACKAGE_ROOT
        / "config"
        / "ur5e_gripper_controllers.yaml"
    )
    with controller_file.open(encoding="utf-8") as stream:
        controllers = yaml.safe_load(stream)

    gripper_joint = controllers[
        "gripper_controller"
    ]["ros__parameters"]["joint"]

    assert gripper_joint == "gripper_finger_joint"
    assert gripper_joint in joint_names


def test_arm_controller_tracks_wrapped_joint_errors():
    controller_file = (
        PACKAGE_ROOT
        / "config"
        / "ur5e_gripper_controllers.yaml"
    )
    with controller_file.open(encoding="utf-8") as stream:
        controllers = yaml.safe_load(stream)

    controller_parameters = controllers[
        "joint_trajectory_controller"
    ]["ros__parameters"]

    assert controller_parameters["command_interfaces"] == [
        "velocity"
    ]

    arm_joints = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    for joint_name in arm_joints:
        assert controller_parameters["constraints"][joint_name] == {
            "trajectory": -1.0,
            "goal": 0.1,
        }

    assert controller_parameters["gains"] == {
        "shoulder_pan_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
            "angle_wraparound": True,
        },
        "shoulder_lift_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
        },
        "elbow_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
        },
        "wrist_1_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
            "angle_wraparound": True,
        },
        "wrist_2_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
            "angle_wraparound": True,
        },
        "wrist_3_joint": {
            "p": 2.0,
            "i": 0.0,
            "d": 0.1,
            "ff_velocity_scale": 1.0,
            "angle_wraparound": True,
        },
    }


def test_suction_joints_match_ripe_apple_models():
    robot = generate_robot_description()
    plugins = robot.findall(
        "./gazebo/plugin"
        "[@name='ignition::gazebo::systems::DetachableJoint']"
    )

    world = load_sorting_world()

    assert len(plugins) == 2

    for apple_id, plugin in enumerate(plugins, start=1):
        ripe_apple = world.find(
            f"./model[@name='ripe_apple_{apple_id}']"
        )
        apple_link = ripe_apple.find("./link[@name='link']")
        parent_link = plugin.findtext("parent_link")
        parent_joint = robot.find(
            f"./joint/child[@link='{parent_link}']/.."
        )

        assert parent_link == "wrist_3_link"
        assert robot.find(f"./link[@name='{parent_link}']") is not None
        assert parent_joint is not None
        assert parent_joint.attrib["type"] != "fixed"
        assert plugin.findtext("child_model") == ripe_apple.attrib["name"]
        assert plugin.findtext("child_link") == apple_link.attrib["name"]
        assert plugin.findtext("detach_topic") == (
            f"/apple_picker/suction/apple_{apple_id}/detach"
        )
        assert plugin.findtext("attach_topic") == (
            f"/apple_picker/suction/apple_{apple_id}/attach"
        )
        assert plugin.findtext("output_topic") == (
            f"/apple_picker/suction/apple_{apple_id}/state"
        )
        assert ripe_apple.findtext("static") == "false"
        assert apple_link.findtext("gravity") == "false"
        assert float(apple_link.findtext("./inertial/mass")) > 0.0
        assert apple_link.find("collision") is None


def test_sorting_world_has_two_platforms_and_six_stable_apples():
    world = load_sorting_world()

    platform_a = world.find(
        "./model[@name='platform_a_pick_area']"
    )
    platform_b = world.find(
        "./model[@name='platform_b_sorting_area']"
    )

    assert platform_a is not None
    assert platform_b is not None
    assert platform_a.findtext("static") == "true"
    assert platform_b.findtext("static") == "true"

    zone_names = {
        visual.attrib["name"]
        for visual in platform_b.findall("./link/visual")
    }
    assert {
        "ripe_zone",
        "half_ripe_zone",
        "unripe_zone",
    }.issubset(zone_names)
    assert platform_b.findtext(
        "./link/visual[@name='ripe_zone']/material/diffuse"
    ) == "0.80 0.60 0.60 1"
    assert platform_b.findtext(
        "./link/visual[@name='half_ripe_zone']/material/diffuse"
    ) == "0.90 0.72 0.45 1"
    assert platform_b.findtext(
        "./link/visual[@name='unripe_zone']/material/diffuse"
    ) == "0.60 0.80 0.60 1"

    apple_names = (
        "ripe_apple_1",
        "ripe_apple_2",
        "half_ripe_apple_1",
        "half_ripe_apple_2",
        "unripe_apple_1",
        "unripe_apple_2",
    )
    apple_positions = []

    for apple_name in apple_names:
        apple = world.find(f"./model[@name='{apple_name}']")
        assert apple is not None
        assert apple.findtext("static") == "false"

        apple_link = apple.find("./link[@name='link']")
        assert apple_link.findtext("gravity") == "false"
        assert apple_link.find("collision") is None

        pose = tuple(float(value) for value in apple.findtext("pose").split())
        apple_positions.append(pose[:3])
        assert 0.46 <= pose[0] <= 0.90
        assert -0.41 <= pose[1] <= 0.41
        assert pose[2] - 0.05 >= 0.42
        assert math.hypot(pose[0] + 0.05, pose[1]) < 0.85

    for index, first_position in enumerate(apple_positions):
        for second_position in apple_positions[index + 1:]:
            center_distance = math.dist(
                first_position,
                second_position,
            )
            assert center_distance > 0.10


def test_sorting_world_uses_vmware_friendly_overview_camera():
    world = load_sorting_world()
    scene_plugin = world.find(
        "./gui/plugin[@filename='MinimalScene']"
    )

    assert scene_plugin is not None
    assert scene_plugin.findtext("engine") == "ogre"
    assert scene_plugin.findtext("camera_pose") == (
        "-1.4 -1.8 1.5 0 0.45 0.75"
    )


def test_half_ripe_apples_use_red_ratio_compatible_visuals():
    world = load_sorting_world()
    patch_radii = {
        "half_ripe_apple_1": 0.03873,
        "half_ripe_apple_2": 0.03536,
    }

    for apple_name, expected_patch_radius in patch_radii.items():
        apple = world.find(f"./model[@name='{apple_name}']")
        green_body = apple.find("./link/visual[@name='green_body']")
        red_patch = apple.find("./link/visual[@name='red_patch']")

        assert green_body is not None
        assert red_patch is not None
        assert green_body.findtext("./material/diffuse") == "0 1 0 1"
        assert red_patch.findtext("./material/diffuse") == "1 0 0 1"
        assert float(green_body.findtext("./geometry/sphere/radius")) == 0.05
        assert float(red_patch.findtext("./geometry/cylinder/radius")) == (
            expected_patch_radius
        )
        assert float(red_patch.findtext("./geometry/cylinder/length")) == 0.004
        assert red_patch.findtext("pose") == "0.052 0 0 0 1.570796 0"
