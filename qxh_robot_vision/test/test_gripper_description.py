from pathlib import Path
import subprocess
import xml.etree.ElementTree as ElementTree

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


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

    world_file = (
        PACKAGE_ROOT
        / "worlds"
        / "ur5e_apple_rgbd_world.sdf"
    )
    world = ElementTree.parse(world_file).getroot().find("world")

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
