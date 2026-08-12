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


def test_suction_joint_matches_ripe_apple_model():
    robot = generate_robot_description()
    plugin = robot.find(
        "./gazebo/plugin"
        "[@name='ignition::gazebo::systems::DetachableJoint']"
    )

    world_file = (
        PACKAGE_ROOT
        / "worlds"
        / "ur5e_apple_rgbd_world.sdf"
    )
    world = ElementTree.parse(world_file).getroot().find("world")
    ripe_apple = world.find("./model[@name='ripe_apple']")
    apple_link = ripe_apple.find("./link[@name='link']")

    assert plugin is not None
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
        "/apple_picker/suction/detach"
    )
    assert plugin.findtext("attach_topic") == (
        "/apple_picker/suction/attach"
    )
    assert plugin.findtext("output_topic") == (
        "/apple_picker/suction/state"
    )
    assert ripe_apple.findtext("static") == "false"
    assert apple_link.findtext("gravity") == "false"
    assert float(apple_link.findtext("./inertial/mass")) > 0.0
