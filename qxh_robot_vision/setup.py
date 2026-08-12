from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'qxh_robot_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.xacro'),
        ),
        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.sdf'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qxh',
    maintainer_email='qxh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "apple_detector_node = "
            "qxh_robot_vision.apple_detector_node:main",
            "apple_point_transformer = "
            "qxh_robot_vision.apple_point_transformer:main",
            "apple_target_node = "
            "qxh_robot_vision.apple_target_node:main",
            "apple_moveit_planner = "
            "qxh_robot_vision.apple_moveit_planner:main",
            "apple_gripper_controller = "
            "qxh_robot_vision.apple_gripper_controller:main"
        ],
    },
)
