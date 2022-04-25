import os
from math import radians
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('lattice_planner_pkg'),
        'config',
        'params_sim.yaml'
        )

    ld = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(get_package_share_directory(
                'f1tenth_gym_ros'), 'launch'), '/gym_bridge_launch.py'])
        ),
        Node(
            package="lattice_planner_pkg",
            executable="waypoint_logger_service.py",
            name="waypoint_logger_service",
            output="screen",
            emulate_tty=True,
            parameters=[config]
        )
    ])
    
    return ld