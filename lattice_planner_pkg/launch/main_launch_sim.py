import os
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
            executable="lattice_planner_node.py",
            name="lattice_planner_node",
            output="screen",
            emulate_tty=True,
            parameters=[config]
        ),
        Node(
            package="lattice_planner_pkg",
            executable="pure_pursuit_node.py",
            name="pure_pursuit_node",
            output="screen",
            emulate_tty=True,
            parameters=[config]
        ),
        Node(
            package="lattice_planner_pkg",
            executable="object_detect_node.py",
            name="object_detect_node",
            output="screen",
            emulate_tty=True,
            parameters=[config]
        )
    ])

    return ld