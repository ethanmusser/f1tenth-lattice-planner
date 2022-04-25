#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Paths
import pathlib
from ament_index_python.packages import get_package_share_directory

# General
import csv
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from laser_scan_helpers import get_range_at_angle

# Messages & Services
from lattice_planner_pkg.srv import BasicService
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan


class OdometryLogger(Node):
    """ 
    Logs odometry to exernal file.
    """

    def __init__(self):
        super().__init__('odometry_logger_node')

        # Class Variables
        self.odom_pose = Pose()
        self.left_distance = np.nan
        self.right_distance = np.nan

        # Parameters
        odometry_topic = self.declare_parameter('odometry_topic').value
        lidar_scan_topic = self.declare_parameter('lidar_scan_topic').value
        self.log_filepath = self.declare_parameter('log_filepath').value

        # Subscriptions & Service
        self.OdometrySubscription = self.create_subscription(
            Odometry, odometry_topic, self.odometry_callback, 1)
        self.LaserScanSubscription = self.create_subscription(
            LaserScan, lidar_scan_topic, self.laser_scan_callback, 1)
        self.LoggerService = self.create_service(
            BasicService, '/lattice_planner_pkg/waypoint_logger_service', self.logger_callback)

    def odometry_callback(self, odom_msg):
        """
        Odometry mesage callback.
        """
        self.odom_pose = odom_msg.pose.pose

    def laser_scan_callback(self, scan_msg):
        """
        Odometry mesage callback.
        """
        self.right_distance = get_range_at_angle(
            scan_msg.ranges, -0.5*np.pi, scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        self.left_distance = get_range_at_angle(
            scan_msg.ranges,  0.5*np.pi, scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)

    def logger_callback(self, request, response):
        """
        Log current odometry on `odometry_topic`.
        """
        # Make File if it Doesn't Exist
        filepath = self.log_filepath
        if not pathlib.Path(filepath).is_file():
            pathlib.Path(filepath).touch()

        # Gather Pose & Convert to RPY
        x = self.odom_pose.position.x
        y = self.odom_pose.position.y
        quat = [self.odom_pose.orientation.x, self.odom_pose.orientation.y,
                self.odom_pose.orientation.z, self.odom_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        wl = self.left_distance
        wr = self.right_distance

        # Log to File
        with open(filepath, 'a', newline='') as csvfile:
            field_names = ['x', 'y', 'w_right', 'w_left', 'yaw']
            csvwriter = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_names)
            csvwriter.writerow({'x': x, 'y': y, 'w_right': wr, 'w_left': wl})
            self.get_logger().info(
                f'Logging [x = {x}, y = {y}, w_right = {wr}, w_left = {wl}')

        # Respond
        response.exit_flag = 0
        return response


def main(args=None):
    rclpy.init(args=args)
    print("OdometryLogger Initialized")
    odometry_logger_node = OdometryLogger()
    rclpy.spin(odometry_logger_node)

    odometry_logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
