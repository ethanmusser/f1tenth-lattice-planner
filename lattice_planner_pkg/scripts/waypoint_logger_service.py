#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import csv
import numpy as np
from ament_index_python.packages import get_package_share_directory
import pathlib
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from pure_pursuit.srv import BasicService
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker


class OdometryLogger(Node):
    """ 
    Logs odometry to exernal file.
    """

    def __init__(self):
        super().__init__('odometry_logger_node')

        # Class Variables
        self.odom_pose = Pose()

        # Parameters
        odometry_topic = self.declare_parameter('odometry_topic').value
        self.log_filename = self.declare_parameter('log_filename').value

        # Subscriptions & Service
        self.OdometrySubscription = self.create_subscription(
            Odometry, odometry_topic, self.odometry_callback, 1)
        self.LoggerService = self.create_service(
            BasicService, 'odometry_logger_srv', self.logger_callback)


    def odometry_callback(self, odom_msg):
        """
        Odometry mesage callback.
        """
        self.odom_pose = odom_msg.pose.pose

    def logger_callback(self, request, response):
        """
        Log current odometry on `odometry_topic`.
        """
        # Make File if it Doesn't Exist
        # pkg_dir = get_package_share_directory('pure_pursuit')
        # filepath = pkg_dir + '/waypoints/' + self.log_filename + '.csv'
        filepath = '/sim_ws/src/pure_pursuit' + '/waypoints/' + self.log_filename + '.csv'
        if not pathlib.Path(filepath).is_file():
            pathlib.Path(filepath).touch()

        # Gather Pose & Convert to RPY
        x = self.odom_pose.position.x
        y = self.odom_pose.position.y
        quat = [self.odom_pose.orientation.x, self.odom_pose.orientation.y,
                self.odom_pose.orientation.z, self.odom_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)

        # Log to File
        with open(filepath, 'a', newline='') as csvfile:
            field_names = ['x', 'y', 'yaw']
            csvwriter = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_names)
            csvwriter.writerow({'x': x, 'y': y, 'yaw': yaw})
            self.get_logger().info(f'Logging [x = {x}, y = {y}, yaw = {yaw}]')
        
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