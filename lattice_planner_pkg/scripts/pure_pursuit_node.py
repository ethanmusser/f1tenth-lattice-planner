#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
from scipy.interpolate import splprep, splev
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point
from scipy.interpolate import interp1d
import sys
import csv
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from ament_index_python.packages import get_package_share_directory
import pathlib

# package_path = rospack.get_path('lab6-slam-and-pure-pursuit-team_07')
# Waypoints = np.genfromtxt(package_path+'/pure-pursuit/datapoints.csv', delimiter=',')[:, :2]

class PurePursuit(Node):
    """ 
    Implements pure-pursuit planner on the car
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Declare Parameters
        self.declare_parameter('lookahead_distance')
        self.declare_parameter('steering_angle_bound')
        self.declare_parameter('desired_speed')
        self.declare_parameter('proportional_control')
        self.declare_parameter('steering_angle_factor')
        self.declare_parameter('speed_factor')
        self.declare_parameter('sparse_waypoint_filename')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('waypoint_distance')
        self.declare_parameter('min_lookahead')
        self.declare_parameter('max_lookahead')
        self.declare_parameter('min_speed')
        self.declare_parameter('max_speed')

        # Class Variables
        self.sparse_waypoint_filename = self.get_parameter('sparse_waypoint_filename').value
        odometry_topic = self.get_parameter('odometry_topic').value
        self.waypoint_distance = self.get_parameter('waypoint_distance').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_speed = self.get_parameter('max_speed').value

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        waypoint_topic = '/pure_pursuit/waypoint'
        waypointmap_topic = '/pure_pursuit/waypoint_map'

        # Subscribers & Publishers
        self.particle_filterSubscription = self.create_subscription(
            Odometry, odometry_topic, self.pose_callback, 1)
        self.AckPublisher = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.WaypointVisualizer = self.create_publisher(Marker, waypoint_topic, 1)
        self.WaypointMapvisualizer = self.create_publisher(MarkerArray, waypointmap_topic, 1)

        # Client
        # self.path = self.generate_waypoint_path(self.get_waypoint_path(), self.waypoint_distance)
        
        self.path, self.k_values, self.velocity = self.get_waypoint_path()
        self.timer = self.create_timer(1.0, self.publish_waypoint_map_msg)
    
    def find_cur_idx(self, odom_msg):
        position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        point_dist = np.linalg.norm(self.path[:, 0:2] - position, axis=1)
        cur_idx = np.argmin(point_dist)
        return cur_idx

    def compute_lookahead(self, cur_idx):
        #velocity based lookahead
        min_vel = np.min(self.velocity)
        max_vel = np.max(self.velocity)
        lookahead = np.interp(self.velocity[cur_idx],
                    np.array([min_vel, max_vel]),
                    np.array([self.min_lookahead, self.max_lookahead]))
        # #curvature based lookahead
        # min_curv = np.min(abs(self.k_values))
        # max_curv = np.max(abs(self.k_values))
        # print('min_lookahead', self.min_lookahead)
        # print('max_lookahead', self.max_lookahead)
        # lookahead = np.interp(self.k_values[cur_idx],
        #             np.array([min_curv, max_curv]),
        #             np.array([self.min_lookahead, self.max_lookahead]))
        # lookahead = self.lookahead_distance
        # print('lookahead: ', lookahead)
        return lookahead

    def get_cur_waypoint(self, cur_idx, odom_msg, lookahead):
        position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        point_dist = np.linalg.norm(self.path[:, 0:2] - position, axis=1)
        while True:
            cur_idx = (cur_idx + 1) % np.size(self.path, axis=0)
            dis_diff = point_dist[cur_idx] - lookahead
            if dis_diff < 0:
                continue
            elif dis_diff > 0:
                x_w = np.interp(lookahead,
                    np.array([point_dist[cur_idx - 1], point_dist[cur_idx]]),
                    np.array([self.path[cur_idx - 1, 0], self.path[cur_idx, 0]]))
                y_w = np.interp(lookahead,
                    np.array([point_dist[cur_idx - 1], point_dist[cur_idx]]),
                    np.array([self.path[cur_idx - 1, 1], self.path[cur_idx, 1]]))
                break
            else:
                x_w = self.path[cur_idx, 0]
                y_w = self.path[cur_idx, 1]
                break

        return x_w, y_w

    def get_waypoint_path(self):
        """
        """
        # Read Waypoint CSV
        pkg_dir = get_package_share_directory('lattice_planner_pkg')
        filepath = pkg_dir + '/inputs/traj_ltpl_cl/' + self.sparse_waypoint_filename + '.csv'
        if not pathlib.Path(filepath).is_file():
            pathlib.Path(filepath).touch()
        data = np.genfromtxt(filepath, delimiter=';', )
        # print('filepath:', filepath)
        # x = data[:,0] + data[:,4]*data[:,6]
        # y = data[:,1] + data[:,5]*data[:,6]
        # xy = np.array([x,y]).transpose()
        xy = data[:,1:3]
        curvature = data[:,4]
        velocity = data[:,5]
        # data = 15 * np.random.rand(20, 3)
        return xy, curvature, velocity
        
    def path_to_array(self, path):
        """
        """
        # Unpack Path into Array of [x, y, yaw]
        arr = []
        for ps in path.poses:
            quat = [ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w]
            _, _, yaw = euler_from_quaternion(quat)
            arr.append([ps.pose.position.x, ps.pose.position.y, yaw])
        
        return arr

    def generate_waypoint_path(self, sparse_points, waypoint_distance, skip_header=3):
        """
        Callback for path service.
        """
        # Spline Interpolate Sparse Path
        # print("sparse_points = ")
        # print(sparse_points)
        tck, u = splprep(sparse_points.transpose(), s=0, per=True)
        approx_length = np.sum(np.linalg.norm(
            np.diff(splev(np.linspace(0, 1, 100), tck), axis=0), axis=1))

        # Discretize Splined Path
        num_waypoints = int(approx_length / waypoint_distance)
        dense_points = splev(np.linspace(0, 1, num_waypoints), tck)
        dense_points = np.array([dense_points[0], dense_points[1], dense_points[2]]).transpose()

        return dense_points

    def transform_point(self, odom_msg, goalx, goaly):
        """
        World frame to vehicle frame
        """
        quaternion = [odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w]

        rot_b_m = quaternion_matrix(quaternion)[:3, :3]
        trans_m = np.array(
            [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        tform_b_m = np.zeros((4, 4))
        tform_b_m[:3, :3] = rot_b_m
        tform_b_m[:3, 3] = trans_m
        tform_b_m[-1, -1] = 1

        goal_m = np.array([[goalx], [goaly], [0], [1]])
        goal_b = (np.linalg.inv(tform_b_m).dot(goal_m)).flatten()

        return goal_b[0], goal_b[1]

    def compute_steering_angle(self, odom_msg, y_goal_b, lookahead):
        """
        Curvature calculation
        """
        y = y_goal_b 
        curvature = (2 * y) / lookahead ** 2
        desired_angle = curvature * self.get_parameter('proportional_control').value
        return desired_angle

    def publish_drive_msg(self, desired_angle, velocity):
        """
        """
        # Compute Control Input
        angle = np.clip(desired_angle,
                        -self.get_parameter('steering_angle_bound').value,
                        self.get_parameter('steering_angle_bound').value)
        # speed = np.interp(abs(angle),
        #                   np.array([0.0, self.get_parameter('steering_angle_bound').value, np.inf]),
        #                   np.array([self.get_parameter('desired_speed').value, self.get_parameter('min_speed').value, self.get_parameter('min_speed').value]))
        preset_vel = velocity
        # speed = np.interp(preset_vel,
        #             np.array([np.min(self.velocity), np.max(self.velocity)]),
        #             np.array([self.min_speed, self.max_speed]))
        speed = preset_vel
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = self.get_parameter('steering_angle_factor').value * angle
        msg.drive.speed = self.get_parameter('speed_factor').value * speed
        self.AckPublisher.publish(msg)

        return 0

    def publish_waypoint_msg(self, x, y):
        """
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = marker.SPHERE
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 255.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        self.WaypointVisualizer.publish(marker)

    def publish_waypoint_map_msg(self):
        """
        """
        marker_array = MarkerArray()
        for idx, ps in enumerate(self.path):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.id = idx
            marker.type = marker.SPHERE
            marker.pose.position.x = ps[0]
            marker.pose.position.y = ps[1]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = np.interp(self.velocity[idx], 
                    np.array([np.min(self.velocity), np.max(self.velocity)]),
                    np.array([255.0, 0.0]))
            marker.color.g = np.interp(self.velocity[idx], 
                    np.array([np.min(self.velocity), np.max(self.velocity)]),
                    np.array([0.0, 255.0]))
            marker.color.b = 0.0
            marker.color.a = 1.0
            pt = Point()
            pt.x = marker.pose.position.x
            pt.y = marker.pose.position.y
            marker.points.append(pt)
            # marker.lifetime = rclpy.duration.Duration(seconds=0.5)
            marker_array.markers.append(marker)

        self.WaypointMapvisualizer.publish(marker_array)


    def pose_callback(self, odom_msg):
        """
        """
        #identify current index position on map
        cur_idx = self.find_cur_idx(odom_msg)
        # print('cur_vel:', self.velocity[cur_idx])
        #obtain appropriate lookahead
        lookahead = self.compute_lookahead(cur_idx)
        #get waypoint to follow based on new lookahead
        x_w, y_w = self.get_cur_waypoint(cur_idx, odom_msg, lookahead)
        # TODO: transform goal point to vehicle frame of reference
        goal_x_body, goal_y_body = self.transform_point(odom_msg, x_w, y_w)
        # TODO: calculate curvature/steering angle
        desired_angle = self.compute_steering_angle(odom_msg, goal_y_body, lookahead)
        # TODO: publish drive message, don't forget to limit the steering angle.
        self.publish_waypoint_msg(x_w, y_w)
        self.publish_drive_msg(desired_angle, self.velocity[cur_idx])


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
