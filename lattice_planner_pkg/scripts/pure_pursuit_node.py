#!/usr/bin/env python3
from visualization_helpers import *
import rclpy
from rclpy.node import Node

import numpy as np
from scipy.interpolate import splprep, splev
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from ament_index_python.packages import get_package_share_directory
import pathlib


class PurePursuit(Node):
    """ 
    Implements pure-pursuit planner on the car
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Declare Parameters
        self.declare_parameter('sparse_waypoint_filename')
        self.declare_parameter('drive_topic')
        self.declare_parameter('trajectory_topic')
        self.declare_parameter('global_trajectory_topic')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('waypoint_vis_topic')
        self.declare_parameter('lookahead_method')
        self.declare_parameter('use_global_traj')
        self.declare_parameter('steering_angle_bound')
        self.declare_parameter('desired_speed')
        self.declare_parameter('min_speed')
        self.declare_parameter('proportional_control')
        self.declare_parameter('steering_angle_factor')
        self.declare_parameter('speed_factor')
        self.declare_parameter('waypoint_distance')
        self.declare_parameter('min_lookahead')
        self.declare_parameter('max_lookahead')

        # Class Variables
        self.sparse_waypoint_filename = self.get_parameter('sparse_waypoint_filename').value
        drive_topic = self.get_parameter('drive_topic').value
        traj_topic = self.get_parameter('trajectory_topic').value
        global_traj_topic = self.get_parameter('global_trajectory_topic').value
        odom_topic = self.get_parameter('odometry_topic').value
        wp_vis_topic = self.get_parameter('waypoint_vis_topic').value
        self.waypoint_distance = self.get_parameter('waypoint_distance').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.lookahead_method = self.get_parameter('lookahead_method').value
        use_global_traj = self.get_parameter('use_global_traj').value

        # Subscribers & Publishers
        if use_global_traj:
            self.traj_sub = self.create_subscription(JointTrajectory, global_traj_topic, self.traj_callback, 1)
        else:
            self.traj_sub = self.create_subscription(JointTrajectory, traj_topic, self.traj_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.wp_vis_pub = self.create_publisher(Marker, wp_vis_topic, 1)

        # Path
        self.trajectory_up = False

    def find_cur_idx(self, odom_msg):
        position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        point_dist = np.linalg.norm(self.path[:, 0:2] - position, axis=1)
        cur_idx = np.argmin(point_dist)
        return cur_idx

    def compute_lookahead(self, cur_idx, method='curvature'):
        if method == 'velocity':
            # Velocity-Based Lookahead
            min_vel = np.min(self.vel)
            max_vel = np.max(self.vel)
            lookahead = np.interp(self.vel[cur_idx],
                                  np.array([min_vel, max_vel]),
                                  np.array([self.min_lookahead, self.max_lookahead]))
        elif method == 'curvature':
            # Curvature-Based Lookahead
            min_curv = np.min(abs(self.curv))
            max_curv = np.max(abs(self.curv))
            lookahead = np.interp(self.curv[cur_idx],
                                  np.array([min_curv, max_curv]),
                                  np.array([self.min_lookahead, self.max_lookahead]))
        else:
            # Constant Lookahead
            lookahead = self.min_lookahead
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
        xy = data[:, 1:3]
        curvature = data[:, 4]
        velocity = data[:, 5]
        return xy, curvature, velocity

    def path_to_array(self, path):
        """
        """
        # Unpack Path into Array of [x, y, yaw]
        arr = []
        for ps in path.poses:
            quat = [ps.pose.orientation.x, ps.pose.orientation.y,
                    ps.pose.orientation.z, ps.pose.orientation.w]
            _, _, yaw = euler_from_quaternion(quat)
            arr.append([ps.pose.position.x, ps.pose.position.y, yaw])

        return arr

    def generate_waypoint_path(self, sparse_points, waypoint_distance, skip_header=3):
        """
        Callback for path service.
        """
        # Spline Interpolate Sparse Path
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

    def publish_drive_msg(self, desired_angle, speed):
        """
        """
        # Compute Control Input
        angle = np.clip(desired_angle,
                        -self.get_parameter('steering_angle_bound').value,
                        self.get_parameter('steering_angle_bound').value)
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = self.get_parameter('steering_angle_factor').value * angle
        msg.drive.speed = self.get_parameter('speed_factor').value * speed
        self.drive_pub.publish(msg)

        return 0

    def publish_waypoint_vis_msg(self, x, y):
        """
        """
        msg = wp_vis_msg([x, y], self.get_clock().now().to_msg())
        self.wp_vis_pub.publish(msg)
    
    def traj_callback(self, traj_msg):
        """
        """
        if not self.trajectory_up:
            self.get_logger().info('Trajectory publisher available.')
            self.trajectory_up = True
        self.path = np.array([[pt.positions[1], pt.positions[2]] for pt in traj_msg.points])
        # self.path = np.concatenate(([self.pos], self.path))
        self.s = np.array([pt.positions[0] for pt in traj_msg.points])
        self.vel = np.array([pt.velocities[0] for pt in traj_msg.points])
        self.acc = np.array([pt.accelerations[0] for pt in traj_msg.points])
        self.curv = np.array([pt.effort[0] for pt in traj_msg.points])        

    def pose_callback(self, odom_msg):
        """
        """
        # Save Odometry
        self.pos = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]
        # Break if no trajectory
        if not self.trajectory_up:
            self.get_logger().debug('No trajectory available.')
            return None
        # Identify current index position on map
        cur_idx = self.find_cur_idx(odom_msg)
        # Obtain appropriate lookahead
        lookahead = self.compute_lookahead(cur_idx, self.lookahead_method)
        # get waypoint to follow based on new lookahead
        x_w, y_w = self.get_cur_waypoint(cur_idx, odom_msg, lookahead)
        # Transform goal point to vehicle frame of reference
        goal_x_body, goal_y_body = self.transform_point(odom_msg, x_w, y_w)
        # Calculate curvature/steering angle
        desired_angle = self.compute_steering_angle(odom_msg, goal_y_body, lookahead)
        # Publish drive message, don't forget to limit the steering angle.
        self.publish_drive_msg(desired_angle, self.vel[cur_idx])
        self.publish_waypoint_vis_msg(x_w, y_w)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
