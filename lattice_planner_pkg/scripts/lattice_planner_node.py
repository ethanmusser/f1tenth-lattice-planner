#!/usr/bin/env python3
from visualization_helpers import *
from nav_msgs.msg import Odometry
from graph_ltpl_helpers import get_path_dict, get_traj_line
import graph_ltpl
import numpy as np
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from time import sleep
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Point

# System
import os
os.environ['OPENBLAS_NUM_THREADS'] = str(1)


class LatticePlanner(Node):

    def __init__(self):
        super().__init__('lattice_planner')

        # Parameter Declarations
        self.declare_parameter('toppath')
        self.declare_parameter('mappath')
        self.declare_parameter('track_specifier')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('trajectory_topic')
        self.declare_parameter('global_trajectory_topic')
        self.declare_parameter('global_traj_vis_topic')
        self.declare_parameter('local_traj_vis_topic')
        self.declare_parameter('visual_mode')
        self.declare_parameter('log_mode')
        self.declare_parameter('publish_global_traj')
        self.declare_parameter('yaw_offset')

        # Read Parameters
        self.toppath = self.get_parameter('toppath').value
        self.track_specifier = self.get_parameter('track_specifier').value
        self.mappath = self.get_parameter('mappath').value
        odom_topic = self.get_parameter('odometry_topic').value
        traj_topic = self.get_parameter('trajectory_topic').value
        global_traj_topic = self.get_parameter('global_trajectory_topic').value
        global_traj_vis_topic = self.get_parameter('global_traj_vis_topic').value
        local_traj_vis_topic = self.get_parameter('local_traj_vis_topic').value
        self.visual_mode = self.get_parameter('visual_mode').value
        self.log_mode = self.get_parameter('log_mode').value
        self.is_publish_global_traj = self.get_parameter('publish_global_traj').value
        self.yaw_offset = self.get_parameter('yaw_offset').value

        # Class Members
        self.pos = None
        self.vel = None
        self.yaw = None
        self.traj_set = {'straight': None}
        self.tic = self.get_clock().now()
        self.graph_ltpl_up = False

        # Subscribers, Publishers, & Timers
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.traj_pub = self.create_publisher(JointTrajectory, traj_topic, 1)
        if self.is_publish_global_traj:
            self.global_traj_pub = self.create_publisher(JointTrajectory, global_traj_topic, 1)
        self.global_traj_vis_pub = self.create_publisher(Marker, global_traj_vis_topic, 1)
        self.local_traj_vis_pub = self.create_publisher(Marker, local_traj_vis_topic, 1)

        # Global Trajectory
        self.global_traj_timer = self.create_timer(1.0, self.publish_global_traj)

    def initialize_graph_ltpl(self):
        # Intialize Graph_LTPL Class
        path_dict = get_path_dict(self.toppath, self.track_specifier)
        self.ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(path_dict=path_dict,
                                                         visual_mode=self.visual_mode,
                                                         log_to_file=self.log_mode)

        # Calculate Offline Graph
        self.ltpl_obj.graph_init()

        # Read Map Params & Trajectory
        map_params = yaml.safe_load(self.mappath + '.yaml')
        self.refline = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[0]
        self.norm_vec = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[3]
        self.alpha = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[4]
        self.vel_rl = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[6]
        self.kappa = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[7]
        self.traj_line = get_traj_line(self.refline, self.norm_vec, self.alpha)

        # Set Start Position
        is_in_track = self.ltpl_obj.set_startpos(pos_est=self.pos,
                                                 heading_est=self.yaw + self.yaw_offset)
        if is_in_track:
            self.get_logger().info('Vehicle in track, LTPL initialized.')
            self.graph_ltpl_up = True
        else:
            self.get_logger().info('Vehicle not in track.')

    def update_local_plan(self):
        # Select Trajectory from List
        # (here: brute-force, replace by sophisticated behavior planner)
        # try to force 'right', else try next in list
        for sel_action in ["right", "left", "straight", "follow"]:
            if sel_action in self.traj_set.keys():
                break

        # Compute Paths for next Time Step
        self.ltpl_obj.calc_paths(prev_action_id=sel_action,
                                 object_list=[])

        # Compute Velocity Profile & Retrieve Tajectories
        self.traj_set = self.ltpl_obj.calc_vel_profile(pos_est=self.pos,
                                                       vel_est=self.vel)[0]

        # Publish Selected Trajectory
        local_path = np.array(self.traj_set[sel_action][0])
        self.traj_pub.publish(self.traj_msg(local_path))

        # Visualize Trajectory
        local_marker_msg = wp_map_line_vis_msg(local_path[:,1:3], self.get_clock().now().to_msg(),
                                               rgba=[0.0, 0.0, 255.0, 0.8])
        self.local_traj_vis_pub.publish(local_marker_msg)

        # Log & Visualize (if enabled)
        self.ltpl_obj.log()
        self.ltpl_obj.visual()

    def traj_msg(self, traj):
        # Write Local Trajectory to ROS Message
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        for wp in traj:
            pt = JointTrajectoryPoint()
            pt.positions.append(wp[0])
            pt.positions.append(wp[1])
            pt.positions.append(wp[2])
            pt.velocities.append(wp[5])
            pt.accelerations.append(wp[6])
            pt.effort.append(wp[4]) # use effort to store curvature
            msg.points.append(pt)
        return msg

    def publish_global_traj(self):
        # Publish
        if self.is_publish_global_traj:
            global_traj = np.concatenate((np.zeros((len(self.traj_line), 1)), self.traj_line, 
                                        np.zeros((len(self.traj_line), 1)), np.array([self.kappa]).T, 
                                        np.array([self.vel_rl]).T, np.zeros((len(self.traj_line), 1))), axis=1)
            self.global_traj_pub.publish(self.traj_msg(global_traj))

        # Visualization
        global_traj_marker_msg = wp_map_line_with_vel_vis_msg(
            self.traj_line, self.vel_rl, self.get_clock().now().to_msg(), wrap=True, scale=0.05)
        self.global_traj_vis_pub.publish(global_traj_marker_msg)

    def odom_callback(self, odom_msg):
        # Convert Odom
        pose = odom_msg.pose.pose
        twist = odom_msg.twist.twist
        self.pos = [pose.position.x, pose.position.y]
        self.vel = [twist.linear.x, twist.linear.y]
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        _, _, self.yaw = euler_from_quaternion(quat)

        # First-Time Graph-LTPL Setup
        if not self.graph_ltpl_up:
            self.initialize_graph_ltpl()

        # Run Local Planner & Publish Commands
        self.update_local_plan()


def main(args=None):
    rclpy.init(args=args)
    lattice_planner = LatticePlanner()
    rclpy.spin(lattice_planner)
    lattice_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
