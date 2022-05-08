#!/usr/bin/env python3
from visualization_helpers import *
from nav_msgs.msg import Odometry
from graph_ltpl_helpers import get_path_dict, get_traj_line, import_global_traj
import graph_ltpl
from graph_ltpl.helper_funcs.src.get_s_coord import get_s_coord
from graph_ltpl.imp_global_traj.src.import_globtraj_csv import import_globtraj_csv
from trajectory_planning_helpers.calc_splines import calc_splines
from trajectory_planning_helpers.interp_splines import interp_splines
from trajectory_planning_helpers.calc_head_curv_an import calc_head_curv_an
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
        self.declare_parameter('start_vel')

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
        self.start_vel = self.get_parameter('start_vel').value

        # Class Members
        self.pos = None
        self.vel = None
        self.yaw = None
        self.traj_set = {'straight': None}
        self.tic = self.get_clock().now()
        self.graph_ltpl_up = False
        self.start_path = None
        self.path_reached = False

        # Subscribers, Publishers, & Timers
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.traj_pub = self.create_publisher(JointTrajectory, traj_topic, 1)
        if self.is_publish_global_traj:
            self.global_traj_pub = self.create_publisher(JointTrajectory, global_traj_topic, 1)
        self.global_traj_vis_pub = self.create_publisher(Marker, global_traj_vis_topic, 1)
        self.local_traj_vis_pub = self.create_publisher(Marker, local_traj_vis_topic, 1)

        # Global Trajectory
        self.global_traj_timer = self.create_timer(1.0, self.publish_global_traj)

    def import_global_traj(self, import_path):
        self.refline, _, _, self.norm_vec, self.alpha, self.s, _, self.kappa_rl, self.vel_rl, self.acc_rl = \
            import_global_traj(import_path=import_path)

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
        self.import_global_traj(import_path=path_dict['globtraj_input_path'])
        self.traj_line = get_traj_line(self.refline, self.norm_vec, self.alpha)

        # Set Start Position
        is_in_track = self.ltpl_obj.set_startpos(pos_est=self.pos,
                                                 heading_est=self.yaw + self.yaw_offset)
        if is_in_track:
            self.get_logger().info('Vehicle in track, LTPL initialized.')
            self.graph_ltpl_up = True
        else:
            self.get_logger().info('Vehicle not in track.')
    
    def compute_start_path(self, pos, yaw, vel, path):
        self.get_logger().info('Computing start trajectory.')
        x_coeff, y_coeff, _, normvec = calc_splines(path=np.array([pos, path[0, 1:3]]), 
                                                    psi_s=yaw, 
                                                    psi_e=path[0, 3])
        xy, splinds, tvals, _ = interp_splines(coeffs_x=x_coeff, 
                                               coeffs_y=y_coeff, 
                                               stepsize_approx=np.mean(np.diff(path[:, 0])))
        s = np.array([get_s_coord(self.refline, point)[0] for point in xy])
        psi, kappa  = calc_head_curv_an(coeffs_x=x_coeff, 
                                        coeffs_y=y_coeff, 
                                        ind_spls=splinds,
                                        t_spls=tvals)
        # v = np.linspace(vel[0], path[0, 5], len(xy))
        v = self.start_vel * np.ones((len(xy),))
        a = np.concatenate(([0], np.diff(v)))
        self.start_path = np.concatenate((np.array([s]).T, xy, np.array([psi]).T, 
                                          np.array([kappa]).T, np.array([v]).T, np.array([a]).T), axis=1)
    
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
        if not self.path_reached:
            cur_s, _ = get_s_coord(ref_line=self.refline, pos=self.pos, s_array=self.s)
            start_s, _ = get_s_coord(ref_line=self.refline, pos=local_path[0, 1:3], s_array=self.s)
            if cur_s < start_s:
                if self.start_path is None:
                    self.compute_start_path(pos=self.pos, yaw=self.yaw+self.yaw_offset, vel=self.vel, path=local_path)
                local_path = np.concatenate((self.start_path, local_path))
            else:
                self.path_reached = True
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
        # Break if Trajectory Not Available
        if not self.graph_ltpl_up:
            return None
        
        # Publish
        if self.is_publish_global_traj:
            global_traj = np.concatenate((np.zeros((len(self.traj_line), 1)), self.traj_line, 
                                        np.zeros((len(self.traj_line), 1)), np.array([self.kappa_rl]).T, 
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
