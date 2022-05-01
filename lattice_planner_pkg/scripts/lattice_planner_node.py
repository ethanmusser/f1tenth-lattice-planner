#!/usr/bin/env python3
from nav_msgs.msg import Odometry
from graph_ltpl_helpers import get_path_dict
import graph_ltpl
import numpy as np
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from time import sleep
import yaml

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
        self.declare_parameter('local_plan_update_period')

        # Read Parameters
        self.toppath = self.get_parameter('toppath').value
        self.track_specifier = self.get_parameter('track_specifier').value
        self.mappath = self.get_parameter('mappath').value
        odom_topic = self.get_parameter('odometry_topic').value
        update_period = self.get_parameter('local_plan_update_period').value

        # Class Members
        self.pos = None
        self.vel = None
        self.yaw = None
        self.traj_set = {'straight': None}
        self.tic = self.get_clock().now()
        self.graph_ltpl_up = False

        # Subscribers, Publishers, & Timers
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        # self.local_plan_timer = self.create_timer(update_period, self.update_local_plan)

    def initialize_graph_ltpl(self):
        # Intialize Graph_LTPL Class
        path_dict = get_path_dict(self.toppath, self.track_specifier)
        self.ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(path_dict=path_dict,
                                                         visual_mode=False,
                                                         log_to_file=False)

        # Calculate Offline Graph
        self.ltpl_obj.graph_init()

        # Read Map & Refline
        map_params = yaml.safe_load(self.mappath + '.yaml')
        self.refline = graph_ltpl.imp_global_traj.src.import_globtraj_csv.import_globtraj_csv(
            import_path=path_dict['globtraj_input_path'])[0]

        # Set Start Pose (default to first point in reference-line)
        self.pos_est = self.pos
        # self.heading_est = self.yaw - np.pi / 2
        self.heading_est = self.yaw
        self.vel_est = self.vel

        # Set Start Position
        is_in_track = self.ltpl_obj.set_startpos(pos_est=self.pos_est,
                                   heading_est=self.heading_est)
        if is_in_track:
            self.graph_ltpl_up = True

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

        # Send Trajectories to Controller
        # select a trajectory from the set and send it to the controller here
        # self.publish_
        print(self.traj_set)
        print(np.shape(self.traj_set['straight']))

        # Log
        self.ltpl_obj.log()
    
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
        
        # Run Local Planner
        self.update_local_plan()


def main(args=None):
    rclpy.init(args=args)
    lattice_planner = LatticePlanner()
    rclpy.spin(lattice_planner)
    lattice_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
