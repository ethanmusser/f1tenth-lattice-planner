#!/usr/bin/env python3
from visualization_helpers import *
from opp_pose_estimation import *
from graph_ltpl.online_graph.src.check_inside_bounds import check_inside_bounds
from graph_ltpl_helpers import get_path_dict, get_traj_line, import_global_traj
import rclpy
from rclpy.node import Node
import numpy as np
import copy
from std_msgs.msg import String
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import quaternion_from_euler, quaternion_matrix
from geometry_msgs.msg import PoseArray, Pose

class ObjectDetect(Node):
    """ 
    Implements static and dynamic object detection on the car
    """

    def __init__(self):
        super().__init__('object_detect_node')

        # Declare Parameters
        self.declare_parameter('lidar_proc_max_dist')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('laserscan_topic')
        self.declare_parameter('toppath_topic')
        self.declare_parameter('map_spec_topic')
        self.declare_parameter('obstacle_vis_topic')
        self.declare_parameter('clusters_vis_topic')
        self.declare_parameter('opponent_list_topic')
        self.declare_parameter('gap_threshold')
        self.declare_parameter('lambda')
        self.declare_parameter('sigma')
        self.declare_parameter('bound_offset')
        self.declare_parameter('opponent_offset_x')
        self.declare_parameter('opponent_offset_y')

        # Class Variables
        self.lidar_max_dist = self.get_parameter('lidar_proc_max_dist').value
        self.gap_threshold = self.get_parameter('gap_threshold').value
        self.lamb = self.get_parameter('lambda').value
        self.sigma = self.get_parameter('sigma').value
        self.bound_offset = self.get_parameter('bound_offset').value
        self.opponent_offset_x = self.get_parameter('opponent_offset_x').value
        self.opponent_offset_y = self.get_parameter('opponent_offset_y').value

        #topics
        odom_topic = self.get_parameter('odometry_topic').value
        laserscan_topic = self.get_parameter('laserscan_topic').value
        toppath_topic = self.get_parameter('toppath_topic').value
        map_spec_topic = self.get_parameter('map_spec_topic').value
        obstacle_vis_topic = self.get_parameter('obstacle_vis_topic').value
        clusters_vis_topic = self.get_parameter('clusters_vis_topic').value
        opp_list_topic = self.get_parameter('opponent_list_topic').value
        
        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.laser_sub = self.create_subscription(LaserScan, laserscan_topic, self.lidar_callback, 1)
        self.toppath_sub = self.create_subscription(String, toppath_topic, self.toppath_callback, 1)
        self.map_spec_sub = self.create_subscription(String, map_spec_topic, self.map_spec_callback, 1)
        self.obstacle_vis_pub = self.create_publisher(MarkerArray, obstacle_vis_topic, 1)
        self.clusters_vis_pub = self.create_publisher(MarkerArray, clusters_vis_topic, 1)
        self.opp_list_pub = self.create_publisher(PoseArray, opp_list_topic, 1)

        # Track Trajectory
        self.toppath = None
        self.map_spec = None
        self.path_dict = None
        self.refline = None
        self.w_right = None
        self.w_left = None
        self.norm_vec = None
        self.s = None
        self.psi = None
        self.vel_rl = None
        self.bound1 = None
        self.bound2 = None
        self.clusters_body = None

    def postprocess_clusters(self, clusters):
        proc_indices = []
        for i in range(len(clusters)):
            out_of_bounds = False
            for j in range(len(clusters[i])):
                if not (check_inside_bounds(self.bound1, self.bound2, clusters[i][j])):
                    out_of_bounds = True
                    break
            if not out_of_bounds:
                proc_indices.append(i)
        return proc_indices

    #Callback functions
    def lidar_callback(self, scan_msg):
        if self.bound1 is None:
            return
        #Find all breakpoints
        b, p = adaptive_breakpoint_detection(scan_msg.ranges, self.lamb, self.sigma, scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        #Find all clusters from breakpoitns
        self.clusters_body = get_clusters(b, p)

    def odom_callback(self, odom_msg):
        if self.clusters_body is None:
            return
        #convert clusters to world frame
        self.clusters_world = self.body_clusters_to_world(odom_msg, self.clusters_body)
        #find indicies of best clusters
        proc_indices = self.postprocess_clusters(self.clusters_world)
        #create new clusters based on processing
        self.proc_clusters_world = tuple(self.clusters_world[i] for i in proc_indices)
        self.proc_clusters_body = tuple(self.clusters_body[i] for i in proc_indices)
        #visualize clusters
        self.visualize_clusters(self.proc_clusters_world)
        #find obstacles
        p_cm_w, phi_w = self.estimate_opponent_pose(odom_msg, self.proc_clusters_body, self.opponent_offset_x, self.opponent_offset_y)
        print(p_cm_w)
        #visualize obstacles
        self.visualize_obstacles(p_cm_w)
        #publish opponent list
        self.publish_opponent_list(p_cm_w, phi_w)
    
    def publish_opponent_list(self, p_cm, phi_w):
        msg = PoseArray()
        for p, phi in zip(p_cm, phi_w):
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            quat = quaternion_from_euler(0.0, 0.0, phi, 'sxyz')
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = 0.0
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            msg.poses.append(pose)
        self.opp_list_pub.publish(msg)




    def toppath_callback(self, msg):
        if self.toppath is None:
            self.toppath = msg.data
        if self.path_dict is None and self.toppath is not None and self.map_spec is not None:
            self.path_dict = get_path_dict(self.toppath, self.map_spec)
            self.refline, self.w_right, self.w_left, self.norm_vec, _, self.s, self.psi, _, self.vel_rl, _ = \
                import_global_traj(import_path=self.path_dict['globtraj_input_path'])
            self.bound1 = (self.refline + self.norm_vec * np.expand_dims(self.w_right - self.bound_offset, 1))
            self.bound2 = (self.refline - self.norm_vec * np.expand_dims(self.w_left - self.bound_offset, 1))

    def map_spec_callback(self, msg):
        if self.map_spec is None:
            self.map_spec = msg.data

    def transform_car_to_global(self, odom_msg, goal_x, goal_y):
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
        goal_b = np.array([[goal_x],[goal_y], [0], [1]])
        goal_m = tform_b_m.dot(goal_b).flatten()
        return goal_m[0], goal_m[1]

    def estimate_opponent_pose(self, odom_msg, clusters, xoff, yoff):
        """
        Returns theta and opponent position (x,y)
        """
        p_cm_w_all = []
        phi_w_all = []
        for cluster in clusters:
            p_i1 = np.array(cluster[0])
            p_i = np.array(cluster[-1])
            p_c_b = 0.5 * (p_i + p_i1)
            p_c_w = self.transform_car_to_global(odom_msg, p_c_b[0], p_c_b[1])
            r_diff = p_i - p_i1
            r_f_b = np.cross(np.concatenate((r_diff, [0])), np.array([0,0,1])) 
            r_f_w = self.transform_car_to_global(odom_msg, r_f_b[0], r_f_b[1])
            r_f_w = r_f_w / np.linalg.norm(r_f_w)
            p_cm_w = p_c_w + np.array([xoff, yoff]) @ r_f_w
            phi_w = np.arctan2(r_f_w[1], r_f_w[0])
            p_cm_w_all.append(p_cm_w)
            phi_w_all.append(phi_w)
        return p_cm_w_all, phi_w_all

    def body_clusters_to_world(self, odom_msg, clusters):
        clusters_world = copy.deepcopy(clusters)
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                clusters_world[i][j][0], clusters_world[i][j][1] = self.transform_car_to_global(odom_msg, clusters[i][j][0] , clusters[i][j][1])
        return clusters_world

    #visualization functions:
    def visualize_clusters(self, clusters):
        cluster_world = []
        for i in range(len(clusters)):
            begin = [clusters[i][0][0], clusters[i][0][1]]
            end = [clusters[i][-1][0], clusters[i][-1][1]]
            cluster_world.append(begin)
            cluster_world.append(end)
        if len(cluster_world) > 0:
            clusters_marker_msg = wp_map_pt_vis_msg(cluster_world, self.get_clock().now().to_msg(),
                                               rgba=[0.0, 255.0, 255.0, 0.8], dur = Duration(seconds =0.3).to_msg())
            self.clusters_vis_pub.publish(clusters_marker_msg)

    def visualize_obstacles(self, p_cm, rgba=[255.0, 255.0, 255.0, 0.8]):
        if len(p_cm) > 0:
            obstacle_marker_msg = wp_map_pt_vis_msg(p_cm, self.get_clock().now().to_msg(),
                                               rgba, dur = Duration(seconds =0.3).to_msg())
            self.obstacle_vis_pub.publish(obstacle_marker_msg)
    

def main(args=None):
    rclpy.init(args=args)
    print("object_detect_node Initialized")
    object_detect_node = ObjectDetect()
    rclpy.spin(object_detect_node)

    object_detect_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
