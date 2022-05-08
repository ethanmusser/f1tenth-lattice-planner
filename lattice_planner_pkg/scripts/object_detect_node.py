#!/usr/bin/env python3
from dis import dis
from visualization_helpers import *
from opp_pose_estimation import *
from graph_ltpl.online_graph.src.check_inside_bounds import check_inside_bounds
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.interpolate import splprep, splev
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from ament_index_python.packages import get_package_share_directory
import pathlib


class ObjectDetect(Node):
    """ 
    Implements static and dynamic object detection on the car
    """

    def __init__(self):
        super().__init__('object_detect_node')

        # Declare Parameters
        self.declare_parameter('lidar_proc_max_dist')
        self.declare_parameter('disparity_threshold')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('laserscan_topic')
        self.declare_parameter('obstacle_vis_topic')
        self.declare_parameter('disparitiy_vis_topic')
        self.declare_parameter('clusters_vis_topic')
        self.declare_parameter('gap_threshold')
        self.declare_parameter('lambda')
        self.declare_parameter('sigma')
        self.declare_parameter('bound_offset')

        # Class Variables
        self.lidar_max_dist = self.get_parameter('lidar_proc_max_dist').value
        self.disparity_threshold = self.get_parameter('disparity_threshold').value
        self.gap_threshold = self.get_parameter('gap_threshold').value
        self.lamb = self.get_parameter('lambda').value
        self.sigma = self.get_parameter('sigma').value
        self.bound_offset = self.get_parameter('bound_offset').value

        #topics
        odom_topic = self.get_parameter('odometry_topic').value
        laserscan_topic = self.get_parameter('laserscan_topic').value
        obstacle_vis_topic = self.get_parameter('obstacle_vis_topic').value
        disparity_vis_topic = self.get_parameter('disparitiy_vis_topic').value
        clusters_vis_topic = self.get_parameter('clusters_vis_topic').value

        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.laser_sub = self.create_subscription(LaserScan, laserscan_topic, self.lidar_callback, 10)
        self.obstacle_vis_pub = self.create_publisher(MarkerArray, obstacle_vis_topic, 1)
        self.disparity_vis_pub = self.create_publisher(MarkerArray, disparity_vis_topic, 1)
        self.clusters_vis_pub = self.create_publisher(MarkerArray, clusters_vis_topic, 1)

        #track bounds:
        # bound1 = (self.__graph_base.refline + self.__graph_base.normvec_normalized
        #           * np.expand_dims(self.__graph_base.track_width_right, 1))
        # bound2 = (self.__graph_base.refline - self.__graph_base.normvec_normalized
        #           * np.expand_dims(self.__graph_base.track_width_left, 1))

        #TODO: import map that has inside and outside bounds specified

    def preprocess_lidar(self, ranges, angle_inc, range_min):
        """
        Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)

        Args:
            ranges

        Returns:
            ranges: processed ranges array

        """
        ranges = np.array(ranges) 
        ranges[np.isnan(ranges)] = 0.0
        ranges = np.convolve(ranges,np.ones(3),'same')/3
        ranges = np.clip(ranges, range_min, self.lidar_max_dist)
        return ranges

    def find_disparities(self, ranges, angle_inc):
        """
        Finds all disparities usind processed lidar data 

        Args: processed lidar data

        Returns: indices in lidar scan that represent disparities
        """
        disparities = np.nonzero(np.diff(ranges) > self.disparity_threshold)[0]
        # disparities = np.append(disparities,[np.argmin(ranges)])
        min_idx = np.radians(45)//angle_inc + 1
        max_idx = len(ranges) - min_idx
        disparities = disparities[disparities > min_idx]
        disparities = disparities[disparities < max_idx]
        # print('disparities', disparities)
        # disparities = disparites >
        #convert indices to x,y pos from car frame
        if len(disparities) != 0:
            angle = angle_inc * disparities - (3*np.pi/4)
            car_x = np.take(ranges, [disparities])[0] * np.cos(angle)
            car_y = np.take(ranges, [disparities])[0] * np.sin(angle)
            # print('disparities', disparities)
            return car_x, car_y
        else:
            # print('no disparities')
            return [], []

    def postprocess_clusters(self, clusters):
        print('clusters shape', len(clusters))
        new_cluster = ()
        for cluster in clusters:
            for i in range(len(cluster)):
                if not (check_inside_bounds(self.bound1, self.bound2, cluster[i])):
                    print('out of bounds')
                    break
            print('valid cluster')
        # return new_cluster

    #Callback functions
    def lidar_callback(self, scan_msg):
        proc_ranges = self.preprocess_lidar(scan_msg.ranges, scan_msg.angle_increment, scan_msg.range_min)
        b, p = adaptive_breakpoint_detection(scan_msg.ranges, self.lamb, self.sigma, scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        self.clusters = get_clusters(b, p)        
        self.postprocess_clusters(self.clusters)
        
        #disparity approach
        self.car_x, self.car_y = self.find_disparities(proc_ranges, scan_msg.angle_increment)

    def odom_callback(self, odom_msg):
        #uncomment below code for disparity approach
        # self.publish_disparities_vis(odom_msg)
        #uncomment below function for clusters approach
        self.visualize_clusters(odom_msg, self.clusters)

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

    #visualization functions:
    def visualize_clusters(self, odom_msg, clusters):
        cluster_world = []
        for i in range(len(clusters)):
            x_begin_w, y_begin_w = self.transform_car_to_global(odom_msg, clusters[i][0][0], clusters[i][0][1])
            x_end_w, y_end_w = self.transform_car_to_global(odom_msg, clusters[i][-1][0], clusters[i][-1][1])
            cluster_world.append([x_begin_w, y_begin_w])
            cluster_world.append([x_end_w, y_end_w])
        if len(cluster_world) > 0:
            print('clusters world shape', np.shape(cluster_world))
            clusters_marker_msg = wp_map_pt_vis_msg(cluster_world, self.get_clock().now().to_msg(),
                                               rgba=[0.0, 255.0, 255.0, 0.8], dur = Duration(seconds =0.3).to_msg())
            self.clusters_vis_pub.publish(clusters_marker_msg)

    def publish_disparities_vis(self, odom_msg):
        #publishes disparities on rviz
        disparities_world = np.zeros((len(self.car_x), 2))
        for i in range(len(disparities_world)):
            x, y = self.transform_car_to_global(odom_msg, self.car_x[i], self.car_y[i]) 
            disparities_world[i] = [x, y]
        print('disparity world', disparities_world)
        disparity_marker_msg = wp_map_pt_vis_msg(disparities_world, self.get_clock().now().to_msg(),
                                               rgba=[0.0, 255.0, 255.0, 0.8], dur = Duration(seconds =0.3).to_msg())
        if len(disparities_world) > 0:
            self.disparity_vis_pub.publish(disparity_marker_msg)


    #TODO functions

    def obstacle_detect(self, disparities):
        pass

    def obstacle_pos(self, data):
        """
        Filters through gaps between disparities to find obstacles. Finds location of obstacle
        Args: 
            disparities indices
            processed lidar data
        
        Returns:
            array of [x,y] of all obstacles
        """
        pass
    
    def local_planner_inputs(self):
        pass

    def stat_dynam_grouping(self):
        pass
    
    def publish_drive_msg(self, desired_angle, speed):
        """
        """
        
        return 0
    
    def traj_callback(self, traj_msg):
        """
        """
        pass

    def pose_callback(self, odom_msg):
        """
        """
        pass


def main(args=None):
    rclpy.init(args=args)
    print("object_detect_node Initialized")
    object_detect_node = ObjectDetect()
    rclpy.spin(object_detect_node)

    object_detect_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
