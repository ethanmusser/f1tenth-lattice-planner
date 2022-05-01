#!/usr/bin/env python3
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def wp_vis_msg(xy, ts, rgba=[255.0, 0.0, 0.0, 1.0], scale=0.3,
                   dur=Duration(seconds=0.5).to_msg(), frame='map', log=False):
    """
    """
    if log:
        print(
            f"Publishing waypoint x = {xy[0]}, y = {xy[1]}, r = {rgba[0]}, g = {rgba[1]}, b = {rgba[2]}")
    marker = Marker()
    marker.header.stamp = ts
    marker.header.frame_id = frame
    marker.type = marker.SPHERE
    marker.pose.position.x = xy[0]
    marker.pose.position.y = xy[1]
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]
    marker.lifetime = dur
    
    return marker


def wp_map_pt_vis_msg(path, ts, rgba=[255.0, 0.0, 0.0, 1.0], scale=0.1, 
                       dur=None, frame='map'):
    """
    """
    marker_array = MarkerArray()
    for idx, ps in enumerate(path):
        marker = Marker()
        marker.header.stamp = ts
        marker.header.frame_id = frame
        marker.id = idx
        # marker.type = marker.SPHERE
        marker.type = marker.LINE_STRIP
        marker.pose.position.x = ps[0]
        marker.pose.position.y = ps[1]
        marker.scale.x = scale
        # marker.scale.y = scale
        # marker.scale.z = scale
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        pt = Point()
        pt.x = marker.pose.position.x
        pt.y = marker.pose.position.y
        marker.points.append(pt)
        if dur:
            marker.lifetime = dur
        marker_array.markers.append(marker)

    return marker_array


def wp_map_line_vis_msg(path, ts, id=0, rgba=[255.0, 0.0, 0.0, 1.0], rgbas=None, scale=0.1, 
                       dur=None, frame='map'):
    """
    """
    marker = Marker()
    marker.header.stamp = ts
    marker.header.frame_id = frame
    marker.id = id
    marker.type = marker.LINE_STRIP
    marker.scale.x = scale
    if dur:
        marker.lifetime = dur
    for idx, ps in enumerate(path):
        pt = Point()
        pt.x = ps[0]
        pt.y = ps[1]
        marker.points.append(pt)
        color = ColorRGBA()
        if rgbas:
            color.r = rgbas[idx, 0]
            color.g = rgbas[idx, 1]
            color.b = rgbas[idx, 2]
            color.a = rgbas[idx, 3]
        else:
            color.r = rgba[0] / 255
            color.g = rgba[1] / 255
            color.b = rgba[2] / 255
            color.a = rgba[3]
        marker.colors.append(color)

    return marker

