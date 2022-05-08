#!/usr/bin/env python3
from math import sin, cos, radians
import numpy as np
from laser_scan_helpers import get_point_at_angle


def adaptive_breakpoint_detection(r, lam, sigma=0.0, amin=radians(-135), amax=radians(135), ainc=radians(0.25)):
    """
    Implements the adaptive breakpoint detection algorithm proposed by Borges et al. in "Line extraction in 2D range 
    images for mobile robotics" (2004).  

    :param r: LiDAR ranges
    :type r: np.ndarray
    :param lam: Worst-case incidence angle on a line for point detection in range [0, pi/2] radians
    :type lam: float 
    :param sigma: Lidar distance resolution
    :type sigma: float 
    :param amin: Minimum LiDAR angle in radians
    :type amin: float
    :param amax: Maximum LiDAR angle in radians
    :type amax: float
    :param ainc: LiDAR angle increment between beams in radians
    :type ainc: float
    :returns: 
        * **b** - np.ndarray - (n,) array with ones signifying breakpoints, zeros otherwise
        * **p** - np.ndarray - (n, 2) array of points corresponding to LiDAR beams
    """

    # Input Validation
    assert not np.any(np.logical_or(np.isinf(r), np.isnan(r))), 'r cannot contain infs or nans.'
    assert amin <= amax, 'amin must be less than amax.'
    assert ainc <= amax - amin, 'ainc must be feasible in range of amin to amax.'

    # Adaptive Breakpoint Detection
    n = len(r)
    phi = amin
    b = np.zeros((n,))
    p = np.zeros((n, 2))
    p[0] = np.asarray(get_point_at_angle(r, phi, amin, amax, ainc))
    for i in range(1, n):
        phi += ainc
        p[i] = np.asarray(get_point_at_angle(r, phi, amin, amax, ainc))
        Dmax = r[i-1] * sin(ainc) / sin(lam - ainc) + 3 * sigma
        if np.linalg.norm(p[i] - p[i-1]) > Dmax:
            b[i-1] = 1
            b[i] = 1
    # min_idx = int((np.radians(-90) - amin)//ainc + 1)
    # b[0:min_idx] = 0
    # b[-min_idx:-1] = 0
    return b, p


def get_clusters(b, p):
    """
    Gathers clusters from breakpoints.

    :param b: Array of breakpoints
    :type b: np.ndarray
    :param p: (n, 2) array of points in body frame
    :type p: np.ndarray
    :return: Ordered clusters of points
    :rtype: tuple
    """
    # Input Validation
    assert len(b) == len(p), 'Array lengths must match.'

    # Compute Clusters
    n = np.shape(b)[0]
    c = ([p[0]],)
    cidx = 0
    for i in range(1, n):
        if b[i] and b[i-1]:
            # c = (c, [p[i]])
            if c[cidx]:
                c = c + ([p[i]],)
                cidx += 1
            else:
                c[cidx].append(p[i])
        else:
            c[cidx].append(p[i])
    return c


def find_opp_bounds(r, disp, amin=radians(-135), amax=radians(135), ainc=radians(0.25)):
    """
    """
    pass


def estimate_opponent_pose():
    """
    """
    pass


