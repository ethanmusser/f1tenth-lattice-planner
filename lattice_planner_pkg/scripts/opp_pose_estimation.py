#!/usr/bin/env python3
from math import sin, cos, radians
import numpy as np
from laser_scan_helpers import get_point_at_angle


def adaptive_breakpoint_detection(r, lam, amin=radians(-135), amax=radians(135), ainc=radians(0.25)):
    """
    Implements the adaptive breakpoint detection algorithm proposed by Borges et al. in "Line extraction in 2D range 
    images for mobile robotics" (2004).  

    :param r: LiDAR ranges
    :type r: np.ndarray
    :param lam: Worst-case incidence angle on a line for point detection in radians
    :type lam: float 
    :param amin: Minimum LiDAR angle in radians
    :type amin: float
    :param amax: Maximum LiDAR angle in radians
    :type amax: float
    :param ainc: LiDAR angle increment between beams in radians
    :type ainc: float
    :return: Boolean array with ones signifying points adjacent to a breakpoint, zeros otherwise
    :rtype: np.ndarray
    """

    # Input Validation
    assert not np.any(np.logical_or(np.isinf(r), np.isnan(r))), 'r cannot contain infs or nans.'
    assert amin <= amax, 'amin must be less than amax.'
    assert ainc <= amax - amin, 'ainc must be feasible in range of amin to amax.'

    # Adaptive Breakpoint Detection
    n = len(r)
    phi = amin
    p_prev = get_point_at_angle(r, phi, amin, amax, ainc)
    p = None
    b = np.zeros((n,))
    for i in range(1, n):
        phi += ainc
        p = get_point_at_angle(r, phi, amin, amax, ainc)
        Dmax = r[i-1] * sin(ainc) / sin(lam - ainc)
        if np.linalg.norm(p - p_prev) > Dmax:
            b[i-1] = 1
            b[i] = 1
        p_prev = p
    
    return b

