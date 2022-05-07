#!/usr/bin/env python3
from math import sin, cos, radians
import numpy as np


def get_range_at_a(ranges, a, amin=radians(-135), amax=radians(135), ainc=radians(0.25)):
    """
    Finds the corresponding range measurement at a given a.

    :param ranges: single range array from the LiDAR
    :param a: angle between amin and amax of the LiDAR
    :return: range measurement in meters at the given a
    """
    # Validate Input
    assert amin <= a <= amax, f'Angles must satisy amin <= a <= amax.'

    # Condition Input Parameters
    a = np.clip(a, amin, amax)
    ranges = np.array(ranges)

    # Linearly Interpolate Invalid Range Datapoints
    mask = np.logical_or(np.isnan(ranges), np.isinf(ranges))
    ranges[mask] = np.interp(np.flatnonzero(mask),
                             np.flatnonzero(np.logical_not(mask)),
                             ranges[np.logical_not(mask)])

    # Return Range at the Provided a
    idx = int(round((a - amin) / ainc))
    return ranges[idx]


def get_point_at_angle(ranges, a, amin=radians(-135), amax=radians(135), ainc=radians(0.25)):
    """
    Finds the corresponding (x, y) coordinates of the point at a specified angle in range data.

    :param ranges: range array from LiDAR data
    :param a: valid angle in LiDAR data in radians
    :return: (x, y) coordinate of point
    """
    r = get_range_at_a(ranges, a, amin, amax, ainc)
    return (r * cos(a), r * sin(a))
    
