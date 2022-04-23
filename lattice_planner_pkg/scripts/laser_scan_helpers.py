#!/usr/bin/env python3
import numpy as np


def get_range_at_angle(range_data, angle, min_angle=-0.75*np.pi, max_angle=0.75*np.pi):
    """
    Simple helper to return the corresponding range measurement at a given angle.
    Args:
        range_data: single range array from the LiDAR
        angle: between angle_min and angle_max of the LiDAR
    Returns:
        range: range measurement in meters at the given angle
    """
    # Validate Input
    assert min_angle <= max_angle, f'Min. angle {min_angle} rad must be less than max. angle {max_angle} rad.' 

    # Condition Input Parameters
    angle = np.clip(angle, min_angle, max_angle)
    range_data = np.array(range_data)

    # Linearly Interpolate Invalid Range Datapoints
    mask = np.logical_or(np.isnan(range_data), np.isinf(range_data))
    range_data[mask] = np.interp(np.flatnonzero(mask),
                                    np.flatnonzero(np.logical_not(mask)),
                                    range_data[np.logical_not(mask)])

    # Return Range at the Provided Angle
    idx = int(round((angle - min_angle) / max_angle))
    return range_data[idx]
