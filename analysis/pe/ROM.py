"""
Range of Motion (ROM) Calculation Module

This module provides functions to compute the range of motion and standard deviation
for key joints from pose keypoint data, using joint definitions and conventions from dataprocessing.pe.py.
Designed for integration with PE_pipeline.py.
"""

import numpy as np

# Import joint definitions (ANGLE_JOINTS) and landmark names from dataprocessing.pe
from constants.pe import ANGLE_JOINTS, LANDMARK_NAMES
from utils.math_utils import compute_angle_3points, calculate_range_of_motion

def compute_joint_angles(df):
    """
    Compute joint angles for all frames in the DataFrame.
    Returns a dictionary: {joint_name: [angle_per_frame]}
    """
    joint_angles = {joint: [] for joint in ANGLE_JOINTS}
    for idx, row in df.iterrows():
        for joint, (a_name, b_name, c_name) in ANGLE_JOINTS.items():
            try:
                a = [row[f"{a_name}_x"], row[f"{a_name}_y"]]
                b = [row[f"{b_name}_x"], row[f"{b_name}_y"]]
                c = [row[f"{c_name}_x"], row[f"{c_name}_y"]]
                angle = compute_angle_3points(np.array(a), np.array(b), np.array(c))
            except KeyError:
                angle = np.nan
            joint_angles[joint].append(angle)
    return joint_angles

def compute_range_of_motion_and_std(df):
    """
    Compute the range of motion (ROM) and standard deviation (STD) for each joint over the entire DataFrame.
    Returns a dictionary: {joint_name: {'rom': ROM_value, 'std': STD_value}}
    """
    joint_angles = compute_joint_angles(df)
    stats = {}
    for joint, angles in joint_angles.items():
        angles = np.array(angles)
        valid_angles = angles[~np.isnan(angles)]
        if valid_angles.size > 0:
            min_angle, max_angle, rom = calculate_range_of_motion(valid_angles)
            std = float(np.std(valid_angles))
        else:
            rom = np.nan
            std = np.nan
        stats[joint] = {'rom': rom, 'std': std}
    return stats

def compute_peak_angular_velocities(df):
    """
    Compute the peak angular velocity for each joint over the entire DataFrame.
    Returns a dictionary: {joint_name: peak_angular_velocity_value}
    """
    joint_angles = compute_joint_angles(df)
    peak_velocities = {}
    for joint, angles in joint_angles.items():
        angles = np.array(angles)
        valid_angles = angles[~np.isnan(angles)]
        if valid_angles.size > 1:
            angular_vel = np.abs(np.diff(valid_angles))
            peak_velocities[joint] = float(np.max(angular_vel))
        else:
            peak_velocities[joint] = np.nan
    return peak_velocities

def print_rom_stats(df):
    """
    Compute and print the Range of Motion per joint (ROM ± STD) for the given DataFrame.
    """
    stats = compute_range_of_motion_and_std(df)
    print("Range of Motion per joint:")
    for joint, values in stats.items():
        print(f"{joint}: {values['rom']:.2f} ± {values['std']:.2f} degrees")

# Example usage (for testing or as a script)
if __name__ == "__main__":
    # Assume df is defined elsewhere or loaded here for testing
    print_rom_stats(df)