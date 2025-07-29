import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def check_and_map_columns(df, hand='right'):
    """
    Checks for expected column names and returns a mapping from standard names to actual DataFrame columns.
    Handles alternative naming conventions (e.g., 'rwrist_x' vs 'right_wrist_x').
    Prints warnings for missing columns.
    Args:
        df (pd.DataFrame): DataFrame containing keypoint columns.
        hand (str): Dominant hand, 'right' or 'left'.
    Returns:
        dict: Mapping from standard column names to DataFrame columns.
    """
    keypoints = [
        'wrist', 'hip', 'shoulder', 'ankle', 'knee', 'foot_index', 'elbow'
    ]
    sides = ['left', 'right']
    mapping = {}
    for side in sides:
        for kp in keypoints:
            col_x = f"{side}_{kp}_x"
            col_y = f"{side}_{kp}_y"
            if col_x not in df.columns or col_y not in df.columns:
                # Try alternative naming (e.g., rwrist_x)
                alt_col_x = f"{side[0]}{kp}_x"
                alt_col_y = f"{side[0]}{kp}_y"
                if alt_col_x in df.columns and alt_col_y in df.columns:
                    mapping[col_x] = alt_col_x
                    mapping[col_y] = alt_col_y
                else:
                    print(f"[MAPPING WARNING] Missing columns: '{col_x}' or '{col_y}' in DataFrame.")
            else:
                mapping[col_x] = col_x
                mapping[col_y] = col_y
    # Nose (no left/right)
    if 'nose_x' not in df.columns or 'nose_y' not in df.columns:
        print("[MAPPING WARNING] Missing columns: 'nose_x' or 'nose_y' in DataFrame.")
    else:
        mapping['nose_x'] = 'nose_x'
        mapping['nose_y'] = 'nose_y'
    return mapping

def euclidean_distance(x1, y1, x2, y2, scale=1.0):
    """
    Computes Euclidean distance between two points (or arrays of points).
    Args:
        x1, y1, x2, y2 (array-like): Coordinates.
        scale (float): Scaling factor (e.g., meters/pixel).
    Returns:
        np.ndarray: Distance(s).
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * scale

def smooth_series(series, window=7, poly=2):
    """
    Applies Savitzky-Golay smoothing to a 1D series.
    Args:
        series (array-like): Input data.
        window (int): Window length (must be odd).
        poly (int): Polynomial order.
    Returns:
        np.ndarray: Smoothed series.
    """
    if len(series) < window:
        return series
    return savgol_filter(series, window, poly)

def bat_speed(df, frame_rate, hand='right', mapping=None, scale=1.0, smooth=False):
    """
    Computes bat (wrist) speed over time.
    Returns:
        np.ndarray: Speed per frame (length N-1).
    """
    wrist_x = df[mapping[f'{hand}_wrist_x']].values
    wrist_y = df[mapping[f'{hand}_wrist_y']].values
    if smooth:
        wrist_x = smooth_series(wrist_x)
        wrist_y = smooth_series(wrist_y)
    dt = 1.0 / frame_rate
    dx = np.diff(wrist_x)
    dy = np.diff(wrist_y)
    speed = np.sqrt(dx ** 2 + dy ** 2) * scale / dt
    return speed

def bat_path_arc(df, hand='right', mapping=None, scale=1.0, smooth=False):
    """
    Computes total arc length of the bat (wrist) path.
    Returns:
        float: Arc length.
    """
    wrist_x = df[mapping[f'{hand}_wrist_x']].values
    wrist_y = df[mapping[f'{hand}_wrist_y']].values
    if smooth:
        wrist_x = smooth_series(wrist_x)
        wrist_y = smooth_series(wrist_y)
    dx = np.diff(wrist_x)
    dy = np.diff(wrist_y)
    arc_length = np.sum(np.sqrt(dx ** 2 + dy ** 2)) * scale
    return arc_length

def swing_tempo(df, frame_rate, hand='right', mapping=None, smooth=False):
    """
    Estimates swing phase timings based on wrist and hip speed.
    Returns:
        dict: Frame indices and times for load, launch, contact, follow-through.
    """
    wrist_x = df[mapping[f'{hand}_wrist_x']].values
    wrist_y = df[mapping[f'{hand}_wrist_y']].values
    if smooth:
        wrist_x = smooth_series(wrist_x)
        wrist_y = smooth_series(wrist_y)
    dt = 1.0 / frame_rate
    speed = np.sqrt(np.diff(wrist_x) ** 2 + np.diff(wrist_y) ** 2) / dt

    hip_x = (df[mapping['left_hip_x']].values + df[mapping['right_hip_x']].values) / 2
    hip_y = (df[mapping['left_hip_y']].values + df[mapping['right_hip_y']].values) / 2
    if smooth:
        hip_x = smooth_series(hip_x)
        hip_y = smooth_series(hip_y)
    hip_speed = np.sqrt(np.diff(hip_x) ** 2 + np.diff(hip_y) ** 2) / dt

    load_frame = 0
    launch_frame = np.argmax(hip_speed > np.percentile(hip_speed, 70))
    contact_frame = np.argmax(speed)
    post_contact = speed[contact_frame:]
    below_thresh = np.where(post_contact < 0.3 * np.max(speed))[0]
    follow_through_frame = contact_frame + below_thresh[0] if len(below_thresh) > 0 else len(speed) - 1

    return {
        'load_frame': load_frame,
        'launch_frame': launch_frame,
        'contact_frame': contact_frame,
        'follow_through_frame': follow_through_frame,
        'load_time': load_frame * dt,
        'launch_time': launch_frame * dt,
        'contact_time': contact_frame * dt,
        'follow_through_time': follow_through_frame * dt,
        'tempo_phases_sec': {
            'load_to_launch': (launch_frame - load_frame) * dt,
            'launch_to_contact': (contact_frame - launch_frame) * dt,
            'contact_to_follow': (follow_through_frame - contact_frame) * dt,
        }
    }

def hip_rotation_angle(df, mapping=None, smooth=False):
    """
    Computes hip rotation angle (degrees) over time.
    Returns:
        np.ndarray: Angle per frame.
    """
    x_rh, y_rh = df[mapping['right_hip_x']].values, df[mapping['right_hip_y']].values
    x_lh, y_lh = df[mapping['left_hip_x']].values, df[mapping['left_hip_y']].values
    if smooth:
        x_rh = smooth_series(x_rh)
        y_rh = smooth_series(y_rh)
        x_lh = smooth_series(x_lh)
        y_lh = smooth_series(y_lh)
    angles = np.degrees(np.arctan2(y_rh - y_lh, x_rh - x_lh))
    return angles

def shoulder_rotation_angle(df, mapping=None, smooth=False):
    """
    Computes shoulder rotation angle (degrees) over time.
    Returns:
        np.ndarray: Angle per frame.
    """
    x_rs, y_rs = df[mapping['right_shoulder_x']].values, df[mapping['right_shoulder_y']].values
    x_ls, y_ls = df[mapping['left_shoulder_x']].values, df[mapping['left_shoulder_y']].values
    if smooth:
        x_rs = smooth_series(x_rs)
        y_rs = smooth_series(y_rs)
        x_ls = smooth_series(x_ls)
        y_ls = smooth_series(y_ls)
    angles = np.degrees(np.arctan2(y_rs - y_ls, x_rs - x_ls))
    return angles

def x_factor(df, mapping=None, smooth=False):
    """
    Computes X-factor (shoulder rotation minus hip rotation).
    Returns:
        np.ndarray: X-factor per frame.
    """
    return shoulder_rotation_angle(df, mapping, smooth) - hip_rotation_angle(df, mapping, smooth)

def stride_length(df, mapping=None, scale=1.0, smooth=False):
    """
    Computes stride length (distance between ankles).
    Returns:
        np.ndarray: Stride length per frame.
    """
    x_la, y_la = df[mapping['left_ankle_x']].values, df[mapping['left_ankle_y']].values
    x_ra, y_ra = df[mapping['right_ankle_x']].values, df[mapping['right_ankle_y']].values
    if smooth:
        x_la = smooth_series(x_la)
        y_la = smooth_series(y_la)
        x_ra = smooth_series(x_ra)
        y_ra = smooth_series(y_ra)
    return euclidean_distance(x_la, y_la, x_ra, y_ra, scale=scale)

def weight_transfer(df, mapping=None, smooth=False):
    """
    Computes center of mass (COM) X position as average of hip Xs.
    Returns:
        np.ndarray: COM X per frame.
    """
    x_lh = df[mapping['left_hip_x']].values
    x_rh = df[mapping['right_hip_x']].values
    if smooth:
        x_lh = smooth_series(x_lh)
        x_rh = smooth_series(x_rh)
    return (x_lh + x_rh) / 2

def shoulder_tilt(df, mapping=None, smooth=False):
    """
    Computes shoulder tilt (left_y - right_y).
    Returns:
        np.ndarray: Tilt per frame.
    """
    y_ls = df[mapping['left_shoulder_y']].values
    y_rs = df[mapping['right_shoulder_y']].values
    if smooth:
        y_ls = smooth_series(y_ls)
        y_rs = smooth_series(y_rs)
    return y_ls - y_rs

def knee_flexion(df, side='right', mapping=None, smooth=False):
    """
    Computes knee flexion angle (degrees) for given side.
    Returns:
        np.ndarray: Angle per frame.
    """
    hx, hy = df[mapping[f'{side}_hip_x']].values, df[mapping[f'{side}_hip_y']].values
    kx, ky = df[mapping[f'{side}_knee_x']].values, df[mapping[f'{side}_knee_y']].values
    ax, ay = df[mapping[f'{side}_ankle_x']].values, df[mapping[f'{side}_ankle_y']].values
    if smooth:
        hx = smooth_series(hx)
        hy = smooth_series(hy)
        kx = smooth_series(kx)
        ky = smooth_series(ky)
        ax = smooth_series(ax)
        ay = smooth_series(ay)
    a = np.stack([hx - kx, hy - ky], axis=1)
    b = np.stack([ax - kx, ay - ky], axis=1)
    dot = np.sum(a * b, axis=1)
    norm_a = np.clip(np.linalg.norm(a, axis=1), 1e-6, None)
    norm_b = np.clip(np.linalg.norm(b, axis=1), 1e-6, None)
    cos_theta = dot / (norm_a * norm_b)
    angles = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return angles

def spine_angle(df, mapping=None, smooth=False):
    """
    Computes spine angle (hip center to nose, degrees).
    Returns:
        np.ndarray: Angle per frame.
    """
    x_hip = (df[mapping['left_hip_x']].values + df[mapping['right_hip_x']].values) / 2
    y_hip = (df[mapping['left_hip_y']].values + df[mapping['right_hip_y']].values) / 2
    x_nose = df[mapping['nose_x']].values
    y_nose = df[mapping['nose_y']].values
    if smooth:
        x_hip = smooth_series(x_hip)
        y_hip = smooth_series(y_hip)
        x_nose = smooth_series(x_nose)
        y_nose = smooth_series(y_nose)
    angles = np.degrees(np.arctan2(y_hip - y_nose, x_hip - x_nose))
    return angles

def stance_width(df, mapping=None, scale=1.0, smooth=False):
    """
    Computes stance width (distance between foot indices).
    Returns:
        np.ndarray: Stance width per frame.
    """
    x_lf, y_lf = df[mapping['left_foot_index_x']].values, df[mapping['left_foot_index_y']].values
    x_rf, y_rf = df[mapping['right_foot_index_x']].values, df[mapping['right_foot_index_y']].values
    if smooth:
        x_lf = smooth_series(x_lf)
        y_lf = smooth_series(y_lf)
        x_rf = smooth_series(x_rf)
        y_rf = smooth_series(y_rf)
    return euclidean_distance(x_lf, y_lf, x_rf, y_rf, scale=scale)

def hand_path_deviation(df, hand='right', mapping=None, smooth=False):
    """
    Computes mean standard deviation of wrist path (as a measure of deviation).
    Returns:
        float: Mean std deviation.
    """
    wrist_x = df[mapping[f'{hand}_wrist_x']].values
    wrist_y = df[mapping[f'{hand}_wrist_y']].values
    if smooth:
        wrist_x = smooth_series(wrist_x)
        wrist_y = smooth_series(wrist_y)
    return np.std(np.stack([wrist_x, wrist_y], axis=1), axis=0).mean()

def pelvic_drop_rise(df, mapping=None, smooth=False):
    """
    Computes frame-to-frame difference in average hip Y (pelvic drop/rise).
    Returns:
        np.ndarray: Difference per frame.
    """
    y_hip = (df[mapping['left_hip_y']].values + df[mapping['right_hip_y']].values) / 2
    if smooth:
        y_hip = smooth_series(y_hip)
    return np.diff(y_hip)

def follow_through_smoothness(df, frame_rate, hand='right', mapping=None, smooth=False):
    """
    Computes jerk (third derivative) of wrist path as a measure of follow-through smoothness.
    Returns:
        np.ndarray: Jerk per frame (length N-3).
    """
    wrist_x = df[mapping[f'{hand}_wrist_x']].values
    wrist_y = df[mapping[f'{hand}_wrist_y']].values
    if smooth:
        wrist_x = smooth_series(wrist_x)
        wrist_y = smooth_series(wrist_y)
    dt = 1.0 / frame_rate
    if len(wrist_x) < 4:
        print("[JERK WARNING] Not enough frames for jerk calculation.")
        return np.zeros(0)
    jerk_x = np.diff(wrist_x, n=3) / (dt ** 3)
    jerk_y = np.diff(wrist_y, n=3) / (dt ** 3)
    jerk = np.sqrt(jerk_x ** 2 + jerk_y ** 2)
    return jerk

def joint_angle(df, a, b, c, mapping=None, smooth=False):
    """
    Computes angle at joint 'b' formed by points a-b-c.
    Args:
        a, b, c (str): Keypoint names (e.g., 'right_hip', 'right_knee', 'right_ankle').
    Returns:
        np.ndarray: Angle per frame.
    """
    ax, ay = df[mapping[f'{a}_x']].values, df[mapping[f'{a}_y']].values
    bx, by = df[mapping[f'{b}_x']].values, df[mapping[f'{b}_y']].values
    cx, cy = df[mapping[f'{c}_x']].values, df[mapping[f'{c}_y']].values
    if smooth:
        ax = smooth_series(ax)
        ay = smooth_series(ay)
        bx = smooth_series(bx)
        by = smooth_series(by)
        cx = smooth_series(cx)
        cy = smooth_series(cy)
    ab = np.stack([ax - bx, ay - by], axis=1)
    cb = np.stack([cx - bx, cy - by], axis=1)
    dot = np.sum(ab * cb, axis=1)
    norm_ab = np.clip(np.linalg.norm(ab, axis=1), 1e-6, None)
    norm_cb = np.clip(np.linalg.norm(cb, axis=1), 1e-6, None)
    cos_theta = dot / (norm_ab * norm_cb)
    angles = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return angles

def angular_velocity(angle_series, frame_rate, smooth=False):
    """
    Computes angular velocity (degrees/sec) from angle series.
    Returns:
        np.ndarray: Angular velocity per frame (length N-1).
    """
    if smooth:
        angle_series = smooth_series(angle_series)
    dt = 1.0 / frame_rate
    return np.diff(angle_series) / dt

def symmetry_metric(left_series, right_series):
    """
    Computes absolute difference between left and right series (symmetry metric).
    Returns:
        np.ndarray: Absolute difference per frame.
    """
    if len(left_series) != len(right_series):
        print("[SYMMETRY WARNING] Series length mismatch.")
        min_len = min(len(left_series), len(right_series))
        left_series = left_series[:min_len]
        right_series = right_series[:min_len]
    return np.abs(left_series - right_series)

def sequence_dynamics(df, frame_rate, joints=['hip', 'shoulder', 'elbow', 'wrist'], side='right', mapping=None, smooth=False):
    """
    Estimates the frame at which each joint exceeds 70th percentile speed (activation sequence).
    Returns:
        dict: {joint: frame_index}
    """
    dt = 1.0 / frame_rate
    activation = {}
    for joint in joints:
        key_x = f'{side}_{joint}_x'
        key_y = f'{side}_{joint}_y'
        if key_x not in mapping or key_y not in mapping:
            print(f"[SEQUENCE WARNING] Skipping joint '{joint}' (missing in mapping)")
            continue
        x = df[mapping[key_x]].values
        y = df[mapping[key_y]].values
        if smooth:
            x = smooth_series(x)
            y = smooth_series(y)
        speed = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / dt
        threshold = np.percentile(speed, 70)
        above = np.where(speed > threshold)[0]
        activation[joint] = above[0] if len(above) > 0 else None
    sorted_activation = sorted(activation.items(), key=lambda x: (x[1] if x[1] is not None else 1e9))
    return {k: v for k, v in sorted_activation}

# Optional: Add a global scaling factor (meters/pixel) if you have calibration info.
# Example usage: bat_speed(..., scale=0.01) for 1cm/pixel.