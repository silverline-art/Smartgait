"""
baseball_test.py

Unified script: Extracts and prints baseball swing biomechanical features,
then generates and saves plots using the same parameters.

Usage:
    python baseball_test.py
"""

import pandas as pd
from baseball_analysis import (
    check_and_map_columns, bat_speed, bat_path_arc, swing_tempo, hip_rotation_angle, shoulder_rotation_angle,
    x_factor, stride_length, weight_transfer, shoulder_tilt, knee_flexion, spine_angle,
    stance_width, hand_path_deviation, pelvic_drop_rise, follow_through_smoothness,
    joint_angle, angular_velocity, symmetry_metric, sequence_dynamics
)

# ---- PARAMETERS (used by both this script and visualization) ----
csv_path = "/home/shivam/Desktop/GitHub prep/SmartGait/test/keypoints_cleaned.csv"
frame_rate = 120  # Set your video frame rate (Hz)
dominant_hand = 'right'  # or 'left'
scale = 1.0  # meters per pixel (set to 1.0 for pixels)
smooth = True  # Set True to enable Savitzky-Golay smoothing

def print_feature(name, value):
    """Pretty-print feature name and value, handling arrays and dicts."""
    print(f"\n--- {name} ---")
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"{k}: {v}")
    elif hasattr(value, "__len__") and not isinstance(value, str):
        # Print summary for arrays/series
        try:
            import numpy as np
            arr = np.array(value)
            print(f"Length: {len(arr)} | Min: {arr.min():.3f} | Max: {arr.max():.3f} | Mean: {arr.mean():.3f}")
            print(f"First 5: {arr[:5]}")
        except Exception:
            print(value)
    else:
        print(value)

def main():
    print(f"Loading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return

    print("Columns in DataFrame:", df.columns.tolist())

    try:
        mapping = check_and_map_columns(df, hand=dominant_hand)
    except KeyError as e:
        print("ERROR:", e)
        return

    # Feature extraction and printing
    print_feature("Bat Speed", bat_speed(df, frame_rate, hand=dominant_hand, mapping=mapping, scale=scale, smooth=smooth))
    print_feature("Bat Path Arc Length", bat_path_arc(df, hand=dominant_hand, mapping=mapping, scale=scale, smooth=smooth))
    print_feature("Swing Tempo", swing_tempo(df, frame_rate, hand=dominant_hand, mapping=mapping, smooth=smooth))
    print_feature("Hip Rotation Angle", hip_rotation_angle(df, mapping=mapping, smooth=smooth))
    print_feature("Shoulder Rotation Angle", shoulder_rotation_angle(df, mapping=mapping, smooth=smooth))
    print_feature("X-Factor", x_factor(df, mapping=mapping, smooth=smooth))
    print_feature("Stride Length", stride_length(df, mapping=mapping, scale=scale, smooth=smooth))
    print_feature("Weight Transfer (COM_x)", weight_transfer(df, mapping=mapping, smooth=smooth))
    print_feature("Shoulder Tilt", shoulder_tilt(df, mapping=mapping, smooth=smooth))
    print_feature("Right Knee Flexion", knee_flexion(df, side='right', mapping=mapping, smooth=smooth))
    print_feature("Left Knee Flexion", knee_flexion(df, side='left', mapping=mapping, smooth=smooth))
    print_feature("Spine Angle", spine_angle(df, mapping=mapping, smooth=smooth))
    print_feature("Stance Width", stance_width(df, mapping=mapping, scale=scale, smooth=smooth))
    print_feature("Hand Path Deviation", hand_path_deviation(df, hand=dominant_hand, mapping=mapping, smooth=smooth))
    print_feature("Pelvic Drop/Rise", pelvic_drop_rise(df, mapping=mapping, smooth=smooth))
    print_feature("Follow-through Smoothness", follow_through_smoothness(df, frame_rate, hand=dominant_hand, mapping=mapping, smooth=smooth))
    print_feature("Right Knee Angle (joint_angle)", joint_angle(df, f'{dominant_hand}_hip', f'{dominant_hand}_knee', f'{dominant_hand}_ankle', mapping=mapping, smooth=smooth))
    print_feature("Angular Velocity (hip)", angular_velocity(hip_rotation_angle(df, mapping=mapping, smooth=smooth), frame_rate, smooth=smooth))
    print_feature("Symmetry (knee flexion)", symmetry_metric(
        knee_flexion(df, 'left', mapping=mapping, smooth=smooth),
        knee_flexion(df, 'right', mapping=mapping, smooth=smooth)
    ))
    print_feature("Sequence Dynamics", sequence_dynamics(df, frame_rate, side=dominant_hand, mapping=mapping, smooth=smooth))

    # --- Call plotting routine after feature extraction ---
    print("\n[INFO] Generating and saving plots...")
    try:
        from baseball_plot import main as plot_main
        plot_main(
            csv_path=csv_path,
            frame_rate=frame_rate,
            dominant_hand=dominant_hand,
            scale=scale,
            smooth=smooth,
            out_dir="plots"
        )
    except ImportError as e:
        print(f"[ERROR] Could not import plotting module: {e}")
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")

if __name__ == "__main__":
    main()