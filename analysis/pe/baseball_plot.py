import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baseball_analysis import (
    check_and_map_columns, bat_speed, bat_path_arc, swing_tempo, hip_rotation_angle, shoulder_rotation_angle,
    x_factor, stride_length, weight_transfer, shoulder_tilt, knee_flexion, spine_angle,
    stance_width, hand_path_deviation, pelvic_drop_rise, follow_through_smoothness,
    joint_angle, angular_velocity, symmetry_metric, sequence_dynamics
)

def plot_and_save(fig, out_dir, name):
    """Save a matplotlib figure to the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)

def main(csv_path, frame_rate=30, dominant_hand="right", scale=1.0, smooth=False, out_dir="plots"):
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    mapping = check_and_map_columns(df, hand=dominant_hand)

    # --- 1. Bat Speed Over Time ---
    bat_speed_series = bat_speed(df, frame_rate, hand=dominant_hand, mapping=mapping, scale=scale, smooth=smooth)
    swing_phases = swing_tempo(df, frame_rate, hand=dominant_hand, mapping=mapping, smooth=smooth)
    contact_frame = swing_phases['contact_frame']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bat_speed_series, label="Bat Speed")
    ax.axvline(contact_frame, color='r', linestyle='--', label='Contact')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Bat Speed (pixels/s)")
    ax.set_title("Bat Speed Over Time")
    ax.legend()
    plot_and_save(fig, out_dir, "bat_speed_over_time")

    # --- 2. Bat Path Trajectory (2D) ---
    wrist_x = df[mapping[f'{dominant_hand}_wrist_x']].values
    wrist_y = df[mapping[f'{dominant_hand}_wrist_y']].values
    fig, ax = plt.subplots(figsize=(6, 6))
    points = ax.scatter(wrist_x, wrist_y, c=np.arange(len(wrist_x)), cmap='viridis', s=10, label='Wrist Path')
    ax.scatter(wrist_x[contact_frame], wrist_y[contact_frame], color='red', s=40, label='Contact')
    ax.set_xlabel("Wrist X")
    ax.set_ylabel("Wrist Y")
    ax.set_title("Bat Path Trajectory (2D)")
    fig.colorbar(points, ax=ax, label='Frame')
    ax.legend()
    plot_and_save(fig, out_dir, "bat_path_trajectory")

    # --- 3. Sequence Dynamics Timeline ---
    seq_dyn = sequence_dynamics(df, frame_rate, side=dominant_hand, mapping=mapping, smooth=smooth)
    fig, ax = plt.subplots(figsize=(7, 3))
    joints = list(seq_dyn.keys())
    frames = [seq_dyn[j] for j in joints]
    ax.hlines(y=joints, xmin=0, xmax=frames, color='b')
    ax.plot(frames, joints, 'ro')
    ax.set_xlabel("Frame of Activation")
    ax.set_title("Sequence Dynamics Timeline")
    plot_and_save(fig, out_dir, "sequence_dynamics_timeline")

    # --- 4. Hip & Shoulder Rotation Angles ---
    hip_angle = hip_rotation_angle(df, mapping=mapping, smooth=smooth)
    shoulder_angle = shoulder_rotation_angle(df, mapping=mapping, smooth=smooth)
    xfactor = x_factor(df, mapping=mapping, smooth=smooth)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hip_angle, label="Hip Angle")
    ax.plot(shoulder_angle, label="Shoulder Angle")
    ax.plot(xfactor, label="X-Factor")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Hip & Shoulder Rotation Angles")
    ax.legend()
    plot_and_save(fig, out_dir, "hip_shoulder_xfactor_angles")

    # --- 5. Spine Angle & Shoulder Tilt ---
    spine = spine_angle(df, mapping=mapping, smooth=smooth)
    tilt = shoulder_tilt(df, mapping=mapping, smooth=smooth)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spine, label="Spine Angle")
    ax.plot(tilt, label="Shoulder Tilt")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (deg) / Offset (pixels)")
    ax.set_title("Spine Angle & Shoulder Tilt")
    ax.legend()
    plot_and_save(fig, out_dir, "spine_angle_shoulder_tilt")

    # --- 6. Weight Transfer (COM X) ---
    com_x = weight_transfer(df, mapping=mapping, smooth=smooth)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(com_x, label="COM X")
    for phase, frame in swing_phases.items():
        if phase.endswith('_frame'):
            ax.axvline(frame, linestyle='--', alpha=0.3, label=phase)
    ax.set_xlabel("Frame")
    ax.set_ylabel("COM X (pixels)")
    ax.set_title("Weight Transfer (COM X Position)")
    ax.legend()
    plot_and_save(fig, out_dir, "weight_transfer_com_x")

    # --- 7. Left vs Right Symmetry (Knee Flexion) ---
    left_knee = knee_flexion(df, side='left', mapping=mapping, smooth=smooth)
    right_knee = knee_flexion(df, side='right', mapping=mapping, smooth=smooth)
    sym = symmetry_metric(left_knee, right_knee)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(left_knee, label="Left Knee Flexion")
    ax.plot(right_knee, label="Right Knee Flexion")
    ax.plot(sym, label="Symmetry (abs diff)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Knee Angle (deg)")
    ax.set_title("Left vs Right Knee Flexion & Symmetry")
    ax.legend()
    plot_and_save(fig, out_dir, "knee_flexion_symmetry")

    # --- 8. Knee Flexion Angle Over Time (Right) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(right_knee, label="Right Knee Flexion")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Knee Angle (deg)")
    ax.set_title("Right Knee Flexion Angle Over Time")
    ax.legend()
    plot_and_save(fig, out_dir, "right_knee_flexion_over_time")

    # --- 9. Follow-through Smoothness (Jerk) ---
    jerk = follow_through_smoothness(df, frame_rate, hand=dominant_hand, mapping=mapping, smooth=smooth)
    fig, ax = plt.subplots(figsize=(8, 4))
    jerk_x = np.arange(len(jerk)) + 3  # jerk is 3 frames shorter
    ax.plot(jerk_x, jerk, label="Jerk (3rd Derivative)")
    ax.axhline(np.median(jerk), color='g', linestyle='--', label='Median Jerk')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Jerk (pixels/s³)")
    ax.set_title("Follow-through Smoothness (Jerk)")
    ax.legend()
    plot_and_save(fig, out_dir, "follow_through_smoothness_jerk")

    # --- 10. Heatmap of All Metrics Per Swing ---
    metrics = {
        "Bat Speed (max)": np.max(bat_speed_series),
        "Bat Path Arc": bat_path_arc(df, hand=dominant_hand, mapping=mapping, scale=scale, smooth=smooth),
        "Hip Angle (max)": np.max(hip_angle),
        "Shoulder Angle (max)": np.max(shoulder_angle),
        "X-Factor (max)": np.max(xfactor),
        "Stride Length (mean)": np.mean(stride_length(df, mapping=mapping, scale=scale, smooth=smooth)),
        "COM X (range)": np.ptp(com_x),
        "Knee Flexion L (min)": np.min(left_knee),
        "Knee Flexion R (min)": np.min(right_knee),
        "Symmetry (mean)": np.mean(sym),
        "Spine Angle (mean)": np.mean(spine),
        "Shoulder Tilt (mean)": np.mean(tilt),
        "Jerk (median)": np.median(jerk) if len(jerk) > 0 else 0,
    }
    metric_names = list(metrics.keys())
    metric_vals = np.array(list(metrics.values())).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(5, len(metrics) * 0.5 + 1))
    sns.heatmap(metric_vals, annot=True, fmt=".2f", yticklabels=metric_names, cmap="YlGnBu", ax=ax, cbar=False)
    ax.set_title("Biomechanical Metrics Heatmap")
    plot_and_save(fig, out_dir, "metrics_heatmap")

    print(f"All plots saved to: {out_dir}")

if __name__ == "__main__":
    # Import parameters from baseball_test.py if available
    try:
        import baseball_test
        csv_path = baseball_test.csv_path
        frame_rate = baseball_test.frame_rate
        dominant_hand = baseball_test.dominant_hand
        scale = getattr(baseball_test, "scale", 1.0)
        smooth = getattr(baseball_test, "smooth", False)
        print(f"[INFO] Using parameters from baseball_test.py: {csv_path}, {frame_rate}, {dominant_hand}, {scale}, {smooth}")
    except ImportError:
        # Fallback: use argparse for CLI
        import argparse
        parser = argparse.ArgumentParser(description="Visualize baseball swing biomechanics from cleaned keypoints CSV.")
        parser.add_argument("csv_path", type=str, help="Path to cleaned keypoints CSV")
        parser.add_argument("--frame_rate", type=int, default=30, help="Video frame rate (Hz)")
        parser.add_argument("--hand", type=str, default="right", help="Dominant hand: 'right' or 'left'")
        parser.add_argument("--scale", type=float, default=1.0, help="Meters per pixel (for real-world units)")
        parser.add_argument("--smooth", action="store_true", help="Apply Savitzky-Golay smoothing to signals")
        args = parser.parse_args()
        csv_path = args.csv_path
        frame_rate = args.frame_rate
        dominant_hand = args.hand
        scale = args.scale
        smooth = args.smooth

    main(csv_path, frame_rate=frame_rate, dominant_hand=dominant_hand, scale=scale, smooth=smooth)