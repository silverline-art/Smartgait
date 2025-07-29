import os
import pandas as pd
from analysis.pe.ROM import compute_range_of_motion_and_std
from analysis.pe import gait_detect

def save_summary(output_dir):
    """
    Loads cleaned keypoints, computes ROM and gait stats, and saves a plain text summary in output_dir.
    The summary matches the format of print_rom_stats and gait_detect.print_gait_stats.
    """
    cleaned_csv_path = os.path.join(output_dir, "pe", "keypoints_cleaned.csv")
    df = pd.read_csv(cleaned_csv_path)

    # ROM stats
    rom_stats = compute_range_of_motion_and_std(df)
    rom_lines = ["Range of Motion per joint:\n"]
    for joint, values in rom_stats.items():
        rom_lines.append(f"{joint}: {values['rom']:.2f} ± {values['std']:.2f} degrees\n")

    # Gait stats
    analyzer = gait_detect.GaitCycleAnalyzer(df)
    results = analyzer.analyze()
    stride_lengths = results['stride_lengths'] * 100  # cm
    step_lengths = results['step_lengths'] * 100      # cm
    stride_mean = pd.Series(stride_lengths).mean()
    stride_std = pd.Series(stride_lengths).std()
    step_mean = pd.Series(step_lengths).mean()
    step_std = pd.Series(step_lengths).std()
    mean_speed_cm = results['mean_speed'] * 100 if results['mean_speed'] is not None else None
    cadence = results['cadence']
    stance_mean = pd.Series(results['stance_durations']).mean()
    swing_mean = pd.Series(results['swing_durations']).mean()
    gait_cycles = len(results['gait_cycles'])

    # Foot strike counts (if possible)
    left_count = right_count = None
    if "left_foot_index_y" in df.columns and "right_foot_index_y" in df.columns:
        left_analyzer = gait_detect.GaitCycleAnalyzer(df, side="left")
        right_analyzer = gait_detect.GaitCycleAnalyzer(df, side="right")
        left_count = len(left_analyzer.detect_foot_strikes()[0])
        right_count = len(right_analyzer.detect_foot_strikes()[0])

    gait_lines = ["\nGait Statistics:\n"]
    if left_count is not None and right_count is not None:
        gait_lines.append(f"Left foot strike count: {left_count}\n")
        gait_lines.append(f"Right foot strike count: {right_count}\n")
    gait_lines.append(f"Gait Cycles: {gait_cycles}\n")
    gait_lines.append(f"Avg Gait Velocity: {mean_speed_cm:.3f} cm/s\n")
    gait_lines.append(f"Avg Stride Length: {stride_mean:.3f} ± {stride_std:.3f} cm\n")
    gait_lines.append(f"Avg Step Length: {step_mean:.3f} ± {step_std:.3f} cm\n")
    gait_lines.append(f"Cadence (steps/min): {cadence:.2f}\n")
    gait_lines.append(f"Mean Stance Duration (s): {stance_mean:.3f}\n")
    gait_lines.append(f"Mean Swing Duration (s): {swing_mean:.3f}\n")

    summary_path = os.path.join(output_dir, "gait_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(rom_lines)
        f.writelines(gait_lines)