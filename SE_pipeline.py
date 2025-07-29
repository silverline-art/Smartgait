import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import file_path, output_path
from utils.data_processing import read_input_csv, detect_time_column
from utils.visualization_utils import create_output_directory

# SE-specific imports
from analysis.si.step_detection import analyze_gait, find_walking_segment
from analysis.si.original_summary import (
    print_gait_report,
    print_temporal_parameter_table,
    print_gait_phase_percentages
)
from visualization.visse_gait import plot_phase_percentages
from analysis.si.initial_pressure import compute_and_inject_initial_pressure
from analysis.si.time_parameters import inject_time_parameters

# --- Robust region mapping for your data columns ---
class SensorRegionConstants:
    LEFT_HINDFOOT = ['L_value4']
    LEFT_MIDFOOT = ['L_value1', 'L_value3']
    LEFT_FOREFOOT = ['L_value2']
    RIGHT_HINDFOOT = ['R_value4']
    RIGHT_MIDFOOT = ['R_value1', 'R_value3']
    RIGHT_FOREFOOT = ['R_value2']

def get_sensor_region_constants():
    return SensorRegionConstants()

def _ensure_time_in_seconds(time_array):
    if np.nanmax(time_array) > 1000:
        return time_array / 1000.0
    return time_array

def filter_existing_cols(df, col_list):
    """Return only columns from col_list that exist in df."""
    return [c for c in col_list if c in df.columns]

def plot_foot_timeseries_subplots_with_steps(
        df: pd.DataFrame,
        time_col: str,
        left_steps=None,
        right_steps=None,
        left_starts=None,
        left_falls=None,
        right_starts=None,
        right_falls=None,
        filename="foot_timeseries_steps.png"
):
    """
    Plots left and right foot sensor time series from a DataFrame, with step markers
    and shaded stance phase regions. Always saves the figure to output_path/plot/filename.
    """
    constants = get_sensor_region_constants()
    time = _ensure_time_in_seconds(df[time_col].values)

    left_region_cols_list = [
        filter_existing_cols(df, constants.LEFT_HINDFOOT),
        filter_existing_cols(df, constants.LEFT_MIDFOOT),
        filter_existing_cols(df, constants.LEFT_FOREFOOT)
    ]
    left_labels = ["Left Hindfoot", "Left Midfoot", "Left Forefoot"]
    right_region_cols_list = [
        filter_existing_cols(df, constants.RIGHT_HINDFOOT),
        filter_existing_cols(df, constants.RIGHT_MIDFOOT),
        filter_existing_cols(df, constants.RIGHT_FOREFOOT)
    ]
    right_labels = ["Right Hindfoot", "Right Midfoot", "Right Forefoot"]

    # Debug: print which columns are being used for each region
    for label, cols in zip(left_labels, left_region_cols_list):
        print(f"[DEBUG] {label} columns: {cols}")
    for label, cols in zip(right_labels, right_region_cols_list):
        print(f"[DEBUG] {label} columns: {cols}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot left foot
    for region_cols, label in zip(left_region_cols_list, left_labels):
        if region_cols:
            region_signal = df[region_cols].mean(axis=1)
            ax0.plot(time, region_signal, label=label, alpha=0.9)

    # Shade the stance phase for the left foot
    if left_starts is not None and left_falls is not None:
        for start, end in zip(left_starts, left_falls):
            if start < len(time) and end < len(time):
                ax0.axvspan(time[start], time[end], color='tab:blue', alpha=0.15, lw=0)

    if left_steps is not None:
        valid_steps = left_steps[left_steps < len(time)]
        step_times = time[valid_steps]
        ax0.vlines(step_times, ymin=0, ymax=ax0.get_ylim()[1], color='tab:blue', linestyle=':', alpha=0.7,
                   label='Step Peaks')

    ax0.set_title("Left Foot Sensor Time Series with Step Events")
    ax0.set_ylabel("Average Pressure")
    ax0.legend(loc='upper right')
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot right foot
    for region_cols, label in zip(right_region_cols_list, right_labels):
        if region_cols:
            region_signal = df[region_cols].mean(axis=1)
            ax1.plot(time, region_signal, label=label, alpha=0.9)

    # Shade the stance phase for the right foot
    if right_starts is not None and right_falls is not None:
        for start, end in zip(right_starts, right_falls):
            if start < len(time) and end < len(time):
                ax1.axvspan(time[start], time[end], color='tab:red', alpha=0.15, lw=0)

    if right_steps is not None:
        valid_steps = right_steps[right_steps < len(time)]
        step_times = time[valid_steps]
        ax1.vlines(step_times, ymin=0, ymax=ax1.get_ylim()[1], color='tab:red', linestyle=':', alpha=0.7,
                   label='Step Peaks')

    ax1.set_title("Right Foot Sensor Time Series with Step Events")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Average Pressure")
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Always save the figure to output_path/plot/filename
    output_dir = os.path.join(output_path, "plot")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Foot timeseries plot saved to: {save_path}")

def main():
    """Main Smart Insole Analysis pipeline execution."""
    debug = False
    
    try:
        print("[INFO] Starting Smart Insole Analysis Pipeline...")
        
        # Create output directory
        create_output_directory(output_path)
        
        # Step 1: Load the CSV data
        print(f"[INFO] Step 1: Loading CSV data from: {file_path}")
        full_df = read_input_csv(file_path)
        print(f"[INFO] Loaded {len(full_df)} rows with {len(full_df.columns)} columns")
        
        # Auto-detect time column
        time_col = detect_time_column(full_df)
        if time_col is None:
            print("[ERROR] Could not find a time column in the CSV.")
            print("[ERROR] Available columns:", list(full_df.columns))
            sys.exit(1)
        print(f"[INFO] Using time column: {time_col}")
        
        # Step 2: Detect walking segment
        print("[INFO] Step 2: Detecting walking segment...")
        start_idx, end_idx = find_walking_segment(full_df, time_col)
        walking_df = full_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        if walking_df.empty or (end_idx - start_idx) < 200:
            print("[ERROR] No significant walking segment detected. Aborting analysis.")
            print(f"[ERROR] Detected segment length: {end_idx - start_idx} samples")
            sys.exit(1)
        
        start_time = full_df[time_col].iloc[start_idx] / 1000.0
        end_time = full_df[time_col].iloc[end_idx] / 1000.0
        print(f"[INFO] Walking segment: {start_time:.2f}s to {end_time:.2f}s ({end_idx-start_idx} samples)")
        
        # Step 3: Gait analysis
        print("[INFO] Step 3: Running gait analysis...")
        result = analyze_gait(walking_df, time_col, debug=debug)
        
        if result.get('steps', 0) == 0:
            print("[ERROR] No steps detected. Creating debug visualization...")
            plot_foot_timeseries_subplots_with_steps(walking_df, time_col)
            print("[INFO] Debug plot saved. Check output directory.")
            sys.exit(1)
        
        print(f"[INFO] Detected {result['steps']} total steps")
        
        # Step 4: Pressure analysis
        print("[INFO] Step 4: Computing pressure distribution analysis...")
        compute_and_inject_initial_pressure(result, walking_df, debug=debug)
        
        # Step 5: Temporal parameters
        print("[INFO] Step 5: Computing temporal gait parameters...")
        constants = get_sensor_region_constants()
        
        # Filter existing columns for each foot region
        region_mapping = {
            'left_hindfoot': filter_existing_cols(walking_df, constants.LEFT_HINDFOOT),
            'left_midfoot': filter_existing_cols(walking_df, constants.LEFT_MIDFOOT),
            'left_forefoot': filter_existing_cols(walking_df, constants.LEFT_FOREFOOT),
            'right_hindfoot': filter_existing_cols(walking_df, constants.RIGHT_HINDFOOT),
            'right_midfoot': filter_existing_cols(walking_df, constants.RIGHT_MIDFOOT),
            'right_forefoot': filter_existing_cols(walking_df, constants.RIGHT_FOREFOOT)
        }
        
        inject_time_parameters(
            result, walking_df, debug=debug, time_col=time_col,
            left_hindfoot_cols=region_mapping['left_hindfoot'],
            left_midfoot_cols=region_mapping['left_midfoot'],
            left_forefoot_cols=region_mapping['left_forefoot'],
            right_hindfoot_cols=region_mapping['right_hindfoot'],
            right_midfoot_cols=region_mapping['right_midfoot'],
            right_forefoot_cols=region_mapping['right_forefoot']
        )
        
        # Step 6: Generate reports and visualizations
        print("[INFO] Step 6: Generating reports and visualizations...")
        
        # Main gait report
        report_str = print_gait_report(result, return_str=True)
        print(report_str)
        
        # Temporal parameters table
        time_array = walking_df[time_col].values
        print_temporal_parameter_table(result, time_array, debug=debug)
        
        # Gait phase percentages
        print_gait_phase_percentages(result)
        
        # Generate visualizations
        phase_plot_dir = os.path.join(output_path, "plot")
        plot_phase_percentages(result, phase_plot_dir)
        
        # Time series plot for verification
        plot_foot_timeseries_subplots_with_steps(
            walking_df, time_col,
            left_steps=result.get("left_step_indices"),
            right_steps=result.get("right_step_indices"),
            left_starts=result.get("left_start_indices"),
            left_falls=result.get("left_fall_indices"),
            right_starts=result.get("right_start_indices"),
            right_falls=result.get("right_fall_indices")
        )
        
        print("[SUCCESS] Smart Insole Analysis pipeline completed successfully!")
        print(f"[INFO] Results saved to: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Smart Insole Analysis pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()