import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
from constants.se import get_sensor_region_constants

def _ensure_time_in_seconds(time_array):
    if np.nanmax(time_array) > 1000:
        return time_array / 1000.0
    return time_array

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
    and shaded stance phase regions. Always saves the figure to output/plot/filename.

    Args:
        df: Input DataFrame.
        time_col: Name of the time column.
        left_steps, right_steps: Indices of detected steps for left/right foot.
        left_starts, left_falls, right_starts, right_falls: Indices for stance phase shading.
        filename: Name of the output PNG file.
    """
    constants = get_sensor_region_constants()
    time = _ensure_time_in_seconds(df[time_col].values)

    left_region_cols_list = [constants.LEFT_HINDFOOT, constants.LEFT_MIDFOOT, constants.LEFT_FOREFOOT]
    left_labels = ["Left Hindfoot", "Left Midfoot", "Left Forefoot"]
    right_region_cols_list = [constants.RIGHT_HINDFOOT, constants.RIGHT_MIDFOOT, constants.RIGHT_FOREFOOT]
    right_labels = ["Right Hindfoot", "Right Midfoot", "Right Forefoot"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot left foot
    for region_cols, label in zip(left_region_cols_list, left_labels):
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

    # Always save the figure to output/plot/filename
    output_dir = os.path.join("output", "plot")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Foot timeseries plot saved to: {save_path}")