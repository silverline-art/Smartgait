import os
import math
import numpy as np
import matplotlib.pyplot as plt
from config import output_path
from analysis.pe.ROM import compute_range_of_motion_and_std, compute_peak_angular_velocities
from utils.visualization_utils import create_output_directory, save_plot_with_dpi, get_foot_colors, plot_angle_range_of_motion

def plot_rom_from_dataframe(df, out_dir=None):
    """
    Computes ROM and peak angular velocities from the DataFrame and plots/saves the ROM comparison figure.

    Args:
        df (pd.DataFrame): DataFrame with joint data.
        out_dir (str, optional): Directory to save the plot. If None, uses output_path/plot.
    """
    rom_stats = compute_range_of_motion_and_std(df)
    # Filter out NaN values and replace with 0
    rom_dict = {k: v['rom'] if not np.isnan(v['rom']) else 0.0 for k, v in rom_stats.items()}

    # Compute peak angular velocities if function is available
    try:
        peak_angular_velocities = compute_peak_angular_velocities(df)
    except Exception:
        peak_angular_velocities = None

    if out_dir is None:
        out_dir = os.path.join(output_path, "plot")
    create_output_directory(out_dir)
    plot_rom_comparison_from_dict(rom_dict, peak_angular_velocities=peak_angular_velocities, out_dir=out_dir)

def plot_rom_comparison_from_dict(rom_dict, peak_angular_velocities=None, out_dir=None):
    """
    Plots ROM comparison and related figures in the style of gait_analysis.py.

    Args:
        rom_dict (dict): Keys like 'hip_left', 'hip_right', etc. and ROM values.
        peak_angular_velocities (dict, optional): Keys like 'hip_left', 'hip_right', etc. and peak angular velocity values.
        out_dir (str, optional): Directory to save the plot. If None, uses output_path/plot.
    """
    if out_dir is None:
        out_dir = os.path.join(output_path, "plot")
    create_output_directory(out_dir)

    joint_pairs = [
        ('Hip', 'hip_left', 'hip_right'),
        ('Knee', 'knee_left', 'knee_right'),
        ('Ankle', 'ankle_left', 'ankle_right'),
        ('Shoulder', 'shoulder_left', 'shoulder_right')
    ]
    labels = []
    left_values = []
    right_values = []
    for label, left_key, right_key in joint_pairs:
        if left_key in rom_dict and right_key in rom_dict:
            labels.append(label)
            left_values.append(rom_dict[left_key])
            right_values.append(rom_dict[right_key])
    if not labels:
        print("No valid ROM data to plot.")
        return

    fig = plt.figure(figsize=(20, 12))
    # 1. Bar plot for ROM comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, left_values, width, label='Left', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, right_values, width, label='Right', color='red', alpha=0.7)
    ax1.set_xlabel('Joint', fontsize=12)
    ax1.set_ylabel('ROM (degrees)', fontsize=12)
    ax1.set_title('Range of Motion Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}°', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}°', ha='center', va='bottom', fontsize=9)

    # 2. Symmetry index plot
    ax2 = plt.subplot(2, 3, 2)
    symmetry_values = []
    for left_val, right_val in zip(left_values, right_values):
        if (left_val + right_val) > 0:
            sym = abs(left_val - right_val) / ((left_val + right_val) / 2) * 100
            symmetry_values.append(sym)
        else:
            symmetry_values.append(0)
    bars3 = ax2.bar(labels, symmetry_values, color='purple', alpha=0.7)
    ax2.set_xlabel('Joint', fontsize=12)
    ax2.set_ylabel('Symmetry Index (%)', fontsize=12)
    ax2.set_title('Joint Symmetry Indices', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, symmetry_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. Peak angular velocities (if provided)
    ax3 = plt.subplot(2, 3, 3)
    if peak_angular_velocities:
        peak_left = []
        peak_right = []
        peak_labels = []
        for label, left_key, right_key in joint_pairs:
            if left_key in peak_angular_velocities and right_key in peak_angular_velocities:
                peak_labels.append(label)
                peak_left.append(peak_angular_velocities[left_key])
                peak_right.append(peak_angular_velocities[right_key])
        if peak_labels:
            x_peak = np.arange(len(peak_labels))
            ax3.bar(x_peak - width / 2, peak_left, width, label='Left', color='blue', alpha=0.7)
            ax3.bar(x_peak + width / 2, peak_right, width, label='Right', color='red', alpha=0.7)
            ax3.set_xlabel('Joint', fontsize=12)
            ax3.set_ylabel('Peak Angular Velocity (deg/s)', fontsize=12)
            ax3.set_title('Peak Angular Velocities', fontsize=14, fontweight='bold')
            ax3.set_xticks(x_peak)
            ax3.set_xticklabels(peak_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.axis('off')

    # 4. Radar chart for ROM
    if len(labels) >= 3:
        ax4 = plt.subplot(2, 3, 4, projection='polar')
        N = len(labels)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        left_plot = left_values + [left_values[0]]
        right_plot = right_values + [right_values[0]]
        ax4.plot(angles, left_plot, 'o-', linewidth=2, label='Left', color='blue')
        ax4.fill(angles, left_plot, alpha=0.25, color='blue')
        ax4.plot(angles, right_plot, 'o-', linewidth=2, label='Right', color='red')
        ax4.fill(angles, right_plot, alpha=0.25, color='red')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(labels)
        # Handle NaN values for axis limits
        valid_values = [v for v in left_values + right_values if not np.isnan(v) and not np.isinf(v)]
        if valid_values:
            ax4.set_ylim(0, max(valid_values) * 1.1)
        else:
            ax4.set_ylim(0, 1)  # Default range if all values are NaN
        ax4.set_title('ROM Radar Chart', size=12, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    else:
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')

    # 5. Histogram of ROM values
    ax5 = plt.subplot(2, 3, 5)
    all_rom_values = left_values + right_values
    # Filter out NaN and infinite values for histogram
    valid_rom_values = [v for v in all_rom_values if not np.isnan(v) and not np.isinf(v)]
    if valid_rom_values:
        ax5.hist(valid_rom_values, bins=15, alpha=0.7, color='green', edgecolor='black')
    else:
        ax5.text(0.5, 0.5, 'No valid ROM data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_xlabel('ROM (degrees)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('ROM Distribution')
    ax5.grid(True, alpha=0.3)

    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = "ROM ANALYSIS SUMMARY\n\n"
    for label, left, right, sym in zip(labels, left_values, right_values, symmetry_values):
        summary_text += (
            f"{label}:\n"
            f"  Left: {left:.1f}°\n"
            f"  Right: {right:.1f}°\n"
            f"  Symmetry: {sym:.1f}%\n\n"
        )

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comprehensive_rom_analysis.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"ROM analysis plot saved to: {fig_path}")
