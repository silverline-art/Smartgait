import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from config import output_path
from analysis.pe import gait_detect

def visualize_gait_analysis(df: pd.DataFrame, side: str = "left"):
    """
    Generates a single dashboard PNG with focused gait visualizations as subplots.
    The dashboard includes:
      1. Foot trajectory for both left and right foot with gait events
      2. Stride length, step length, and duration per cycle (merged bar plot)
      3. Support phases over time
      4. Gait summary text
    """
    out_dir = os.path.join(output_path, "plot")
    os.makedirs(out_dir, exist_ok=True)
    dashboard_path = os.path.join(out_dir, "gait_dashboard.png")

    # Analyze for both left and right foot
    analyzers = {
        "left": gait_detect.GaitCycleAnalyzer(df, side="left"),
        "right": gait_detect.GaitCycleAnalyzer(df, side="right")
    }
    results = {s: a.analyze() for s, a in analyzers.items()}

    # Prepare gait events for both sides
    gait_events = {
        "left": {
            "heel_strikes": results["left"]["foot_strike_indices"],
            "toe_offs": results["left"]["foot_off_indices"]
        },
        "right": {
            "heel_strikes": results["right"]["foot_strike_indices"],
            "toe_offs": results["right"]["foot_off_indices"]
        }
    }

    # Prepare gait cycles for the selected side (default: left)
    selected = side
    gait_cycles = []
    for i, (start, end) in enumerate(results[selected]["gait_cycles"]):
        gait_cycles.append({
            "stride_length": results[selected]["stride_lengths"][i] * 100 if i < len(results[selected]["stride_lengths"]) else None,
            "step_length": results[selected]["step_lengths"][i] * 100 if i < len(results[selected]["step_lengths"]) else None,
            "duration": results[selected]["foot_strike_times"][i+1] - results[selected]["foot_strike_times"][i] if i+1 < len(results[selected]["foot_strike_times"]) else None,
            "start_frame": start,
            "end_frame": end,
            "support_phase": "single"  # Placeholder; update if you have phase info
        })
    stride_lengths = results[selected]["stride_lengths"] * 100
    step_lengths = results[selected]["step_lengths"] * 100
    stance_durations = results[selected]["stance_durations"]
    swing_durations = results[selected]["swing_durations"]

    stats = {
        "mean_stride_length": np.nanmean(stride_lengths) if len(stride_lengths) > 0 else np.nan,
        "mean_step_length": np.nanmean(step_lengths) if len(step_lengths) > 0 else np.nan,
        "mean_cadence": results[selected]["cadence"],
        "mean_speed_cm_s": results[selected]["mean_speed"] * 100 if results[selected][
                                                                        "mean_speed"] is not None else np.nan,
        "mean_stance_duration": np.nanmean(stance_durations) if len(stance_durations) > 0 else np.nan,
        "mean_swing_duration": np.nanmean(swing_durations) if len(swing_durations) > 0 else np.nan,
        "gait_cycles": len(results[selected]["gait_cycles"])
    }

    if "frame" not in df.columns:
        df["frame"] = df.index
    # Add foot_x/foot_y for both sides if not present
    for s in ["left", "right"]:
        if f"{s}_foot_index_x" in df.columns and f"{s}_foot_index_y" in df.columns:
            df[f"{s}_foot_x"] = df[f"{s}_foot_index_x"]
            df[f"{s}_foot_y"] = df[f"{s}_foot_index_y"]
        else:
            raise ValueError(f"{s}_foot_index_x and {s}_foot_index_y columns not found in DataFrame.")

    # Create dashboard figure (2 rows x 2 columns)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. Foot trajectory subplot for both feet (Y vs. Time)
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot left and right foot Y vs. time
    ax1.plot(df['time'], df['left_foot_y'], color='blue', label='Left Foot Y', alpha=0.7)
    ax1.plot(df['time'], df['right_foot_y'], color='orange', label='Right Foot Y', alpha=0.7)

    # Overlay gait events for both feet
    for side, color in zip(['left', 'right'], ['blue', 'orange']):
        strikes = results[side]["foot_strike_indices"]
        offs = results[side]["foot_off_indices"]
        if strikes is not None and len(strikes) > 0:
            ax1.scatter(df.loc[strikes, 'time'], df.loc[strikes, f'{side}_foot_y'],
                        color=color, marker='o', s=60, label=f'{side.capitalize()} Strikes')
        if offs is not None and len(offs) > 0:
            ax1.scatter(df.loc[offs, 'time'], df.loc[offs, f'{side}_foot_y'],
                        color=color, marker='x', s=60, label=f'{side.capitalize()} Offs')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Foot Y Position')
    ax1.set_title('Foot Y Trajectory Over Time (Left & Right)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)


    # 2. Stride/step length and duration per cycle subplot (for selected side)
    # COMMENT TWO: This subplot merges stride length, step length, and duration per cycle into a grouped bar chart.
    stride_lengths = [cycle.get('stride_length', np.nan) for cycle in gait_cycles]
    step_lengths = [cycle.get('step_length', np.nan) for cycle in gait_cycles]
    durations = [cycle.get('duration', np.nan) for cycle in gait_cycles]
    cycles = np.arange(len(gait_cycles))
    width =  0.25

    ax2 = fig.add_subplot(gs[0, 1])
    # Draw duration first (back), then stride, then step (top)
    ax2.bar(cycles, durations, width=width, color='orange', alpha=0.5, label='Duration')
    ax2.bar(cycles, stride_lengths, width=width, color='skyblue', alpha=0.7, label='Stride Length')
    ax2.bar(cycles, step_lengths, width=width, color='lightgreen', alpha=0.7, label='Step Length')
    ax2.set_title('Stride Length, Step Length, and Duration per Cycle')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    # 3. Support phases subplot (for selected side)
    ax3 = fig.add_subplot(gs[1, 0])

    n_frames = len(df)
    left_strikes = np.array(results["left"]["foot_strike_indices"])
    left_offs = np.array(results["left"]["foot_off_indices"])
    right_strikes = np.array(results["right"]["foot_strike_indices"])
    right_offs = np.array(results["right"]["foot_off_indices"])

    support_phase = gait_detect.GaitCycleAnalyzer.compute_support_phase_per_frame(
        n_frames, left_strikes, left_offs, right_strikes, right_offs
    )

    t = np.arange(n_frames)

    # --- BACKGROUND: stance (green) vs swing (orange) ---
    stance_mask = (support_phase == 1) | (support_phase == 2) | (support_phase == 3)
    swing_mask = (support_phase == 0)
    ax3.fill_between(t, 0, 1.2, where=stance_mask, color='green', alpha=0.2, step='post', label='Stance Phase')
    ax3.fill_between(t, 0, 1.2, where=swing_mask, color='orange', alpha=0.2, step='post', label='Swing Phase')

    bar_ymin = 0.4  # Lower edge of the bar (centered in the plot)
    bar_ymax = 0.7  # Upper edge of the bar
    bar_height = bar_ymax - bar_ymin

    current_phase = support_phase[0]
    start_idx = 0
    for i in range(1, n_frames):
        if support_phase[i] != current_phase or i == n_frames - 1:
            end_idx = i
            color = None
            label = None
            if current_phase == 1:
                color = 'blue'
                label = 'Single Support Left'
            elif current_phase == 2:
                color = 'red'
                label = 'Single Support Right'
            elif current_phase == 3:
                color = 'purple'
                label = 'Double Support'
            # Only add label for the first occurrence to avoid duplicate legend entries
            if color:
                if label and not any([l.get_label() == label for l in ax3.patches]):
                    ax3.fill_between([start_idx, end_idx], bar_ymin, bar_ymax, color=color, alpha=1.0, label=label)
                else:
                    ax3.fill_between([start_idx, end_idx], bar_ymin, bar_ymax, color=color, alpha=1.0)
            start_idx = i
            current_phase = support_phase[i]

    ax3.set_ylim(0, 1.2)
    ax3.set_yticks([])
    ax3.set_xlabel('Frame')
    ax3.set_title('Support Phases with Stance/Swing')
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, alpha=0.2, label='Stance Phase'),
        Line2D([0], [0], color='orange', lw=4, alpha=0.2, label='Swing Phase'),
        Line2D([0], [0], color='blue', lw=4, label='Single Support Left'),
        Line2D([0], [0], color='red', lw=4, label='Single Support Right'),
        Line2D([0], [0], color='purple', lw=4, label='Double Support')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')


    # 4. Gait summary text subplot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary = "GAIT ANALYSIS SUMMARY\n\n"
    for k, v in stats.items():
        try:
            summary += f"{k.replace('_', ' ').title()}: {float(v):.2f}\n"
        except Exception:
            summary += f"{k.replace('_', ' ').title()}: {v}\n"
    ax4.text(0.05, 0.95, summary, fontsize=12, va='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig(dashboard_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {dashboard_path}")