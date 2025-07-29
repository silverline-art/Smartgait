import numpy as np
import pandas as pd

def print_gait_report(result, return_str=False):
    """Print basic gait metrics in a formatted report."""
    lines = []
    lines.append("=" * 60)
    lines.append("🚶‍♀️ BASIC TEST SUMMARY")
    lines.append("=" * 60)

    time_taken = result.get('time_taken_s', 0)
    steps = result.get('steps', 0)
    cadence = result.get('cadence_steps_per_min', 0)
    step_length = result.get('step_length_mm', 0)
    stride_length = result.get('stride_length_mm', 0)

    lines.append(f"Time Taken: {time_taken:.2f} seconds")
    lines.append(f"Steps Taken: {steps} steps")
    lines.append(f"Cadence: {cadence:.1f} steps/min")
    lines.append(f"Step Length: {step_length:.2f} mm")
    lines.append(f"Stride Length: {stride_length:.2f} mm")

    # Foot pressure distribution
    lines.append("")
    lines.append("🦶 FOOT PRESSURE DISTRIBUTION (Stance Phase %)")
    lines.append("-" * 60)

    left_pressure = result.get('left_pressure_distribution', {})
    right_pressure = result.get('right_pressure_distribution', {})

    lines.append(f"Left Foot: Forefoot: {left_pressure.get('forefoot', 0):.2f}%, "
                 f"Midfoot: {left_pressure.get('midfoot', 0):.2f}%, "
                 f"Hindfoot: {left_pressure.get('hindfoot', 0):.2f}%")
    lines.append(f"Right Foot: Forefoot: {right_pressure.get('forefoot', 0):.2f}%, "
                 f"Midfoot: {right_pressure.get('midfoot', 0):.2f}%, "
                 f"Hindfoot: {right_pressure.get('hindfoot', 0):.2f}%")

    # First contact regions
    lines.append("")
    lines.append("🦶 TRUE FIRST CONTACT POINT")
    lines.append("-" * 60)

    left_fc = result.get('left_first_contact_region_stats', {})
    right_fc = result.get('right_first_contact_region_stats', {})

    left_counts = left_fc.get('region_counts', {})
    right_counts = right_fc.get('region_counts', {})
    left_pcts = left_fc.get('region_percentages', {})
    right_pcts = right_fc.get('region_percentages', {})

    left_valid = sum(left_counts.values()) - left_counts.get('none', 0)
    right_valid = sum(right_counts.values()) - right_counts.get('none', 0)

    lines.append(f"Left Foot (Valid Steps: {left_valid})")
    lines.append(f"Forefoot: {left_pcts.get('forefoot', 0):.1f}% (count: {left_counts.get('forefoot', 0)})")
    lines.append(f"Midfoot: {left_pcts.get('midfoot', 0):.1f}% (count: {left_counts.get('midfoot', 0)})")
    lines.append(f"Hindfoot: {left_pcts.get('hindfoot', 0):.1f}% (count: {left_counts.get('hindfoot', 0)})")
    lines.append("")
    lines.append(f"Right Foot (Valid Steps: {right_valid})")
    lines.append(f"Forefoot: {right_pcts.get('forefoot', 0):.1f}% (count: {right_counts.get('forefoot', 0)})")
    lines.append(f"Midfoot: {right_pcts.get('midfoot', 0):.1f}% (count: {right_counts.get('midfoot', 0)})")
    lines.append(f"Hindfoot: {right_pcts.get('hindfoot', 0):.1f}% (count: {right_counts.get('hindfoot', 0)})")

    # Ground contact time
    lines.append("")
    lines.append("🦶 FOOT GROUND CONTACT TIME")
    lines.append("-" * 60)

    left_contact = result.get('avg_left_ground_contact_time_s', 0)
    right_contact = result.get('avg_right_ground_contact_time_s', 0)
    left_pct = result.get('left_ground_contact_time_pct', 0)
    right_pct = result.get('right_ground_contact_time_pct', 0)

    lines.append(f"Left Foot: Average: {left_contact:.2f} s, Total Contact %: {left_pct:.2f}%")
    lines.append(f"Right Foot: Average: {right_contact:.2f} s, Total Contact %: {right_pct:.2f}%")

    lines.append("=" * 60)

    if return_str:
        return "\n".join(lines)
    else:
        print("\n".join(lines))


def calculate_phase_percentages(result):
    """Calculate gait phase percentages (mean, std) from temporal parameters."""
    summary = result.get('summary', {})

    # Get stride times for percentage calculations
    stride_left = summary.get('stride_time_left', (np.nan, np.nan))
    stride_right = summary.get('stride_time_right', (np.nan, np.nan))

    def get_percentage(phase_key, stride_tuple):
        """Convert phase time to percentage of stride, propagate std."""
        phase_tuple = summary.get(phase_key, (np.nan, np.nan))
        mean, std = phase_tuple
        stride_mean, stride_std = stride_tuple
        if np.isnan(mean) or np.isnan(stride_mean) or stride_mean == 0:
            return (np.nan, np.nan)
        pct_mean = (mean / stride_mean) * 100
        # Propagate error: std_p = p * sqrt((std/mean)^2 + (stride_std/stride_mean)^2)
        rel_std = (std / mean) if mean != 0 else 0
        rel_stride_std = (stride_std / stride_mean) if stride_mean != 0 else 0
        pct_std = abs(pct_mean) * np.sqrt(rel_std**2 + rel_stride_std**2)
        return (pct_mean, pct_std)

    # Calculate percentages for each phase
    percentages = {
        'stance_phase_left': get_percentage('stance_time_left', stride_left),
        'stance_phase_right': get_percentage('stance_time_right', stride_right),
        'swing_phase_left': get_percentage('swing_time_left', stride_left),
        'swing_phase_right': get_percentage('swing_time_right', stride_right),
        'single_support_left': get_percentage('single_support_time_left', stride_left),
        'single_support_right': get_percentage('single_support_time_right', stride_right),
        'mid_stance_left': get_percentage('mid_stance_time_left', stride_left),
        'mid_stance_right': get_percentage('mid_stance_time_right', stride_right),
        'terminal_stance_left': get_percentage('terminal_stance_time_left', stride_left),
        'terminal_stance_right': get_percentage('terminal_stance_time_right', stride_right),
        'double_support_left': get_percentage('double_support_time_left', stride_left),
        'double_support_right': get_percentage('double_support_time_right', stride_right),
        'loading_response_left': get_percentage('loading_response_time_left', stride_left),
        'loading_response_right': get_percentage('loading_response_time_right', stride_right),
        'pre_swing_left': get_percentage('pre_swing_time_left', stride_left),
        'pre_swing_right': get_percentage('pre_swing_time_right', stride_right),
    }

    return percentages


def print_temporal_parameter_table(result, time_array, debug=False):
    """Print temporal parameters as absolute times."""
    params = result.get('summary', {})
    param_names = [
        ("Stride time (s)", "stride_time"),
        ("Step time (s)", "step_time"),
        ("Stance time (s)", "stance_time"),
        ("Swing time (s)", "swing_time"),
        ("Single support time (s)", "single_support_time"),
        ("Mid stance time (s)", "mid_stance_time"),
        ("Terminal stance time (s)", "terminal_stance_time"),
        ("Double support time (s)", "double_support_time"),
        ("Loading response time (s)", "loading_response_time"),
        ("Pre-swing time (s)", "pre_swing_time"),
    ]

    def format_value(val):
        if isinstance(val, tuple) and len(val) == 2:
            if np.isnan(val[0]) or np.isnan(val[1]):
                return "nan ± nan"
            return f"{val[0]:.2f} ± {val[1]:.2f}"
        elif isinstance(val, float):
            if np.isnan(val):
                return "nan"
            return f"{val:.2f}"
        elif isinstance(val, (int, np.integer)):
            return str(val)
        else:
            return str(val)

    def diff_tuple(a, b):
        if a is None or b is None or np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return (np.nan, np.nan)
        return (a[0] - b[0], np.sqrt(a[1] ** 2 + b[1] ** 2))

    print("\n" + "=" * 80)
    print(f"{'Parameter':<28} {'Left Leg':<20} {'Right Leg':<20} {'Difference':<20}")
    print("-" * 80)

    for label, key in param_names:
        left = params.get(f"{key}_left", (np.nan, np.nan))
        right = params.get(f"{key}_right", (np.nan, np.nan))
        diff = diff_tuple(left, right)
        print(f"{label:<28} {format_value(left):<20} {format_value(right):<20} {format_value(diff):<20}")

    print("=" * 80 + "\n")


def print_gait_phase_percentages(result):
    """Print gait phase percentages table with mean ± std."""
    percentages = calculate_phase_percentages(result)

    phase_names = [
        ("Stance phase (%)", "stance_phase"),
        ("Swing phase (%)", "swing_phase"),
        ("Single support (%)", "single_support"),
        ("Mid stance (%)", "mid_stance"),
        ("Terminal stance (%)", "terminal_stance"),
        ("Double support (%)", "double_support"),
        ("Loading response (%)", "loading_response"),
        ("Pre-swing (%)", "pre_swing"),
    ]

    def format_percentage(val):
        if isinstance(val, tuple) and len(val) == 2:
            if np.isnan(val[0]) or np.isnan(val[1]):
                return "nan ± nan"
            return f"{val[0]:.1f} ± {val[1]:.1f}"
        elif isinstance(val, float):
            if np.isnan(val):
                return "nan"
            return f"{val:.1f}"
        else:
            return str(val)

    def diff_tuple(a, b):
        if a is None or b is None or np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return (np.nan, np.nan)
        return (a[0] - b[0], np.sqrt(a[1] ** 2 + b[1] ** 2))

    print("\n" + "=" * 80)
    print("GAIT CYCLE PHASE PERCENTAGES with standard deviation")
    print("=" * 80)
    print(f"{'Phase':<25} {'Left':<20} {'Right':<20} {'Difference':<20}")
    print("-" * 80)

    for label, key in phase_names:
        left = percentages.get(f"{key}_left", (np.nan, np.nan))
        right = percentages.get(f"{key}_right", (np.nan, np.nan))
        diff = diff_tuple(left, right)
        print(f"{label:<25} {format_percentage(left):<20} {format_percentage(right):<20} {format_percentage(diff):<20}")

    print("=" * 80 + "\n")