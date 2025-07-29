import numpy as np
import pandas as pd
from scipy.integrate import simpson
from constants.se import get_sensor_region_constants, NOISE_THRESHOLD
from utils.data_processing import get_region_average

constants = get_sensor_region_constants()


def _get_region_cols_dict(side: str) -> dict[str, list[str]]:
    # Always return regions in the order: forefoot, midfoot, hindfoot
    if side.lower() == 'left':
        return {
            'forefoot': constants.LEFT_FOREFOOT,
            'midfoot': constants.LEFT_MIDFOOT,
            'hindfoot': constants.LEFT_HINDFOOT
        }
    else:
        return {
            'forefoot': constants.RIGHT_FOREFOOT,
            'midfoot': constants.RIGHT_MIDFOOT,
            'hindfoot': constants.RIGHT_HINDFOOT
        }


def analyze_stance_phase_pressure_distribution(
        df: pd.DataFrame, contact_starts: list[int], contact_ends: list[int], side: str, debug: bool = False
) -> dict[str, float]:
    regions = _get_region_cols_dict(side)
    per_step_dist = {r: [] for r in regions}
    starts, ends = np.array(contact_starts, dtype=int), np.array(contact_ends, dtype=int)

    if starts.size == 0:
        return {r: 0.0 for r in regions}

    for s, e in zip(starts, ends):
        if e <= s:
            continue
        win_df = df.iloc[s:e]
        if win_df.shape[0] < 2:
            continue
        aucs = {r: simpson(get_region_average(win_df, cols), dx=1) for r, cols in regions.items()}
        total_auc = sum(aucs.values())
        for r in regions:
            per_step_dist[r].append((aucs[r] / total_auc * 100) if total_auc > 0 else 0.0)

    return {r: float(np.mean(per_step_dist[r])) if per_step_dist[r] else 0.0 for r in regions}


def calculate_first_contact_region_stats(
        df: pd.DataFrame, contact_starts: list[int], contact_ends: list[int],
        region_cols_dict: dict[str, list[str]], noise_threshold: float, debug: bool = False
) -> dict[str, any]:
    starts, ends = list(map(int, contact_starts)), list(map(int, contact_ends))
    region_names = list(region_cols_dict.keys())
    counts = {region: 0 for region in region_names}
    counts['none'] = 0
    total_steps_analyzed = len(starts)

    if total_steps_analyzed == 0:
        return {'counts': counts, 'percentages': {r: 0.0 for r in counts}, 'total_steps': 0}

    # For debug: track which region was first for each step
    debug_first_regions = []

    for i in range(total_steps_analyzed):
        df_step_window = df.iloc[starts[i]:ends[i]]
        if df_step_window.empty:
            counts['none'] += 1
            debug_first_regions.append('none')
            continue

        first_touch_indices = {}
        for region, cols in region_cols_dict.items():
            region_signal = get_region_average(df_step_window, cols)
            above_threshold_indices = np.where(region_signal > noise_threshold)[0]
            if len(above_threshold_indices) > 0:
                first_touch_indices[region] = above_threshold_indices[0]

        if not first_touch_indices:
            counts['none'] += 1
            debug_first_regions.append('none')
            continue

        # Find the region with the minimum index (earliest contact)
        true_first_contact_region = min(first_touch_indices, key=first_touch_indices.get)
        counts[true_first_contact_region] += 1
        debug_first_regions.append(true_first_contact_region)

        if debug:
            print(f"[DEBUG] Step {i+1}: First contact indices: {first_touch_indices} -> {true_first_contact_region}")

    valid_steps = total_steps_analyzed - counts['none']
    percentages = {r: (c / valid_steps * 100 if valid_steps > 0 else 0.0) for r, c in counts.items() if r != 'none'}

    if debug:
        print(f"[DEBUG] True First Contact region counts: {counts}")
        print(f"[DEBUG] True First Contact region percentages: {percentages}")
        print(f"[DEBUG] Step-by-step first contact regions: {debug_first_regions}")

    return {'counts': counts, 'percentages': percentages, 'total_steps': total_steps_analyzed}


def compute_and_inject_initial_pressure(
        step_result: dict, df: pd.DataFrame, debug: bool = False
) -> None:
    left_starts, right_starts = step_result.get('left_start_indices', []), step_result.get('right_start_indices', [])
    left_falls, right_falls = step_result.get('left_fall_indices', []), step_result.get('right_fall_indices', [])

    pl = analyze_stance_phase_pressure_distribution(df, left_starts, left_falls, 'left', debug)
    pr = analyze_stance_phase_pressure_distribution(df, right_starts, right_falls, 'right', debug)
    step_result.update({'pressure_left': pl, 'pressure_right': pr})

    left_cols, right_cols = _get_region_cols_dict('left'), _get_region_cols_dict('right')
    lfcs = calculate_first_contact_region_stats(df, left_starts, left_falls, left_cols, NOISE_THRESHOLD, debug)
    rfcs = calculate_first_contact_region_stats(df, right_starts, right_falls, right_cols, NOISE_THRESHOLD, debug)
    step_result.update({'left_first_contact_region_stats': lfcs, 'right_first_contact_region_stats': rfcs})