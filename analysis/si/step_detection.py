import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from constants.se import (
    get_sensor_region_constants,
    NOISE_THRESHOLD, MIN_STEP_DURATION_MS, MAX_GAP_MS_FOR_MERGE
)
from utils.data_processing import get_region_average, calculate_sampling_rate
from utils.signal_processing import merge_close_periods

constants = get_sensor_region_constants()


def find_walking_segment(df: pd.DataFrame, time_col: str, window_sec: float = 2.0,
                         std_threshold_factor: float = 0.25) -> tuple[int, int]:
    left_cols = constants.LEFT_FOREFOOT + constants.LEFT_MIDFOOT + constants.LEFT_HINDFOOT
    right_cols = constants.RIGHT_FOREFOOT + constants.RIGHT_MIDFOOT + constants.RIGHT_HINDFOOT
    total_pressure = get_region_average(df, left_cols) + get_region_average(df, right_cols)

    if df.shape[0] < 10: return 0, df.shape[0]
    sampling_rate = calculate_sampling_rate(df[time_col].values)
    window_size = int(window_sec * sampling_rate)
    rolling_std = pd.Series(total_pressure).rolling(window=window_size, center=True).std().fillna(0)
    peak_std = rolling_std.max()
    if peak_std == 0: return 0, len(df)
    std_threshold = peak_std * std_threshold_factor
    is_walking = rolling_std > std_threshold
    walking_indices = np.where(is_walking)[0]
    if len(walking_indices) == 0: return 0, len(df)
    return walking_indices[0], walking_indices[-1]




def _get_region_cols_dict(side: str) -> dict[str, list[str]]:
    if side.lower() == 'left':
        return {'forefoot': constants.LEFT_FOREFOOT, 'midfoot': constants.LEFT_MIDFOOT,
                'hindfoot': constants.LEFT_HINDFOOT}
    else:
        return {'forefoot': constants.RIGHT_FOREFOOT, 'midfoot': constants.RIGHT_MIDFOOT,
                'hindfoot': constants.RIGHT_HINDFOOT}


def _regions_any_above_threshold_window(region_signals, center, window, threshold, debug=False, label=""):
    """
    Returns True if at least one region signal is above threshold in the window [center-window, center)
    """
    for region, signal in region_signals.items():
        start = max(0, center - window)
        end = center
        if end > start:
            if np.any(signal[start:end] > threshold):
                if debug:
                    print(f"DEBUG: {label} region '{region}' IS above threshold in [{start}:{end}]")
                return True
    if debug:
        print(f"DEBUG: {label} no region above threshold in [{start}:{end}]")
    return False


def _regions_any_above_threshold_window_after(region_signals, center, window, threshold, debug=False, label=""):
    """
    Returns True if at least one region signal is above threshold in the window [center+1, center+1+window)
    """
    for region, signal in region_signals.items():
        start = center + 1
        end = min(len(signal), center + 1 + window)
        if end > start:
            if np.any(signal[start:end] > threshold):
                if debug:
                    print(f"DEBUG: {label} region '{region}' IS above threshold in [{start}:{end}]")
                return True
    if debug:
        print(f"DEBUG: {label} no region above threshold in [{start}:{end}]")
    return False


def _regions_below_threshold_window(region_signals, center, window, threshold, debug=False, label=""):
    for region, signal in region_signals.items():
        start = max(0, center - window)
        end = center
        if end > start:
            if np.any(signal[start:end] > threshold):
                if debug:
                    print(f"DEBUG: {label} region '{region}' not below threshold in [{start}:{end}]")
                return False
    return True


def _regions_below_threshold_window_after(region_signals, center, window, threshold, debug=False, label=""):
    for region, signal in region_signals.items():
        start = center + 1
        end = min(len(signal), center + 1 + window)
        if end > start:
            if np.any(signal[start:end] > threshold):
                if debug:
                    print(f"DEBUG: {label} region '{region}' not below threshold in [{start}:{end}]")
                return False
    return True


def _dynamic_threshold(signal, factor=0.15):
    return np.median(signal) + factor * (np.percentile(signal, 95) - np.median(signal))


def _dynamic_prominence(signal, factor=0.15):
    return factor * (np.max(signal) - np.min(signal))


def detect_steps_for_foot(
        df, time_col, sensor_cols, base_noise_threshold, min_step_duration_ms, max_gap_ms, debug=False,
        max_loops=5
):
    """
    Adaptive step detection: tries up to max_loops with dynamic threshold/prominence.
    For first and last detected step, require at least one region above threshold in pre/post window.
    """
    foot_signal = get_region_average(df, sensor_cols)
    time = df[time_col].values
    sampling_rate = calculate_sampling_rate(time)

    side = 'left' if any('L_' in c or 'L' in c for c in sensor_cols) else 'right'
    region_cols = _get_region_cols_dict(side)
    region_signals = {name: get_region_average(df, cols) for name, cols in region_cols.items()}

    min_duration_frames = int((min_step_duration_ms / 1000.0) * sampling_rate)
    validation_window = 5

    best_steps = []
    best_count = 0
    best_params = (base_noise_threshold, _dynamic_prominence(foot_signal))

    for loop in range(max_loops):
        noise_threshold = base_noise_threshold + loop * 0.5 * np.std(foot_signal)
        prominence = _dynamic_prominence(foot_signal, factor=0.12 + 0.05 * loop)

        peaks, _ = find_peaks(foot_signal, height=noise_threshold, prominence=prominence, distance=min_duration_frames)
        is_in_contact = pd.Series(foot_signal > noise_threshold)
        diffs = is_in_contact.astype(int).diff()
        initial_starts = np.where(diffs == 1)[0]
        initial_ends = np.where(diffs == -1)[0]

        if is_in_contact.iloc[0]: initial_starts = np.insert(initial_starts, 0, 0)
        if is_in_contact.iloc[-1]: initial_ends = np.append(initial_ends, len(is_in_contact) - 1)

        min_len = min(len(initial_starts), len(initial_ends))
        initial_starts, initial_ends = initial_starts[:min_len], initial_ends[:min_len]
        merged_starts, merged_ends = merge_close_periods(initial_starts, initial_ends, time, max_gap_ms)

        candidate_steps = []
        for start, end in zip(merged_starts, merged_ends):
            if (end - start) < min_duration_frames:
                continue

            is_clean_start = True
            pre_window_start = max(0, start - validation_window)
            pre_window_end = start
            if pre_window_start < pre_window_end:
                if np.mean(foot_signal[pre_window_start:pre_window_end]) >= noise_threshold:
                    is_clean_start = False

            is_clean_end = True
            post_window_start = end + 1
            post_window_end = min(len(df), end + 1 + validation_window)
            if post_window_start < post_window_end:
                if np.mean(foot_signal[post_window_start:post_window_end]) >= noise_threshold:
                    is_clean_end = False

            candidate_steps.append((start, end, is_clean_start, is_clean_end))

        filtered_steps = []
        for i, (start, end, is_clean_start, is_clean_end) in enumerate(candidate_steps):
            # INVERTED LOGIC: For first step, require at least one region above threshold in window before start
            if i == 0:
                if not _regions_any_above_threshold_window(region_signals, start, validation_window, noise_threshold, debug, label="First step"):
                    if debug:
                        print(f"DEBUG: Discarding first step at [{start}, {end}] because no region is above threshold in window before start.")
                    continue
            # INVERTED LOGIC: For last step, require at least one region above threshold in window after end
            if i == len(candidate_steps) - 1:
                if not _regions_any_above_threshold_window_after(region_signals, end, validation_window, noise_threshold, debug, label="Last step"):
                    if debug:
                        print(f"DEBUG: Discarding last step at [{start}, {end}] because no region is above threshold in window after end.")
                    continue
            if is_clean_start and is_clean_end:
                filtered_steps.append((start, end))

        step_peaks = []
        valid_starts = []
        valid_ends = []
        for start, end in filtered_steps:
            valid_starts.append(start)
            valid_ends.append(end)
            if start < end:
                peak_in_window = np.argmax(foot_signal[start:end])
                step_peaks.append(start + peak_in_window)
            else:
                step_peaks.append(start)

        if debug:
            print(f"DEBUG[loop {loop+1}]: noise_threshold={noise_threshold:.2f}, prominence={prominence:.2f}, steps={len(step_peaks)}")

        if len(step_peaks) > best_count:
            best_steps = (np.array(step_peaks, dtype=int), np.array(valid_ends, dtype=int), np.array(valid_starts, dtype=int))
            best_count = len(step_peaks)
            best_params = (noise_threshold, prominence)

        if 8 <= len(step_peaks) <= 40:
            break

    if debug:
        print(f"DEBUG: Best step detection: threshold={best_params[0]:.2f}, prominence={best_params[1]:.2f}, steps={best_count}")

    return best_steps


def _get_all_foot_sensor_cols(side):
    if side == "left":
        return constants.LEFT_FOREFOOT + constants.LEFT_MIDFOOT + constants.LEFT_HINDFOOT
    else:
        return constants.RIGHT_FOREFOOT + constants.RIGHT_MIDFOOT + constants.RIGHT_HINDFOOT


def _calculate_ground_contact_time(time_array, contact_starts, contact_ends):
    if len(contact_starts) != len(contact_ends) or len(contact_starts) == 0: return [], 0.0
    durations = (time_array[contact_ends] - time_array[contact_starts]) / 1000.0
    return durations.tolist(), np.mean(durations) if len(durations) > 0 else 0.0


def analyze_gait(df: pd.DataFrame, time_col: str, debug=True):
    left_sensor_cols = _get_all_foot_sensor_cols("left")
    right_sensor_cols = _get_all_foot_sensor_cols("right")

    if debug: print("\n[DEBUG] Performing event-based step detection for LEFT foot...")
    left_peaks, left_falls, left_starts = detect_steps_for_foot(
        df, time_col, left_sensor_cols, NOISE_THRESHOLD, MIN_STEP_DURATION_MS, MAX_GAP_MS_FOR_MERGE, debug
    )
    if debug: print("\n[DEBUG] Performing event-based step detection for RIGHT foot...")
    right_peaks, right_falls, right_starts = detect_steps_for_foot(
        df, time_col, right_sensor_cols, NOISE_THRESHOLD, MIN_STEP_DURATION_MS, MAX_GAP_MS_FOR_MERGE, debug
    )
    time_array = df[time_col].values
    all_steps = np.unique(np.concatenate([left_peaks, right_peaks])).astype(int)
    all_steps.sort()

    if len(all_steps) < 2:
        duration_sec, cadence, step_length_mm, stride_length_mm = 0.0, 0.0, 0.0, 0.0
    else:
        start_time = time_array[int(all_steps[0])]
        end_time = time_array[int(all_steps[-1])]
        duration_sec = (end_time - start_time) / 1000.0
        cadence = (len(all_steps) / duration_sec) * 60 if duration_sec > 0 else 0
        num_intervals = len(all_steps) - 1
        total_distance_mm = num_intervals * constants.INSOLE_LENGTH_MM
        step_length_mm = total_distance_mm / len(all_steps) if len(all_steps) > 0 else 0
        stride_length_mm = 2 * step_length_mm

    left_contact_times, avg_left_contact = _calculate_ground_contact_time(time_array, left_starts, left_falls)
    right_contact_times, avg_right_contact = _calculate_ground_contact_time(time_array, right_starts, right_falls)

    total_contact = avg_left_contact + avg_right_contact
    left_pct = 100 * avg_left_contact / total_contact if total_contact > 0 else 0.0
    right_pct = 100 * avg_right_contact / total_contact if total_contact > 0 else 0.0

    if debug:
        print(f"\nDEBUG: Final step count: {len(all_steps)}")
        print(f"DEBUG: Duration (s): {duration_sec:.2f}, Cadence: {cadence:.2f}")

    return {
        'time_taken_s': duration_sec, 'steps': len(all_steps), 'cadence_steps_per_min': cadence,
        'step_length_mm': step_length_mm, 'stride_length_mm': stride_length_mm,
        'steps_left': len(left_peaks), 'steps_right': len(right_peaks),
        'avg_left_ground_contact_time_s': avg_left_contact, 'avg_right_ground_contact_time_s': avg_right_contact,
        'left_ground_contact_time_pct': left_pct, 'right_ground_contact_time_pct': right_pct,
        'left_ground_contact_times_per_step': left_contact_times,
        'right_ground_contact_times_per_step': right_contact_times,
        'left_step_indices': left_peaks, 'left_fall_indices': left_falls, 'left_start_indices': left_starts,
        'right_step_indices': right_peaks, 'right_fall_indices': right_falls, 'right_start_indices': right_starts,
    }