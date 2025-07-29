import numpy as np
import pandas as pd
from analysis.si.step_detection import get_region_average

# Global configuration, per‐trial quantile bounds
CFG = {
    'min_stance_s': 0.1,
    'max_stance_s': 2.0,
    'min_stride_s': 0.3,
    'max_stride_s': 2.5,
    'min_swing_s': 0.1,
    'max_swing_s': 2.0,
    'min_phase_s': 0.05,
    'max_phase_s': 1.5,
    'low_quantile': 5,  # More permissive
    'high_quantile': 95,  # More permissive
    'phase_tol_pct': 0.15  # More tolerant
}


def extract_time_array_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extract time column (ms) or default to index."""
    for col in df.columns:
        if col.lower().startswith(('time', 'timestamp')):
            return df[col].values
    return df.index.values


def _safe_divide_arrays(numerator, denominator):
    """Safely divide two arrays of potentially different lengths."""
    if len(numerator) == 0 or len(denominator) == 0:
        return np.array([])
    
    # Align arrays to same length (use minimum length)
    min_len = min(len(numerator), len(denominator))
    num_aligned = numerator[:min_len]
    den_aligned = denominator[:min_len]
    
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(num_aligned, den_aligned)
        result[den_aligned == 0] = 0  # Replace inf/nan with 0
        return result


def _phase_stats(arr, min_val=None, max_val=None):
    """Filter by bounds then return (mean,std) or (nan,nan)."""
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    if min_val is not None:
        a = a[a >= min_val]
    if max_val is not None:
        a = a[a <= max_val]
    return (float(np.mean(a)), float(np.std(a))) if len(a) else (np.nan, np.nan)


def _align_and_subtract(a, b):
    """Truncate to same length, subtract, clamp negatives to nan."""
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    d = a - b
    d[d < 0] = np.nan
    return d


def calculate_stance_times_robust(hs_indices, to_indices, time_s):
    """
    Robust stance time calculation: for each HS, find the next TO.
    """
    n_strides = len(hs_indices) - 1
    if n_strides <= 0:
        return np.array([])

    stance_times = np.full(n_strides, np.nan)
    hs_sorted = np.sort(hs_indices)
    to_sorted = np.sort(to_indices)

    for i in range(n_strides):
        hs_current = hs_sorted[i]
        hs_next = hs_sorted[i + 1]

        # Find TO events between current HS and next HS
        valid_tos = to_sorted[(to_sorted > hs_current) & (to_sorted < hs_next)]

        if len(valid_tos) > 0:
            to_event = valid_tos[0]  # Take first TO after HS
            stance_time = time_s[to_event] - time_s[hs_current]
            if CFG['min_stance_s'] <= stance_time <= CFG['max_stance_s']:
                stance_times[i] = stance_time

    return stance_times


def calculate_loading_response_robust(ipsi_hs, contra_to, time_s):
    """
    Loading response: from ipsi HS to next contra TO.
    """
    n_strides = len(ipsi_hs) - 1
    if n_strides <= 0:
        return np.array([])

    lr_times = np.full(n_strides, np.nan)
    hs_sorted = np.sort(ipsi_hs)
    to_sorted = np.sort(contra_to)

    for i in range(n_strides):
        hs_current = hs_sorted[i]
        hs_next = hs_sorted[i + 1] if i + 1 < len(hs_sorted) else hs_current + 1000

        # Find contra TO after current HS
        valid_tos = to_sorted[(to_sorted > hs_current) & (to_sorted < hs_next)]

        if len(valid_tos) > 0:
            to_event = valid_tos[0]
            lr_time = time_s[to_event] - time_s[hs_current]
            if CFG['min_phase_s'] <= lr_time <= CFG['max_phase_s']:
                lr_times[i] = lr_time

    return lr_times


def calculate_pre_swing_robust(ipsi_to, contra_hs, time_s):
    """
    Pre-swing: from contra HS to next ipsi TO.
    """
    if len(contra_hs) == 0 or len(ipsi_to) == 0:
        return np.array([])

    ps_times = []
    hs_sorted = np.sort(contra_hs)
    to_sorted = np.sort(ipsi_to)

    for hs in hs_sorted:
        # Find next ipsi TO after this contra HS
        valid_tos = to_sorted[to_sorted > hs]

        if len(valid_tos) > 0:
            to_event = valid_tos[0]
            ps_time = time_s[to_event] - time_s[hs]
            if CFG['min_phase_s'] <= ps_time <= CFG['max_phase_s']:
                ps_times.append(ps_time)

    return np.array(ps_times)


def compute_single_support_robust(t_s, ipsi_start, ipsi_end, contra_start, contra_end, hs_indices):
    """
    Single support per stride = ipsi contact AND contra not contact.
    """
    n = len(t_s)
    ipsi_contact = np.zeros(n, bool)
    contra_contact = np.zeros(n, bool)

    # Build contact arrays
    for s, e in zip(ipsi_start, ipsi_end):
        if 0 <= s < n and 0 <= e < n:
            ipsi_contact[s:e + 1] = True

    for s, e in zip(contra_start, contra_end):
        if 0 <= s < n and 0 <= e < n:
            contra_contact[s:e + 1] = True

    n_strides = len(hs_indices) - 1
    if n_strides <= 0:
        return np.array([])

    ss_times = np.full(n_strides, np.nan)
    dt = np.diff(t_s, prepend=t_s[0])

    for i in range(n_strides):
        start_idx = hs_indices[i]
        end_idx = hs_indices[i + 1]

        if end_idx <= start_idx:
            continue

        stride_indices = np.arange(start_idx, end_idx)
        single_support_mask = ipsi_contact[stride_indices] & ~contra_contact[stride_indices]

        if np.any(single_support_mask):
            ss_time = np.sum(dt[stride_indices[single_support_mask]])
            ss_times[i] = ss_time

    return ss_times


def compute_terminal_stance_robust(df, time_col, hs_indices, to_indices, hind_cols, mid_cols, fore_cols):
    """
    Terminal stance: hind<20% & (mid>20% or fore>20%)
    """
    if time_col is None or len(hs_indices) == 0:
        return np.array([])

    time_array = df[time_col].values / 1000.0
    n_strides = len(hs_indices) - 1
    if n_strides <= 0:
        return np.array([])

    terminal_times = np.full(n_strides, np.nan)
    to_sorted = np.sort(to_indices)

    for i in range(n_strides):
        hs_current = hs_indices[i]
        hs_next = hs_indices[i + 1]

        # Find TO between current and next HS
        valid_tos = to_sorted[(to_sorted > hs_current) & (to_sorted < hs_next)]

        if len(valid_tos) == 0:
            continue

        to_event = valid_tos[0]

        try:
            stance_window = df.iloc[hs_current:to_event + 1]
            stance_times = time_array[hs_current:to_event + 1]

            if len(stance_window) == 0:
                continue

            # Calculate region averages
            hind_avg = stance_window[hind_cols].mean(axis=1).values
            mid_avg = stance_window[mid_cols].mean(axis=1).values
            fore_avg = stance_window[fore_cols].mean(axis=1).values

            # Normalize
            with np.errstate(invalid='ignore', divide='ignore'):
                hind_norm = hind_avg / (np.max(hind_avg) if np.max(hind_avg) > 0 else 1)
                mid_norm = mid_avg / (np.max(mid_avg) if np.max(mid_avg) > 0 else 1)
                fore_norm = fore_avg / (np.max(fore_avg) if np.max(fore_avg) > 0 else 1)

            # Terminal stance condition
            terminal_mask = (hind_norm < 0.2) & ((mid_norm > 0.2) | (fore_norm > 0.2))

            if np.any(terminal_mask):
                terminal_periods = stance_times[terminal_mask]
                if len(terminal_periods) > 1:
                    terminal_times[i] = terminal_periods[-1] - terminal_periods[0]
                else:
                    terminal_times[i] = 0.05  # Minimum terminal stance
        except (IndexError, KeyError):
            continue

    return terminal_times


def inject_time_parameters(
        result: dict,
        df: pd.DataFrame,
        debug: bool = False,
        left_hindfoot_cols=None,
        left_midfoot_cols=None,
        left_forefoot_cols=None,
        right_hindfoot_cols=None,
        right_midfoot_cols=None,
        right_forefoot_cols=None,
        time_col=None
):
    """
    Compute + inject all gait temporal parameters.
    Uses robust calculation methods with fallbacks.
    """
    # Extract time in seconds
    t_ms = extract_time_array_from_df(df)
    t_s = t_ms / 1000.0

    # Get event indices - use start_indices as heel strikes for better accuracy
    lhs = np.sort(result.get('left_start_indices', result.get('left_step_indices', [])))
    lto = np.sort(result.get('left_fall_indices', []))
    lstart = np.sort(result.get('left_start_indices', []))

    rhs = np.sort(result.get('right_start_indices', result.get('right_step_indices', [])))
    rto = np.sort(result.get('right_fall_indices', []))
    rstart = np.sort(result.get('right_start_indices', []))

    if debug:
        print(f"\n[DEBUG] Event counts:")
        print(f"  Left HS: {len(lhs)}, Left TO: {len(lto)}")
        print(f"  Right HS: {len(rhs)}, Right TO: {len(rto)}")

    # Stride times
    st_l = np.diff(t_s[lhs]) if len(lhs) > 1 else np.array([])
    st_r = np.diff(t_s[rhs]) if len(rhs) > 1 else np.array([])

    # Robust stance time calculations
    stance_l = calculate_stance_times_robust(lhs, lto, t_s)
    stance_r = calculate_stance_times_robust(rhs, rto, t_s)

    # Swing times = stride - stance
    swing_l = _align_and_subtract(st_l, stance_l)
    swing_r = _align_and_subtract(st_r, stance_r)

    # Loading response times
    lr_l = calculate_loading_response_robust(lhs, rto, t_s)
    lr_r = calculate_loading_response_robust(rhs, lto, t_s)

    # Pre-swing times
    ps_l = calculate_pre_swing_robust(lto, rhs, t_s)
    ps_r = calculate_pre_swing_robust(rto, lhs, t_s)

    # Double support = loading response + pre-swing (when both available)
    ds_l = lr_l + ps_l[:len(lr_l)] if len(ps_l) >= len(lr_l) else np.full_like(lr_l, np.nan)
    ds_r = lr_r + ps_r[:len(lr_r)] if len(ps_r) >= len(lr_r) else np.full_like(lr_r, np.nan)

    # Single support per stride
    ss_l = compute_single_support_robust(t_s, lstart, lto, rstart, rto, lhs)
    ss_r = compute_single_support_robust(t_s, rstart, rto, lstart, lto, rhs)

    # Terminal stance via regions
    term_l = compute_terminal_stance_robust(
        df, time_col, lhs, lto,
        left_hindfoot_cols or [], left_midfoot_cols or [], left_forefoot_cols or []
    )
    term_r = compute_terminal_stance_robust(
        df, time_col, rhs, rto,
        right_hindfoot_cols or [], right_midfoot_cols or [], right_forefoot_cols or []
    )

    # Mid-stance = single_support - terminal
    mid_l = _align_and_subtract(ss_l, term_l)
    mid_r = _align_and_subtract(ss_r, term_r)

    # Step times (alternating foot contacts)
    all_hs_times = np.sort(np.concatenate([t_s[lhs], t_s[rhs]]))
    steps = np.diff(all_hs_times) if len(all_hs_times) > 1 else np.array([])

    # Swing phase percentage
    with np.errstate(divide='ignore', invalid='ignore'):
        sp_l = (swing_l / st_l) * 100
        sp_r = (swing_r / st_r) * 100

    if debug:
        print(f"\n[DEBUG] Raw phase lengths:")
        print(f"  Stance L/R: {len(stance_l)}/{len(stance_r)}")
        print(f"  Swing L/R: {len(swing_l)}/{len(swing_r)}")
        print(f"  Single support L/R: {len(ss_l)}/{len(ss_r)}")
        print(f"  Loading response L/R: {len(lr_l)}/{len(lr_r)}")
        print(f"  Pre-swing L/R: {len(ps_l)}/{len(ps_r)}")

    # Collect summary statistics
    params = {
        'stride_time_left': _phase_stats(st_l, CFG['min_stride_s'], CFG['max_stride_s']),
        'stride_time_right': _phase_stats(st_r, CFG['min_stride_s'], CFG['max_stride_s']),
        'step_time_left': _phase_stats(steps, CFG['min_stride_s'] / 2, CFG['max_stride_s']),
        'step_time_right': _phase_stats(steps, CFG['min_stride_s'] / 2, CFG['max_stride_s']),
        'stance_time_left': _phase_stats(stance_l, CFG['min_stance_s'], CFG['max_stance_s']),
        'stance_time_right': _phase_stats(stance_r, CFG['min_stance_s'], CFG['max_stance_s']),
        'swing_time_left': _phase_stats(swing_l, CFG['min_swing_s'], CFG['max_swing_s']),
        'swing_time_right': _phase_stats(swing_r, CFG['min_swing_s'], CFG['max_swing_s']),
        'single_support_time_left': _phase_stats(ss_l),
        'single_support_time_right': _phase_stats(ss_r),
        'mid_stance_time_left': _phase_stats(mid_l),
        'mid_stance_time_right': _phase_stats(mid_r),
        'terminal_stance_time_left': _phase_stats(term_l),
        'terminal_stance_time_right': _phase_stats(term_r),
        'double_support_time_left': _phase_stats(ds_l),
        'double_support_time_right': _phase_stats(ds_r),
        'loading_response_time_left': _phase_stats(lr_l),
        'loading_response_time_right': _phase_stats(lr_r),
        'pre_swing_time_left': _phase_stats(ps_l),
        'pre_swing_time_right': _phase_stats(ps_r),
        'swing_phase_left': _phase_stats(sp_l, 0, 100),
        'swing_phase_right': _phase_stats(sp_r, 0, 100),
        
        # Additional phase percentages needed for visualization
        'stance_phase_left': _phase_stats(100 - sp_l if len(sp_l) > 0 else np.array([]), 0, 100),
        'stance_phase_right': _phase_stats(100 - sp_r if len(sp_r) > 0 else np.array([]), 0, 100),
        'single_support_left': _phase_stats(_safe_divide_arrays(ss_l, st_l) * 100, 0, 100),
        'single_support_right': _phase_stats(_safe_divide_arrays(ss_r, st_r) * 100, 0, 100),
        'mid_stance_left': _phase_stats(_safe_divide_arrays(mid_l, st_l) * 100, 0, 100),
        'mid_stance_right': _phase_stats(_safe_divide_arrays(mid_r, st_r) * 100, 0, 100),
        'terminal_stance_left': _phase_stats(_safe_divide_arrays(term_l, st_l) * 100, 0, 100),
        'terminal_stance_right': _phase_stats(_safe_divide_arrays(term_r, st_r) * 100, 0, 100),
        'double_support_left': _phase_stats(_safe_divide_arrays(ds_l, st_l) * 100, 0, 100),
        'double_support_right': _phase_stats(_safe_divide_arrays(ds_r, st_r) * 100, 0, 100),
        'loading_response_left': _phase_stats(_safe_divide_arrays(lr_l, st_l) * 100, 0, 100),
        'loading_response_right': _phase_stats(_safe_divide_arrays(lr_r, st_r) * 100, 0, 100),
        'pre_swing_left': _phase_stats(_safe_divide_arrays(ps_l, st_l) * 100, 0, 100),
        'pre_swing_right': _phase_stats(_safe_divide_arrays(ps_r, st_r) * 100, 0, 100),
    }

    if debug:
        print("\n[DEBUG] Temporal Parameters (robust):")
        for k, v in params.items():
            if np.isnan(v[0]):
                print(f"  {k:<30}: nan")
            else:
                print(f"  {k:<30}: {v[0]:.3f} ± {v[1]:.3f}")

    result.setdefault('summary', {}).update(params)