import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
try:
    from scipy.signal import cwt, ricker
except ImportError:
    # For newer scipy versions, cwt is in scipy.signal.wavelets
    try:
        from scipy.signal.wavelets import cwt, ricker
    except ImportError:
        # Fallback - define simple versions or skip wavelet functionality
        def cwt(data, wavelet, widths):
            return np.zeros((len(widths), len(data)))
        def ricker(points, a):
            return np.zeros(points)
import pandas as pd
import warnings

from constants.pe import (
    ANGLE_JOINTS,
    LANDMARK_NAMES,
    FOOT_COLS,
    TIME_COL,
    DEFAULT_PROMINENCE,
    DEFAULT_FRAME_RATE,
    DEFAULT_SMOOTH_WIN,
    DEFAULT_POLYORDER,
    DEFAULT_Z_THRESH,
    DEFAULT_VIS_THRESH
)

class GaitCycleAnalyzer:
    """
    Analyze gait cycles and compute spatiotemporal parameters from foot trajectory data.
    Applies visibility thresholding, robust outlier removal, interpolation, and smoothing.
    """

    def __init__(
        self,
        df,
        side="left",
        time_col=TIME_COL,
        pixel_to_meter=1.0,
        fps=DEFAULT_FRAME_RATE,
        debug=False,
        smoothing_method="savgol",  # "savgol", "butter", "ema"
        smoothing_kwargs=None,
        outlier_method="zscore",    # "zscore", "mad", "iqr"
        outlier_kwargs=None,
        interpolation_method="linear",  # "linear", "polynomial", "spline"
        peak_method="find_peaks",   # "find_peaks", "wavelet"
        peak_kwargs=None,
        support_phase_method="default",  # "default", "velocity"
    ):
        if time_col not in df.columns:
            df[time_col] = df.index / fps

        # Frame rate consistency check
        time_diffs = np.diff(df[time_col])
        if np.std(time_diffs) > (1.0 / fps) * 0.05:
            warnings.warn("Detected non-uniform frame intervals in time_col!")

        missing_cols = []
        for lm in LANDMARK_NAMES:
            for suffix in ["_x", "_y", "_vis"]:
                col = f"{lm}_{suffix[1:]}" if suffix != "_vis" else f"{lm}_vis"
                if col not in df.columns:
                    missing_cols.append(col)
        if missing_cols and debug:
            print(f"Warning: Missing landmark columns: {missing_cols}")

        foot_y_col = FOOT_COLS[side]["y"]
        # Robustly get the visibility column
        if "vis" in FOOT_COLS[side]:
            foot_vis_col = FOOT_COLS[side]["vis"]
        else:
            foot_vis_col = foot_y_col.replace("_y", "_vis")
        if foot_y_col not in df.columns or foot_vis_col not in df.columns:
            raise ValueError(f"Required columns '{foot_y_col}' and/or '{foot_vis_col}' not found in DataFrame. "
                             f"Available columns: {df.columns.tolist()}")

        # 1. Start with the raw y trajectory and visibility
        foot_y = df[foot_y_col].values.astype(float)
        foot_vis = df[foot_vis_col].values.astype(float)

        # 2. Mask low-visibility points
        low_vis_mask = foot_vis < DEFAULT_VIS_THRESH
        if debug:
            print(f"Visibility thresholding: {np.sum(low_vis_mask)} points below threshold {DEFAULT_VIS_THRESH}")
        foot_y[low_vis_mask] = np.nan

        # 3. Outlier removal (robust)
        foot_y = self.remove_outliers(
            foot_y,
            method=outlier_method,
            kwargs=outlier_kwargs or {},
            debug=debug
        )

        # 4. Interpolate NaNs (using pandas)
        foot_y = self.interpolate_nans(
            foot_y,
            method=interpolation_method,
            max_gap=10,
            debug=debug
        )

        # 5. Smoothing (multiple options)
        foot_y = self.smooth_signal(
            foot_y,
            method=smoothing_method,
            fps=fps,
            kwargs=smoothing_kwargs or {},
            debug=debug
        )

        if debug:
            print(f"Using foot marker column: {foot_y_col}")
            print(f"First few values of {foot_y_col}: {foot_y[:5].tolist()}")
            print(f"Describe {foot_y_col}:")
            print(f"count: {np.sum(~np.isnan(foot_y))}, mean: {np.nanmean(foot_y)}, std: {np.nanstd(foot_y)}, min: {np.nanmin(foot_y)}, max: {np.nanmax(foot_y)}")
            print(f"Any NaNs in {foot_y_col}? {np.isnan(foot_y).sum()}")

        self.foot_y = foot_y
        self.time = df[time_col].values
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps
        self.debug = debug
        self.peak_method = peak_method
        self.peak_kwargs = peak_kwargs or {}
        self.support_phase_method = support_phase_method

    @staticmethod
    def remove_outliers(arr, method="zscore", kwargs=None, debug=False):
        arr = arr.copy()
        kwargs = kwargs or {}
        if method == "zscore":
            thresh = kwargs.get("z_thresh", DEFAULT_Z_THRESH)
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            z_scores = (arr - mean) / (std if std > 0 else 1)
            outliers = np.abs(z_scores) > thresh
            if debug:
                print(f"Z-score outlier removal: {np.sum(outliers)} outliers detected with threshold {thresh}")
            arr[outliers] = np.nan
        elif method == "mad":
            thresh = kwargs.get("mad_thresh", 3.5)
            median = np.nanmedian(arr)
            mad = np.nanmedian(np.abs(arr - median))
            mod_z = 0.6745 * (arr - median) / (mad if mad > 0 else 1)
            outliers = np.abs(mod_z) > thresh
            if debug:
                print(f"MAD outlier removal: {np.sum(outliers)} outliers detected with threshold {thresh}")
            arr[outliers] = np.nan
        elif method == "iqr":
            factor = kwargs.get("iqr_factor", 1.5)
            q1, q3 = np.nanpercentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outliers = (arr < lower) | (arr > upper)
            if debug:
                print(f"IQR outlier removal: {np.sum(outliers)} outliers outside [{lower}, {upper}]")
            arr[outliers] = np.nan
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
        return arr

    @staticmethod
    def interpolate_nans(arr, method="linear", max_gap=10, debug=False):
        s = pd.Series(arr)
        # Find NaN runs
        is_nan = s.isna()
        nan_groups = (is_nan != is_nan.shift()).cumsum()[is_nan]
        group_sizes = nan_groups.value_counts()
        # Only interpolate gaps <= max_gap
        for group, size in group_sizes.items():
            if size > max_gap:
                idxs = nan_groups[nan_groups == group].index
                s.iloc[idxs] = np.nan  # Ensure large gaps remain NaN
        s_interp = s.interpolate(method=method, limit=max_gap, limit_direction="both")
        if debug:
            print(f"Interpolation ({method}): {s.isna().sum()} NaNs before, {s_interp.isna().sum()} after")
        return s_interp.values

    @staticmethod
    def smooth_signal(arr, method="savgol", fps=30, kwargs=None, debug=False):
        arr = arr.copy()
        kwargs = kwargs or {}
        if method == "savgol":
            win = kwargs.get("window_length", DEFAULT_SMOOTH_WIN)
            poly = kwargs.get("polyorder", DEFAULT_POLYORDER)
            if len(arr) >= win and win % 2 == 1:
                try:
                    arr = savgol_filter(arr, window_length=win, polyorder=poly)
                    if debug:
                        print(f"Savitzky-Golay smoothing applied: window={win}, polyorder={poly}")
                except Exception as e:
                    if debug:
                        print(f"Could not apply Savitzky-Golay filter: {e}. Falling back to rolling mean.")
                    arr = pd.Series(arr).rolling(win, min_periods=1, center=True).mean().values
        elif method == "butter":
            order = kwargs.get("order", 2)
            cutoff = kwargs.get("cutoff", 3)  # Hz
            nyq = 0.5 * fps
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            arr = filtfilt(b, a, arr)
            if debug:
                print(f"Butterworth smoothing applied: order={order}, cutoff={cutoff}Hz")
        elif method == "ema":
            span = kwargs.get("span", 10)
            arr = pd.Series(arr).ewm(span=span, min_periods=1, adjust=False).mean().values
            if debug:
                print(f"EMA smoothing applied: span={span}")
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        return arr

    def detect_foot_strikes(self, prominence=DEFAULT_PROMINENCE):
        if self.peak_method == "find_peaks":
            inverted = -self.foot_y
            peaks, _ = find_peaks(inverted, prominence=prominence, **self.peak_kwargs)
            return peaks, self.time[peaks]
        elif self.peak_method == "wavelet":
            widths = self.peak_kwargs.get("widths", np.arange(1, 10))
            cwt_mat = cwt(-self.foot_y, ricker, widths)
            peak_idxs = np.argwhere((np.abs(cwt_mat).max(axis=0) > prominence)).flatten()
            return peak_idxs, self.time[peak_idxs]
        else:
            raise ValueError(f"Unknown peak detection method: {self.peak_method}")

    def detect_foot_offs(self, prominence=DEFAULT_PROMINENCE):
        if self.peak_method == "find_peaks":
            peaks, _ = find_peaks(self.foot_y, prominence=prominence, **self.peak_kwargs)
            return peaks, self.time[peaks]
        elif self.peak_method == "wavelet":
            widths = self.peak_kwargs.get("widths", np.arange(1, 10))
            cwt_mat = cwt(self.foot_y, ricker, widths)
            peak_idxs = np.argwhere((np.abs(cwt_mat).max(axis=0) > prominence)).flatten()
            return peak_idxs, self.time[peak_idxs]
        else:
            raise ValueError(f"Unknown peak detection method: {self.peak_method}")

    def compute_gait_cycles(self, foot_strike_indices):
        cycles = []
        for i in range(len(foot_strike_indices) - 1):
            cycles.append((foot_strike_indices[i], foot_strike_indices[i+1]))
        return cycles

    def compute_stride_lengths(self, foot_strike_indices):
        stride_lengths = []
        for i in range(len(foot_strike_indices) - 1):
            idx1, idx2 = foot_strike_indices[i], foot_strike_indices[i+1]
            displacement = np.abs(self.foot_y[idx2] - self.foot_y[idx1])
            stride_lengths.append(displacement * self.pixel_to_meter)
        return np.array(stride_lengths)

    def compute_step_lengths(self, foot_strike_indices, other_foot_strike_indices=None, max_gap_frames=30):
        if other_foot_strike_indices is None or len(other_foot_strike_indices) == 0:
            stride_lengths = self.compute_stride_lengths(foot_strike_indices)
            return stride_lengths / 2 if len(stride_lengths) > 0 else np.array([])
        step_lengths = []
        for idx in foot_strike_indices:
            diffs = np.array(other_foot_strike_indices) - idx
            diffs = diffs[diffs < 0]
            if len(diffs) > 0:
                nearest_idx = other_foot_strike_indices[np.argmax(diffs)]
                if abs(idx - nearest_idx) <= max_gap_frames:
                    displacement = np.abs(self.foot_y[idx] - self.foot_y[nearest_idx])
                    step_lengths.append(displacement * self.pixel_to_meter)
        return np.array(step_lengths)

    def compute_support_phases(self, foot_strike_indices, foot_off_indices):
        if self.support_phase_method == "velocity":
            # Use velocity threshold to estimate stance/swing
            vel = np.gradient(self.foot_y, self.time)
            threshold = np.percentile(np.abs(vel), 20)  # 20th percentile as threshold
            stance_mask = np.abs(vel) < threshold
            stance_durations = []
            swing_durations = []
            in_stance = False
            start_idx = 0
            for i, val in enumerate(stance_mask):
                if val and not in_stance:
                    in_stance = True
                    start_idx = i
                elif not val and in_stance:
                    in_stance = False
                    stance_durations.append(self.time[i-1] - self.time[start_idx])
            # Swing is the rest
            swing_durations = np.diff(self.time)[~stance_mask[:-1]]
            return np.array(stance_durations), np.array(swing_durations)
        else:
            stance_durations = []
            swing_durations = []
            try:
                for i in range(len(foot_strike_indices) - 1):
                    fs = foot_strike_indices[i]
                    next_fs = foot_strike_indices[i + 1]
                    foot_offs_in_cycle = foot_off_indices[(foot_off_indices > fs) & (foot_off_indices < next_fs)]
                    if len(foot_offs_in_cycle) > 0:
                        fo = foot_offs_in_cycle[0]
                        stance = self.time[fo] - self.time[fs]
                        swing = self.time[next_fs] - self.time[fo]
                        stance_durations.append(stance)
                        swing_durations.append(swing)
            except Exception as e:
                print(f"Support phase computation failed: {e}")
                return np.array([]), np.array([])
            return np.array(stance_durations), np.array(swing_durations)

    def compute_speed_and_cadence(self, stride_lengths, foot_strike_times, steps_per_stride=2):
        stride_times = np.diff(foot_strike_times)
        mean_stride_time = np.mean(stride_times) if len(stride_times) > 0 else np.nan
        mean_stride_length = np.mean(stride_lengths) if len(stride_lengths) > 0 else np.nan
        mean_speed = mean_stride_length / mean_stride_time if mean_stride_time > 0 else np.nan
        cadence = (60.0 * steps_per_stride) / mean_stride_time if mean_stride_time > 0 else np.nan
        # Per-cycle speed/cadence
        per_cycle_speed = stride_lengths / stride_times if len(stride_times) == len(stride_lengths) and len(stride_times) > 0 else np.array([])
        per_cycle_cadence = (60.0 * steps_per_stride) / stride_times if len(stride_times) > 0 else np.array([])
        return mean_speed, cadence, per_cycle_speed, per_cycle_cadence

    @staticmethod
    def compute_support_phase_per_frame(n_frames, left_strikes, left_offs, right_strikes, right_offs):
        phase = np.zeros(n_frames, dtype=int)
        left_on_ground = np.zeros(n_frames, dtype=bool)
        right_on_ground = np.zeros(n_frames, dtype=bool)

        def mark_stance(strikes, offs, on_ground):
            s_idx, o_idx = 0, 0
            while s_idx < len(strikes) and o_idx < len(offs):
                fs = strikes[s_idx]
                while o_idx < len(offs) and offs[o_idx] <= fs:
                    o_idx += 1
                if o_idx < len(offs):
                    fo = offs[o_idx]
                    if fo > fs:
                        on_ground[fs:fo] = True
                s_idx += 1

        mark_stance(left_strikes, left_offs, left_on_ground)
        mark_stance(right_strikes, right_offs, right_on_ground)

        for i in range(n_frames):
            if left_on_ground[i] and right_on_ground[i]:
                phase[i] = 3  # double support
            elif left_on_ground[i]:
                phase[i] = 1  # single support left
            elif right_on_ground[i]:
                phase[i] = 2  # single support right
            else:
                phase[i] = 0  # swing (neither foot on ground)
        return phase

    def analyze(self, prominence=DEFAULT_PROMINENCE):
        fs_idx, fs_times = self.detect_foot_strikes(prominence=prominence)
        fo_idx, fo_times = self.detect_foot_offs(prominence=prominence)
        cycles = self.compute_gait_cycles(fs_idx)
        stride_lengths = self.compute_stride_lengths(fs_idx)
        step_lengths = self.compute_step_lengths(fs_idx)
        mean_speed, cadence, per_cycle_speed, per_cycle_cadence = self.compute_speed_and_cadence(stride_lengths, fs_times)
        stance_durations, swing_durations = self.compute_support_phases(fs_idx, fo_idx)
        return {
            "foot_strike_indices": fs_idx,
            "foot_strike_times": fs_times,
            "foot_off_indices": fo_idx,
            "foot_off_times": fo_times,
            "gait_cycles": cycles,
            "stride_lengths": stride_lengths,
            "step_lengths": step_lengths,
            "mean_speed": mean_speed,
            "cadence": cadence,
            "per_cycle_speed": per_cycle_speed,
            "per_cycle_cadence": per_cycle_cadence,
            "stance_durations": stance_durations,
            "swing_durations": swing_durations
        }

    def plot_gait_cycles(self, foot_strike_indices, foot_off_indices):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(self.foot_y, label='Foot Y')
        plt.scatter(foot_strike_indices, self.foot_y[foot_strike_indices], color='blue', label='Strikes')
        plt.scatter(foot_off_indices, self.foot_y[foot_off_indices], color='red', label='Offs')
        plt.legend()
        plt.title("QC: Foot Y with Strikes and Offs")
        plt.show()

    def to_dict(self, results):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}

    def compute_symmetry(self, left_vals, right_vals):
        left_mean = np.nanmean(left_vals)
        right_mean = np.nanmean(right_vals)
        return 100 * abs(left_mean - right_mean) / ((left_mean + right_mean) / 2) if (left_mean + right_mean) != 0 else np.nan

def print_gait_stats(
    df,
    side="left",
    time_col=TIME_COL,
    pixel_to_meter=1.0,
    fps=DEFAULT_FRAME_RATE,
    prominence=DEFAULT_PROMINENCE,
    debug=False,
    **analyzer_kwargs
):
    analyzer = GaitCycleAnalyzer(
        df,
        side=side,
        time_col=time_col,
        pixel_to_meter=pixel_to_meter,
        fps=fps,
        debug=debug,
        **analyzer_kwargs
    )
    results = analyzer.analyze(prominence=prominence)

    left_count = right_count = None
    if "left_foot_index_y" in df.columns and "right_foot_index_y" in df.columns:
        left_analyzer = GaitCycleAnalyzer(df, side="left", time_col=time_col, pixel_to_meter=pixel_to_meter, fps=fps, **analyzer_kwargs)
        right_analyzer = GaitCycleAnalyzer(df, side="right", time_col=time_col, pixel_to_meter=pixel_to_meter, fps=fps, **analyzer_kwargs)
        left_count = len(left_analyzer.detect_foot_strikes(prominence=prominence)[0])
        right_count = len(right_analyzer.detect_foot_strikes(prominence=prominence)[0])

    stride_lengths = results['stride_lengths'] * 100  # convert to cm
    step_lengths = results['step_lengths'] * 100      # convert to cm
    stride_mean = np.nanmean(stride_lengths) if len(stride_lengths) > 0 else np.nan
    stride_std = np.nanstd(stride_lengths) if len(stride_lengths) > 0 else np.nan
    step_mean = np.nanmean(step_lengths) if len(step_lengths) > 0 else np.nan
    step_std = np.nanstd(step_lengths) if len(step_lengths) > 0 else np.nan
    mean_speed_cm = results['mean_speed'] * 100 if results['mean_speed'] is not None else np.nan

    print("Gait Statistics")
    if left_count is not None and right_count is not None:
        print(f"  Left foot strike count: {left_count}")
        print(f"  Right foot strike count: {right_count}")
    print(f"  Gait Cycles: {len(results['gait_cycles'])}")
    print(f"  Avg Gait Velocity: {mean_speed_cm:.3f} cm/s")
    print(f"  Avg Stride Length: {stride_mean:.3f} ± {stride_std:.3f} cm")
    print(f"  Avg Step Length: {step_mean:.3f} ± {step_std:.3f} cm")
    print(f"  Cadence (steps/min): {results['cadence']:.2f}")
    if "per_cycle_speed" in results and len(results["per_cycle_speed"]) > 0:
        print(f"  Per-cycle speed (cm/s): {results['per_cycle_speed']*100}")
    if "per_cycle_cadence" in results and len(results["per_cycle_cadence"]) > 0:
        print(f"  Per-cycle cadence (steps/min): {results['per_cycle_cadence']}")
    print(f"  Mean Stance Duration (s): {np.nanmean(results['stance_durations']):.3f}" if len(
        results['stance_durations']) > 0 else "  Mean Stance Duration (s): N/A")
    print(f"  Mean Swing Duration (s): {np.nanmean(results['swing_durations']):.3f}" if len(
        results['swing_durations']) > 0 else "  Mean Swing Duration (s): N/A")