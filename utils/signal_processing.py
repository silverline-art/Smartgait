"""Signal processing utilities for SmartGait analysis."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple, Optional, List


def smooth_with_savgol(data: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing filter to data.
    
    Args:
        data: Input data array
        window: Window length (must be odd)
        poly: Polynomial order
        
    Returns:
        Smoothed data array
    """
    if len(data) < window:
        return data
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Handle edge cases
    if window >= len(data):
        window = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window < 3:
            return data
    
    try:
        return savgol_filter(data, window, poly)
    except:
        return data


def remove_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers using Z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    if len(data) == 0:
        return data, np.array([], dtype=bool)
    
    mu = np.mean(data)
    sigma = np.std(data)
    
    if sigma == 0:
        return data, np.zeros(len(data), dtype=bool)
    
    z_scores = np.abs((data - mu) / sigma)
    outlier_mask = z_scores > threshold
    
    cleaned_data = data.copy()
    cleaned_data[outlier_mask] = mu
    
    return cleaned_data, outlier_mask


def interpolate_gaps(data: np.ndarray, method: str = 'linear', max_gap: int = 10) -> np.ndarray:
    """Interpolate missing values (NaN) in data.
    
    Args:
        data: Input data array with potential NaN values
        method: Interpolation method ('linear', 'nearest', 'cubic')
        max_gap: Maximum gap size to interpolate
        
    Returns:
        Data with interpolated values
    """
    if not np.any(np.isnan(data)):
        return data
    
    df = pd.DataFrame({'value': data})
    
    # Find gaps
    nan_mask = np.isnan(data)
    gap_starts = np.where(~nan_mask[:-1] & nan_mask[1:])[0] + 1
    gap_ends = np.where(nan_mask[:-1] & ~nan_mask[1:])[0] + 1
    
    # Handle edge cases
    if len(gap_starts) == 0 and len(gap_ends) == 0:
        if np.all(nan_mask):
            return np.zeros_like(data)
        return data
    
    if len(gap_starts) > len(gap_ends):
        gap_ends = np.append(gap_ends, len(data))
    elif len(gap_ends) > len(gap_starts):
        gap_starts = np.insert(gap_starts, 0, 0)
    
    # Only interpolate small gaps
    result = data.copy()
    for start, end in zip(gap_starts, gap_ends):
        gap_size = end - start
        if gap_size <= max_gap:
            # Use pandas interpolation
            temp_series = pd.Series(result)
            temp_series.iloc[start:end] = temp_series.iloc[start:end].interpolate(method=method)
            result[start:end] = temp_series.iloc[start:end].values
    
    return result


def find_contact_periods(pressure_data: np.ndarray, 
                        noise_threshold: float,
                        min_duration_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Find contact periods in pressure data.
    
    Args:
        pressure_data: Array of pressure values
        noise_threshold: Minimum pressure threshold
        min_duration_samples: Minimum duration in samples
        
    Returns:
        Tuple of (contact_starts, contact_ends) indices
    """
    if len(pressure_data) == 0:
        return np.array([]), np.array([])
    
    # Find periods above threshold
    above_threshold = pressure_data > noise_threshold
    
    if not np.any(above_threshold):
        return np.array([]), np.array([])
    
    # Find transitions
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(pressure_data))
    
    # Filter by minimum duration
    if len(starts) > 0 and len(ends) > 0:
        durations = ends - starts
        valid_mask = durations >= min_duration_samples
        starts = starts[valid_mask]
        ends = ends[valid_mask]
    
    return starts, ends


def merge_close_periods(starts: np.ndarray, ends: np.ndarray, 
                       time_array: np.ndarray, max_gap_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """Merge contact periods that are close together.
    
    Args:
        starts: Array of start indices
        ends: Array of end indices
        time_array: Time values array
        max_gap_ms: Maximum gap in milliseconds to merge
        
    Returns:
        Tuple of (merged_starts, merged_ends)
    """
    if len(starts) < 2:
        return starts, ends
    
    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    
    for i in range(1, len(starts)):
        gap_ms = time_array[starts[i]] - time_array[merged_ends[-1]]
        if gap_ms < max_gap_ms:
            # Merge with previous period
            merged_ends[-1] = ends[i]
        else:
            # Start new period
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])
    
    return np.array(merged_starts), np.array(merged_ends)


def detect_peaks_adaptive(data: np.ndarray, 
                         height_percentile: float = 75,
                         distance_samples: int = 10) -> np.ndarray:
    """Detect peaks in data using adaptive threshold.
    
    Args:
        data: Input signal
        height_percentile: Percentile for height threshold
        distance_samples: Minimum distance between peaks
        
    Returns:
        Array of peak indices
    """
    if len(data) < 3:
        return np.array([])
    
    height_threshold = np.percentile(data, height_percentile)
    peaks, _ = find_peaks(data, height=height_threshold, distance=distance_samples)
    
    return peaks