"""Mathematical utilities for SmartGait analysis."""

import numpy as np
from typing import Tuple, Union


def compute_angle_3points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle at point b formed by points a-b-c.
    
    Args:
        a: First point coordinates [x, y]
        b: Middle point coordinates [x, y] 
        c: Third point coordinates [x, y]
        
    Returns:
        Angle in degrees
    """
    ba = a - b
    bc = c - b
    
    # Handle zero vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def euclidean_distance(x1: float, y1: float, x2: float, y2: float, 
                      scale: float = 1.0) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates  
        scale: Scaling factor for distance
        
    Returns:
        Scaled Euclidean distance
    """
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance * scale


def calculate_symmetry_index(left_val: float, right_val: float) -> float:
    """Calculate symmetry index between left and right values.
    
    Args:
        left_val: Left side value
        right_val: Right side value
        
    Returns:
        Symmetry index as percentage (0 = perfect symmetry)
    """
    if left_val == 0 and right_val == 0:
        return 0.0
    
    mean_val = (left_val + right_val) / 2
    if mean_val == 0:
        return 100.0  # Maximum asymmetry
    
    symmetry = abs(left_val - right_val) / mean_val * 100
    return symmetry


def calculate_cadence(step_count: int, duration_sec: float) -> float:
    """Calculate cadence (steps per minute).
    
    Args:
        step_count: Number of steps
        duration_sec: Duration in seconds
        
    Returns:
        Cadence in steps per minute
    """
    if duration_sec <= 0:
        return 0.0
    
    return (step_count / duration_sec) * 60


def calculate_velocity(distance: float, time: float) -> float:
    """Calculate velocity from distance and time.
    
    Args:
        distance: Distance traveled
        time: Time taken
        
    Returns:
        Velocity (distance/time)
    """
    if time <= 0:
        return 0.0
    return distance / time


def calculate_range_of_motion(angles: np.ndarray) -> Tuple[float, float, float]:
    """Calculate range of motion statistics from angle array.
    
    Args:
        angles: Array of joint angles
        
    Returns:
        Tuple of (min_angle, max_angle, range_of_motion)
    """
    if len(angles) == 0:
        return 0.0, 0.0, 0.0
    
    min_angle = np.min(angles)
    max_angle = np.max(angles)
    rom = max_angle - min_angle
    
    return min_angle, max_angle, rom


def smooth_angle_sequence(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth angle sequence using moving average.
    
    Args:
        angles: Array of angles in degrees
        window: Window size for smoothing
        
    Returns:
        Smoothed angle array
    """
    if len(angles) < window:
        return angles
    
    # Convert to radians for smoothing to handle angle wrapping
    radians = np.radians(angles)
    
    # Use complex representation to handle angle wrapping
    complex_angles = np.exp(1j * radians)
    
    # Apply moving average
    smoothed_complex = np.convolve(complex_angles, np.ones(window)/window, mode='same')
    
    # Convert back to degrees
    smoothed_radians = np.angle(smoothed_complex)
    smoothed_angles = np.degrees(smoothed_radians)
    
    return smoothed_angles


def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage with safe division.
    
    Args:
        part: Part value
        total: Total value
        
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def normalize_array(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize array using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        Normalized array
    """
    if len(data) == 0:
        return data
    
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max == data_min:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    elif method == 'unit':
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        return data / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate area under curve using trapezoidal rule.
    
    Args:
        x: X coordinates
        y: Y coordinates
        
    Returns:
        Area under curve
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    return np.trapz(y, x)