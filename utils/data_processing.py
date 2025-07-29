"""Shared data processing utilities for SmartGait analysis."""

import numpy as np
import pandas as pd
from typing import Optional, List, Union


def get_region_average(df: pd.DataFrame, region_cols: List[str]) -> np.ndarray:
    """Get average values for specified columns in DataFrame.
    
    Args:
        df: Input DataFrame
        region_cols: List of column names to average
        
    Returns:
        Array of averaged values, zeros if no columns exist
    """
    available_cols = [col for col in region_cols if col in df.columns]
    if not available_cols:
        return np.zeros(len(df))
    return df[available_cols].mean(axis=1).values


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Validate that required columns exist in DataFrame.
    
    Args:
        df: Input DataFrame
        required_cols: List of required column names
        
    Returns:
        List of missing column names
    """
    return [col for col in required_cols if col not in df.columns]


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect time column in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of time column if found, None otherwise
    """
    time_candidates = [col for col in df.columns 
                      if col.lower().startswith(('time', 'timestamp', 't_'))]
    return time_candidates[0] if time_candidates else None


def calculate_sampling_rate(time_array: np.ndarray) -> float:
    """Calculate sampling rate from time array.
    
    Args:
        time_array: Array of time values
        
    Returns:
        Sampling rate in Hz
    """
    if len(time_array) < 2:
        return 1000.0  # Default fallback
    median_diff = np.median(np.diff(time_array))
    return 1000.0 / median_diff if median_diff > 0 else 1000.0


def find_header_line(file_path: str) -> int:
    """Find the header line in CSV file by looking for column headers.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Line number where header starts (0-indexed)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    # Look for lines with multiple comma-separated values and likely column names
                    parts = line.strip().split(',')
                    if len(parts) > 5 and any(part.strip().isalpha() or '_' in part for part in parts):
                        return i
                    # Also check if this looks like a timestamp column header
                    if 'timestamp' in line.lower() or 'time' in line.lower():
                        return i
    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            parts = line.strip().split(',')
                            if len(parts) > 5 and any(part.strip().isalpha() or '_' in part for part in parts):
                                return i
                            if 'timestamp' in line.lower() or 'time' in line.lower():
                                return i
                break
            except UnicodeDecodeError:
                continue
    return 0


def read_input_csv(file_path: str) -> pd.DataFrame:
    """Read CSV file with automatic header detection.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    skiprows = find_header_line(file_path)
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, skiprows=skiprows, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]:  # Last encoding attempt
                raise ValueError(f"Failed to read CSV file {file_path}: {str(e)}")
            continue
    
    raise ValueError(f"Failed to read CSV file {file_path}: Unable to decode with any encoding")


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray], 
               default: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide with handling for zero division.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value for zero division
        
    Returns:
        Division result or default value
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    else:
        return numerator / denominator if denominator != 0 else default