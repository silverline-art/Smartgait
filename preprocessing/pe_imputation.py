import os
import numpy as np
import pandas as pd
import logging
from utils.signal_processing import smooth_with_savgol, remove_outliers_zscore, interpolate_gaps

def clean_pose_keypoints_csv(
    csv_path,
    cleaned_csv_path=None,
    smooth_win=7,
    polyorder=2,
    z_thresh=3.0,
    time_col="time"
):
    """
    Cleans pose keypoints CSV:
      1) Pre-fills NaNs (linear, both directions)
      2) Applies Savitzky–Golay smoothing
      3) Masks outliers by Z-score
      4) Interpolates again and forward/backward fills to fill gaps
    Returns cleaned DataFrame and writes to cleaned_csv_path.
    """
    # Load data
    df = pd.read_csv(csv_path, delimiter='\t')
    df_clean = df.copy()

    # Select numeric columns, excluding time column
    numeric_cols = [
        col for col in df_clean.select_dtypes(include='number').columns
        if col != time_col
    ]
    logging.info(f"Cleaning {len(numeric_cols)} numeric columns...")

    # Ensure odd window and valid polyorder
    if smooth_win % 2 == 0:
        smooth_win += 1
    polyorder = min(polyorder, smooth_win - 1)

    for col in numeric_cols:
        # Extract values
        vals = df_clean[col].values

        # 1) Pre-fill NaNs
        vals = pd.Series(vals).interpolate(method='linear', limit_direction='both').values

        # 2) Smooth
        try:
            vals = smooth_with_savgol(vals, smooth_win, polyorder)
        except Exception as e:
            logging.warning(f"Smoothing failed on {col}: {e}")

        # 3) Outlier masking
        vals, _ = remove_outliers_zscore(vals, z_thresh)

        # 4) Final interpolation and fill
        vals = interpolate_gaps(vals, method='linear', max_gap=10)
        df_clean[col] = vals

    # Save output
    if not cleaned_csv_path:
        cleaned_csv_path = csv_path.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(cleaned_csv_path, sep='\t', index=False)

    return df_clean


# Example usage:
# clean_pose_keypoints_csv('/path/to/your/pose_keypoints.csv')