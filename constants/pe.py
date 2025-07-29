import os

# -------------------------------
# 0. Defaults & Constants
# -------------------------------
DEFAULT_FRAME_RATE = 30
DEFAULT_SMOOTH_WIN = 5  # must be odd
DEFAULT_POLYORDER = 5
DEFAULT_Z_THRESH = 3.0
DEFAULT_PROMINENCE = 0.001
DEFAULT_VIS_THRESH = 0.5
TIME_COL = "time"

# Updated defaults for batch processing

# -------------------------------
# 1. Utility Functions
# -------------------------------
def find_relevant_csvs(input_dir):
    """
    Recursively find all CSV files containing 'raw_pose_fix_cleaned' in their name.
    Returns a list of file paths.
    """
    relevant_csvs = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and 'raw_pose_fix_cleaned' in file:
                relevant_csvs.append(os.path.join(root, file))
    return relevant_csvs

# -------------------------------
# 2. Landmark and Joint Definitions
# -------------------------------
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

FOOT_COLS = {
    "left": {"x": "left_foot_index_x", "y": "left_foot_index_y"},
    "right": {"x": "right_foot_index_x", "y": "right_foot_index_y"}
}

ANGLE_JOINTS = {
    "hip_left": ("left_shoulder", "left_hip", "left_knee"),
    "hip_right": ("right_shoulder", "right_hip", "right_knee"),
    "knee_left": ("left_hip", "left_knee", "left_ankle"),
    "knee_right": ("right_hip", "right_knee", "right_ankle"),
    "ankle_left": ("left_knee", "left_ankle", "left_foot_index"),
    "ankle_right": ("right_knee", "right_ankle", "right_foot_index"),
    "shoulder_left": ("left_hip", "left_shoulder", "left_elbow"),
    "shoulder_right": ("right_hip", "right_shoulder", "right_elbow")
}