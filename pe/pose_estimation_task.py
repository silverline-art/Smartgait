import os
import sys

# --- SUPPRESS ALL LOGS AND WARNINGS BEFORE ANY ML IMPORTS ---

# Suppress TensorFlow, absl, and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Optional: disables GPU if not needed
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1' # Optional: disables MediaPipe GPU logs

# Suppress C++ backend logs by redirecting stderr (works for most cases)
class SuppressStdErr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.old_stderr_fd = os.dup(2)
        os.dup2(self.null_fd, 2)
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stderr_fd, 2)
        os.close(self.null_fd)
        os.close(self.old_stderr_fd)

with SuppressStdErr():
    import logging
    import absl.logging
    absl.logging.set_verbosity('error')
    absl.logging.set_stderrthreshold('error')
    logging.getLogger('cv2').setLevel(logging.ERROR)
    import cv2
    import mediapipe as mp
    import csv

mp_pose = mp.solutions.pose
POSE_KEYPOINTS = [lm.name.lower() for lm in mp_pose.PoseLandmark]

def detect_pose_in_video(video_path):
    """
    Detects pose landmarks in each frame of the video.
    Returns a list of frames, each containing a list of (x, y, visibility) for each keypoint.
    """
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        smooth_landmarks=True
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_row = [(lm.x, lm.y, lm.visibility) for lm in landmarks]
                all_keypoints.append(frame_row)
            else:
                all_keypoints.append([(float('nan'), float('nan'), float('nan'))] * len(POSE_KEYPOINTS))

    cap.release()
    return all_keypoints

def run_pose_estimation(input_file, output_dir):
    """
    Runs pose estimation on the input video file and saves the keypoints as a CSV in the output directory.
    """
    keypoints_per_frame = detect_pose_in_video(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    csv_path = os.path.join(output_dir, "keypoints.csv")
    header = []
    for name in POSE_KEYPOINTS:
        header.extend([f'{name}_x', f'{name}_y', f'{name}_vis'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_row in keypoints_per_frame:
            # Flatten the list of tuples for CSV writing
            flat_row = [item for lm in frame_row for item in lm]
            writer.writerow(flat_row)
    print("Pose keypoints saved.")

def draw_keypoints(frame, landmarks, visibilities, color=(0, 255, 0)):
    """
    Draws keypoints on the frame.
    Args:
        frame: The image frame (numpy array).
        landmarks: List of (x, y, z) tuples, normalized coordinates.
        visibilities: List of visibility scores for each keypoint.
        color: Color for keypoints (BGR tuple).
    Returns:
        frame with keypoints drawn.
    """
    h, w = frame.shape[:2]
    for idx, (lm, vis) in enumerate(zip(landmarks, visibilities)):
        # Check for nan and visibility threshold
        if vis > 0.5 and not (lm[0] != lm[0] or lm[1] != lm[1]):
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            name = POSE_KEYPOINTS[idx]
            cv2.putText(frame, name, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame