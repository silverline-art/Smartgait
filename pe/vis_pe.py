import os
import cv2
import csv
import numpy as np
from pe.pose_estimation_task import POSE_KEYPOINTS, draw_keypoints

def get_labeled_video_path(video_path, output_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1]
    labeled_name = f"{base}_labeled{ext}"
    return os.path.join(output_dir, labeled_name)

def label_video_from_csv(video_path, csv_path, output_dir):
    """
    Overlays keypoints from a CSV file onto the video and saves the labeled video in the output directory.
    Args:
        video_path (str): Path to the original video.
        csv_path (str): Path to the cleaned keypoints CSV.
        output_dir (str): Directory to save the labeled video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width, height)

    # Always save labeled video in the output directory
    output_path_full = get_labeled_video_path(video_path, output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path_full, fourcc, fps, output_size)

    # Read keypoints from CSV (handle both comma and tab delimiters)
    keypoints_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # If row has only one element, try splitting by comma
            if len(row) == 1:
                values = row[0].split(',')
            else:
                values = row
            keypoints_list.append([float(x) if x != 'nan' else np.nan for x in values])

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(keypoints_list) != num_frames:
        print(f"Warning: CSV rows ({len(keypoints_list)}) != video frames ({num_frames})")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints_list):
            break

        row = keypoints_list[frame_idx]
        landmarks = []
        visibilities = []
        for i in range(len(POSE_KEYPOINTS)):
            x = row[i*3]
            y = row[i*3+1]
            vis = row[i*3+2]
            landmarks.append((x, y, 0))  # z is not used for drawing
            visibilities.append(vis)
        draw_keypoints(frame, landmarks, visibilities)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Labeled video saved: {output_path_full}")