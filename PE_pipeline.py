import os
import sys
import pandas as pd
from config import file_path, output_path
from utils.visualization_utils import create_output_directory

# PE-specific imports
from pe.pose_estimation_task import run_pose_estimation
from preprocessing.pe_imputation import clean_pose_keypoints_csv
from pe.vis_pe import label_video_from_csv
from analysis.pe.ROM import print_rom_stats
from analysis.pe import gait_detect
from visualization.pe_summary import save_summary
from visualization.vispe_ROM import plot_rom_from_dataframe
from visualization.vispe_gait import visualize_gait_analysis

# Define paths
raw_csv_path = os.path.join(output_path, "keypoints.csv")
cleaned_csv_path = os.path.join(output_path, "keypoints_cleaned.csv")

def main():
    """Main Pose Estimation pipeline execution."""
    try:
        print("[INFO] Starting Pose Estimation Pipeline...")
        
        # Create output directory
        create_output_directory(output_path)
        
        # Step 1: Run pose estimation on video
        print("[INFO] Step 1: Running pose estimation on video...")
        if not os.path.exists(raw_csv_path):
            print(f"[INFO] Running pose estimation on: {file_path}")
            run_pose_estimation(file_path, output_path)
        else:
            print(f"[INFO] Using existing keypoints file: {raw_csv_path}")
        
        # Check if raw CSV exists
        if not os.path.exists(raw_csv_path):
            print(f"[ERROR] Raw keypoints file not found: {raw_csv_path}")
            print("[ERROR] Pose estimation may have failed. Please check the video file.")
            sys.exit(1)
        
        # Step 2: Clean keypoints data
        print("[INFO] Step 2: Cleaning keypoints data...")
        clean_pose_keypoints_csv(raw_csv_path)
        
        if not os.path.exists(cleaned_csv_path):
            print(f"[ERROR] Cleaned keypoints file not found: {cleaned_csv_path}")
            sys.exit(1)
        
        # Step 3: Load cleaned data
        print("[INFO] Step 3: Loading cleaned keypoints data...")
        # Try different delimiters
        try:
            df = pd.read_csv(cleaned_csv_path, delimiter='\t')
            # Check if columns were parsed correctly
            if len(df.columns) == 1:  # All columns in one string
                df = pd.read_csv(cleaned_csv_path, delimiter=',')
        except:
            df = pd.read_csv(cleaned_csv_path, delimiter=',')
        print(f"[INFO] Loaded {len(df)} frames of keypoint data")
        print(f"[INFO] Data shape: {df.shape}")
        
        # Check data quality
        nan_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        print(f"[INFO] Data quality: {nan_percentage:.1f}% NaN values")
        
        if nan_percentage > 90:
            print("[WARNING] High percentage of NaN values detected. Results may be unreliable.")
            print("[WARNING] This often indicates pose estimation issues with the video.")
        
        # Step 4: Range of Motion Analysis
        print("[INFO] Step 4: Calculating Range of Motion...")
        print_rom_stats(df)
        plot_rom_from_dataframe(df)
        
        # Step 5: Gait Detection and Analysis
        print("[INFO] Step 5: Detecting gait events...")
        try:
            gait_detect.print_gait_stats(df)
        except Exception as e:
            print(f"[WARNING] Gait detection failed: {e}")
            print("[WARNING] Skipping gait analysis due to data issues.")
        
        # Step 6: Generate visualizations
        print("[INFO] Step 6: Generating gait visualizations...")
        visualize_gait_analysis(df)
        
        # Step 7: Create labeled video (optional)
        print("[INFO] Step 7: Creating labeled video...")
        try:
            label_video_from_csv(file_path, cleaned_csv_path, output_path)
        except Exception as e:
            print(f"[WARNING] Video labeling failed: {e}")
        
        # Step 8: Save summary
        print("[INFO] Step 8: Saving analysis summary...")
        save_summary(output_path)
        
        print("[SUCCESS] Pose Estimation pipeline completed successfully!")
        print(f"[INFO] Results saved to: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Pose Estimation pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()