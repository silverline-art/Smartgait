# SmartGait

A comprehensive gait analysis system with dual pipeline architecture for pose estimation and smart insole sensor data analysis.

## Overview

SmartGait provides advanced biomechanical analysis through two independent processing pipelines:

- **Pose Estimation (PE) Pipeline**: Video-based gait analysis using MediaPipe
- **Smart Equipment (SE) Pipeline**: Sensor-based analysis from smart insoles

## Features

### Pose Estimation Pipeline
- MediaPipe-based pose detection from video
- Joint angle calculations for 8 key joints (hip, knee, ankle, shoulder)
- Range of Motion (ROM) analysis
- Gait cycle detection and analysis
- Comprehensive visualization dashboards

### Smart Insole Pipeline
- 4-sensor insole pressure analysis
- Walking segment detection
- Step detection and temporal parameters
- Gait phase percentage analysis
- Pressure distribution visualization

## Installation

### Prerequisites
- Python 3.7+

### Dependencies
Install required packages:

```bash
pip install -r Github/requirements.txt
```

Required packages:
- mediapipe
- pandas
- numpy
- matplotlib
- scipy
- seaborn

### MediaPipe Model
Download the MediaPipe pose detection model:
- Place `pose_landmarker_heavy.task` in `Github/model/` directory
- Model size: ~30MB

## Quick Start

### Configuration
Edit `Github/config.py` to configure:

```python
SELECT_TASK_INDEX = 0  # 0 for PE, 1 for SE
file_path = "path/to/your/input/file"  # Video for PE, CSV for SE
output_path = "path/to/output/directory"
```

### Running Analysis

#### Main Interface
```bash
cd "Github"
python main.py
```

#### Direct Pipeline Execution
```bash
# Pose Estimation Pipeline
python PE_pipeline.py

# Smart Insole Pipeline  
python SE_pipeline.py

# Interactive CLI (incomplete)
python int.py
```

## Project Structure

```
SmartGait/
├── Github/
│   ├── main.py                 # Main entry point
│   ├── config.py              # Configuration settings
│   ├── PE_pipeline.py         # Pose estimation workflow
│   ├── SE_pipeline.py         # Smart insole workflow
│   ├── requirements.txt       # Python dependencies
│   │
│   ├── analysis/              # Core analysis modules
│   │   ├── pe/               # Pose estimation analysis
│   │   │   ├── ROM.py        # Range of motion calculations
│   │   │   ├── gait_detect.py # Gait cycle detection
│   │   └── si/               # Smart insole analysis
│   │       ├── step_detection.py # Step detection algorithms
│   │       ├── time_parameters.py # Temporal analysis
│   │       ├── phase_percentage.py # Gait phase analysis
│   │       └── initial_pressure.py # Pressure analysis
│   │
│   ├── preprocessing/         # Data preprocessing
│   │   ├── pe_imputation.py  # Pose data cleaning
│   │   └── se_process.py     # Sensor data processing
│   │
│   ├── visualization/         # Plotting and reports
│   │   ├── pe_summary.py     # PE analysis reports
│   │   ├── vispe_*.py        # PE visualizations
│   │   └── visse_*.py        # SE visualizations
│   │
│   ├── constants/             # System constants
│   │   ├── pe.py             # MediaPipe landmarks, joints
│   │   └── se_constants.py   # Sensor mappings
│   │
│   ├── utils/                 # Utility functions
│   │   ├── math_utils.py     # Mathematical operations
│   │   ├── signal_processing.py # Signal processing
│   │   ├── data_processing.py # Data manipulation
│   │   └── visualization_utils.py # Plot utilities
│   │
│   ├── pe/                   # Pose estimation core
│   │   ├── pose_estimation_task.py # MediaPipe integration
│   │   └── vis_pe.py         # Video visualization
│   │
│   └── model/                # ML models
│       └── pose_landmarker_heavy.task # MediaPipe model
│
├── raw codes/                # Legacy analysis scripts
```

## Usage Examples

### Pose Estimation Analysis
1. Set `SELECT_TASK_INDEX = 0` in config.py
2. Set `file_path` to your video file path
3. Run: `python main.py`

**Outputs:**
- `keypoints.csv` - Raw pose keypoints
- `keypoints_cleaned.csv` - Processed keypoints
- `gait_summary.txt` - Analysis report
- `plot/` directory with ROM and gait visualizations

### Smart Insole Analysis
1. Set `SELECT_TASK_INDEX = 1` in config.py
2. Set `file_path` to your CSV sensor data
3. Run: `python main.py`

**Outputs:**
- Step detection results
- Temporal parameters
- Phase percentage analysis
- Pressure distribution plots

## Key Parameters

### Pose Estimation
- **Video Processing**: 30fps
- **Smoothing**: 5-frame Savitzky-Golay filter
- **Model**: MediaPipe Heavy model
- **Joints Analyzed**: 8 key joints (bilateral hip, knee, ankle, shoulder)

### Smart Insole
- **Sensors**: 4-sensor configuration per insole
- **Noise Threshold**: 12.0
- **Minimum Step Duration**: 100ms
- **Analysis Window**: 30% of total data

## Advanced Features

### Signal Processing
- Savitzky-Golay smoothing
- Z-score outlier removal
- Gap interpolation
- Peak detection algorithms

## Contributing

When contributing to this project:
1. Follow existing code conventions
2. Update documentation for new features
3. Ensure compatibility with both pipelines
4. Test with sample data

## Technical Notes

- **Coordinate System**: MediaPipe normalized coordinates (0-1)
- **Sensor Regions**: Heel, midfoot, forefoot, toe regions
- **Data Format**: CSV for sensor data, MP4/AVI for videos
- **Processing**: Real-time capable with optimization
- **Memory**: Efficient processing for large datasets

## Troubleshooting

### Common Issues
1. **MediaPipe Model Missing**: Download `pose_landmarker_heavy.task`
2. **Video Format**: Ensure supported format (MP4, AVI)
3. **Sensor Data**: Check CSV column naming conventions
4. **Dependencies**: Install all required packages from requirements.txt

### Performance Optimization
- Use appropriate video resolution (720p recommended)
- Ensure sufficient RAM for large datasets
- Consider batch processing for multiple files

## License
MIT License


## Contact

1shivam.bhola@gmail.com
