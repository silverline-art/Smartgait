from collections import namedtuple

# --- Hysteresis and Event Detection ---
# Absolute pressure value to distinguish contact from noise.
# Adjusted for the new, stricter step validation logic.
NOISE_THRESHOLD = 12.0
# A step event must last at least this long to be considered valid
MIN_STEP_DURATION_MS = 100
# Gaps between contact blocks shorter than this (in ms) will be merged into a single step
MAX_GAP_MS_FOR_MERGE = 50

# --- Analysis Parameters ---
# Fraction of each step interval to use for pressure calculation (default 30%)
PRESSURE_PERCENT = 0.3

def get_sensor_region_constants():
    SensorRegionConstants = namedtuple("SensorRegionConstants", [
        "LEFT_FOREFOOT",
        "LEFT_MIDFOOT",
        "LEFT_HINDFOOT",
        "RIGHT_FOREFOOT",
        "RIGHT_MIDFOOT",
        "RIGHT_HINDFOOT",
        "INSOLE_LENGTH_MM"
    ])
    return SensorRegionConstants(
        ['L_value2'],                  # LEFT_FOREFOOT
        ['L_value1', 'L_value3'],      # LEFT_MIDFOOT
        ['L_value4'],                  # LEFT_HINDFOOT
        ['R_value2'],                  # RIGHT_FOREFOOT
        ['R_value1', 'R_value3'],      # RIGHT_MIDFOOT
        ['R_value4'],                  # RIGHT_HINDFOOT
        230.0                          # INSOLE_LENGTH_MM
    )