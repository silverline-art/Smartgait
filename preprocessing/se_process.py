from config import file_path, output_path
import pandas as pd
from utils.data_processing import read_input_csv

def format_floats(obj):
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, tuple):
        return tuple(format_floats(x) for x in obj)
    elif isinstance(obj, list):
        return [format_floats(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: format_floats(v) for k, v in obj.items()}
    else:
        return obj