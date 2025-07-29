import os
import numpy as np
import matplotlib.pyplot as plt

PHASE_NAMES = [
    ("Stance phase (%)",      "stance_phase"),
    ("Swing phase (%)",       "swing_phase"),
    ("Single support (%)",    "single_support"),
    ("Mid stance (%)",        "mid_stance"),
    ("Terminal stance (%)",   "terminal_stance"),
    ("Double support (%)",    "double_support"),
    ("Loading response (%)",  "loading_response"),
    ("Pre-swing (%)",         "pre_swing"),
]

def _extract_flat_phase_percentages(phase_metrics):
    """
    Extract a flat dict of phase percentages from the input.
    Accepts:
      - dict with keys matching PHASE_NAMES
      - dict with left/right keys (e.g., 'stance_phase_left'), uses mean of both
      - dict with 'summary' or 'percentages' subdicts
    Returns:
      dict with keys matching PHASE_NAMES and float values, or None if not found.
    """
    # Direct flat mapping
    if all(k in phase_metrics for _, k in PHASE_NAMES):
        return {k: float(phase_metrics[k]) for _, k in PHASE_NAMES}

    # If left/right keys exist, average them
    left_keys = [f"{k}_left" for _, k in PHASE_NAMES]
    right_keys = [f"{k}_right" for _, k in PHASE_NAMES]
    if all((lk in phase_metrics and rk in phase_metrics) for lk, rk in zip(left_keys, right_keys)):
        out = {}
        for (_, k), lk, rk in zip(PHASE_NAMES, left_keys, right_keys):
            try:
                lval = phase_metrics[lk]
                rval = phase_metrics[rk]
                # Accept tuple (mean, std) or float
                if isinstance(lval, (tuple, list, np.ndarray)):
                    lval = lval[0]
                if isinstance(rval, (tuple, list, np.ndarray)):
                    rval = rval[0]
                out[k] = float(lval + rval) / 2.0
            except Exception:
                out[k] = 0.0
        return out

    # If nested under 'percentages' or 'summary'
    for subkey in ['percentages', 'summary']:
        if subkey in phase_metrics:
            return _extract_flat_phase_percentages(phase_metrics[subkey])

    # If nested under 'phase_percentages'
    if 'phase_percentages' in phase_metrics:
        return _extract_flat_phase_percentages(phase_metrics['phase_percentages'])

    # Not found
    return None

def plot_phase_percentages(phase_metrics: dict, output_dir: str, filename: str = "phase_percentages.png"):
    """
    Given a dict of phase keys→percentage values (or a result dict), generate and save a bar chart.
    If all values are zero or missing, a warning is printed and the plot is not saved.
    """
    # Try to extract phase percentages robustly
    flat = _extract_flat_phase_percentages(phase_metrics)
    if flat is None:
        print("[ERROR] Could not extract phase percentages from input dict. No plot will be generated.")
        return

    labels, values = [], []
    missing_keys = []
    for disp, key in PHASE_NAMES:
        labels.append(disp)
        val = flat.get(key, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_keys.append(key)
            values.append(0.0)
        else:
            try:
                values.append(float(val))
            except Exception:
                values.append(0.0)
                missing_keys.append(key)

    if missing_keys:
        print(f"[WARNING] The following phase keys are missing or invalid in input: {missing_keys}")

    if all(v == 0.0 for v in values):
        print("[WARNING] All phase percentage values are zero or missing. No plot will be generated.")
        return

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    plt.bar(labels, values, color=colors[:len(labels)])
    plt.ylabel("Percentage (%)")
    plt.title("Gait Phase Percentages")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved phase percentage plot to: {out_path}")