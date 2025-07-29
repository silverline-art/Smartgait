"""Visualization utilities for SmartGait analysis."""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, Dict, Optional, List


def setup_matplotlib_defaults():
    """Set up default matplotlib parameters for consistent plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 100
    })


def get_foot_colors() -> Dict[str, str]:
    """Get consistent color scheme for left/right foot visualization.
    
    Returns:
        Dictionary with left and right foot colors
    """
    return {
        'left': '#1f77b4',   # Blue
        'right': '#d62728',  # Red
        'both': '#2ca02c'    # Green
    }


def create_output_directory(output_path: str) -> None:
    """Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output directory
    """
    os.makedirs(output_path, exist_ok=True)


def save_plot_with_dpi(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """Save plot with specified DPI and create directory if needed.
    
    Args:
        fig: Matplotlib figure object
        filepath: Full path where to save the plot
        dpi: Resolution in dots per inch
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')


def create_subplot_grid(nrows: int, ncols: int, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Create subplot grid with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes_array)
    """
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always an array
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    
    fig.tight_layout(pad=3.0)
    return fig, axes


def add_phase_shading(ax: plt.Axes, phase_starts: List[int], phase_ends: List[int], 
                     colors: List[str] = None, alpha: float = 0.3, labels: List[str] = None) -> None:
    """Add shaded regions to indicate gait phases.
    
    Args:
        ax: Matplotlib axes object
        phase_starts: List of phase start indices
        phase_ends: List of phase end indices  
        colors: List of colors for each phase
        alpha: Transparency level
        labels: Labels for each phase
    """
    if colors is None:
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, (start, end) in enumerate(zip(phase_starts, phase_ends)):
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else None
        ax.axvspan(start, end, alpha=alpha, color=color, label=label)


def plot_time_series_with_events(ax: plt.Axes, time: np.ndarray, data: np.ndarray,
                                events: Dict[str, np.ndarray] = None, 
                                title: str = "", ylabel: str = "", 
                                color: str = 'blue') -> None:
    """Plot time series data with event markers.
    
    Args:
        ax: Matplotlib axes object
        time: Time array
        data: Data array
        events: Dictionary of event types and their time points
        title: Plot title
        ylabel: Y-axis label
        color: Line color
    """
    ax.plot(time, data, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    
    if events:
        event_colors = {'peaks': 'red', 'valleys': 'green', 'contacts': 'orange'}
        for event_type, event_times in events.items():
            color = event_colors.get(event_type, 'black')
            ax.scatter(event_times, np.interp(event_times, time, data), 
                      c=color, s=30, label=event_type, zorder=5)
        ax.legend()


def create_gait_dashboard(time: np.ndarray, left_data: np.ndarray, right_data: np.ndarray,
                         left_events: Dict = None, right_events: Dict = None,
                         title: str = "Gait Analysis Dashboard") -> plt.Figure:
    """Create comprehensive gait analysis dashboard.
    
    Args:
        time: Time array
        left_data: Left foot data
        right_data: Right foot data
        left_events: Left foot events
        right_events: Right foot events
        title: Dashboard title
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = create_subplot_grid(2, 1, figsize=(14, 10))
    colors = get_foot_colors()
    
    # Left foot plot
    plot_time_series_with_events(axes[0], time, left_data, left_events,
                                title=f"{title} - Left Foot", 
                                ylabel="Pressure/Angle", color=colors['left'])
    
    # Right foot plot  
    plot_time_series_with_events(axes[1], time, right_data, right_events,
                                title=f"{title} - Right Foot",
                                ylabel="Pressure/Angle", color=colors['right'])
    
    return fig


def plot_pressure_heatmap(ax: plt.Axes, pressure_matrix: np.ndarray, 
                         regions: List[str] = None, title: str = "Pressure Distribution") -> None:
    """Plot pressure distribution as heatmap.
    
    Args:
        ax: Matplotlib axes object
        pressure_matrix: 2D array of pressure values
        regions: List of region names
        title: Plot title
    """
    im = ax.imshow(pressure_matrix, cmap='hot', interpolation='nearest', aspect='auto')
    ax.set_title(title)
    
    if regions:
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
    
    plt.colorbar(im, ax=ax, label='Pressure')


def plot_angle_range_of_motion(ax: plt.Axes, angles_dict: Dict[str, np.ndarray],
                              title: str = "Range of Motion") -> None:
    """Plot range of motion for multiple joints.
    
    Args:
        ax: Matplotlib axes object
        angles_dict: Dictionary of joint names and angle arrays
        title: Plot title
    """
    joint_names = list(angles_dict.keys())
    rom_values = []
    
    for joint, angles in angles_dict.items():
        if len(angles) > 0:
            rom = np.max(angles) - np.min(angles)
            rom_values.append(rom)
        else:
            rom_values.append(0)
    
    bars = ax.bar(joint_names, rom_values, color='skyblue', alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel('Range of Motion (degrees)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, rom_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}°', ha='center', va='bottom')


def plot_step_parameters_comparison(ax: plt.Axes, left_params: Dict, right_params: Dict,
                                  title: str = "Step Parameters Comparison") -> None:
    """Plot comparison of step parameters between left and right feet.
    
    Args:
        ax: Matplotlib axes object
        left_params: Dictionary of left foot parameters
        right_params: Dictionary of right foot parameters  
        title: Plot title
    """
    parameters = list(left_params.keys())
    left_values = list(left_params.values())
    right_values = list(right_params.values())
    
    x = np.arange(len(parameters))
    width = 0.35
    colors = get_foot_colors()
    
    ax.bar(x - width/2, left_values, width, label='Left', color=colors['left'], alpha=0.7)
    ax.bar(x + width/2, right_values, width, label='Right', color=colors['right'], alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Parameters')
    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)


def add_statistical_annotations(ax: plt.Axes, data1: np.ndarray, data2: np.ndarray,
                               positions: Tuple[float, float], test_type: str = "t-test") -> None:
    """Add statistical significance annotations to plots.
    
    Args:
        ax: Matplotlib axes object
        data1: First dataset
        data2: Second dataset
        positions: (x1, x2) positions for annotation
        test_type: Type of statistical test performed
    """
    # This is a placeholder for statistical testing
    # In a full implementation, you would perform actual statistical tests
    p_value = 0.05  # Placeholder
    
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"  
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    # Add significance bar
    y_max = max(np.max(data1), np.max(data2))
    y_pos = y_max * 1.1
    
    ax.plot([positions[0], positions[1]], [y_pos, y_pos], 'k-', linewidth=1)
    ax.text((positions[0] + positions[1])/2, y_pos * 1.02, significance, 
            ha='center', va='bottom')