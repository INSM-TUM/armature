"""cdrift-compatible wrapper for ARM-Hybrid drift detector.

Provides test_arm_hybrid() function matching cdrift's standard detector
signature for integration into testAll_reproducibility.py orchestration.
"""

import math
from datetime import datetime
from pathlib import Path
from timeit import default_timer
from typing import List, Dict, Any

from armature.discovery.xes_parser import parse_xes
from armature.drift.hybrid_detector import HybridDriftDetector
from armature.drift.cdrift_metrics import F1_Score, get_avg_lag


def calcDurFromSeconds(seconds: float) -> str:
    """Format seconds to hh:mm:ss string.
    
    Matches cdrift's duration formatting convention.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Duration formatted as "HH:MM:SS"
    """
    seconds = math.floor(seconds)
    return datetime.strftime(
        datetime.utcfromtimestamp(seconds), '%H:%M:%S'
    )


def test_arm_hybrid(
    filepath: str,
    window_size: int,
    prominence: float,
    step_size: int,
    F1_LAG: int,
    cp_locations: List[int],
    position: int = None,
    show_progress_bar: bool = True
) -> List[Dict[str, Any]]:
    """Run ARM-Hybrid detector on single log.
    
    Matches cdrift signature for integration into testAll_reproducibility.py
    
    Args:
        filepath: Path to XES log file (.xes or .xes.gz)
        window_size: Sliding window size in traces
        prominence: Peak prominence threshold for chi-squared detection (0.5-2.0)
        step_size: Step between windows
        F1_LAG: Lag window for F1 computation (from orchestrator, typically 200)
        cp_locations: Ground truth changepoints (from CSV, list of ints)
        position: tqdm position for parallel progress bar (unused currently)
        show_progress_bar: Enable progress reporting (unused currently)
        
    Returns:
        List containing single dict with standardized columns:
        - Algorithm, Log, Log Source
        - window_size, prominence, step_size (parameter columns)
        - Detected Changepoints, Actual Changepoints for Log
        - F1-Score, Average Lag
        - Duration, Duration (Seconds), Seconds per Case
    """
    # Extract metadata from filepath
    logname = Path(filepath).stem.replace('.xes', '')  # Handle .xes.gz -> remove .xes
    log_source = Path(filepath).parent.name
    
    # Parse XES OUTSIDE timer (exclude parsing overhead)
    traces = parse_xes(filepath)
    
    # Start timing AFTER parsing
    start = default_timer()
    
    # Create and run detector
    detector = HybridDriftDetector(
        window_size=window_size,
        step_size=step_size,
        prominence=prominence,
        min_gap=window_size,  # Standard: min_gap = window_size
        explain=False,  # Skip explanations for speed
    )
    result = detector.detect(traces)
    
    # Stop timer
    duration = default_timer() - start
    
    # Convert numpy array to Python list (critical for CSV serialization)
    detected = list(result.drift_indices)
    
    # Compute metrics using cdrift standard functions
    f1 = F1_Score(detected, cp_locations, lag=F1_LAG)
    avg_lag = get_avg_lag(detected, cp_locations, lag=F1_LAG)
    
    # Return in cdrift format (list with single dict)
    return [{
        'Algorithm': 'ARM-Hybrid',
        'Log Source': log_source,
        'Log': logname,
        'window_size': window_size,         # Parameter column
        'prominence': prominence,           # Parameter column
        'step_size': step_size,             # Parameter column
        'Detected Changepoints': detected,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': f1,
        'Average Lag': avg_lag,
        'Duration': calcDurFromSeconds(duration),
        'Duration (Seconds)': duration,
        'Seconds per Case': duration / len(traces),
    }]
