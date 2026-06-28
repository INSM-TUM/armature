"""CSV writer for cdrift-compatible benchmark results.

Outputs results in exact cdrift-evaluation framework format for comparison.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Exact column names from cdrift framework
REQUIRED_COLUMNS = [
    "Algorithm",
    "Log Source",
    "Log",
    "Detected Changepoints",
    "Actual Changepoints for Log",
    "F1-Score",
    "Average Lag",
    "Duration",
    "Duration (Seconds)",
    "Seconds per Case",
    "Window Size",
    "Threshold",
    "Step Size",
]


@dataclass
class BenchmarkResult:
    """Single benchmark result for one algorithm + log + parameters.

    Attributes:
        algorithm: Algorithm name (e.g., "ARM", "Bose")
        log_source: Log source dataset (Bose/Ceravolo/Ostovar)
        log_name: Log filename without extension
        detected_changepoints: Detected changepoint indices
        actual_changepoints: Ground truth changepoint indices
        f1_score: F1-score (computed by evaluation)
        average_lag: Average lag for matched pairs
        duration_seconds: Execution time in seconds
        num_traces: Number of traces in log
        window_size: Window size parameter
        threshold: Threshold parameter
        step_size: Step size parameter
    """

    algorithm: str
    log_source: str
    log_name: str
    detected_changepoints: List[int]
    actual_changepoints: List[int]
    f1_score: Optional[float] = None
    average_lag: Optional[float] = None
    duration_seconds: float = 0.0
    num_traces: int = 0
    window_size: int = 0
    threshold: float = 0.0
    step_size: int = 1


def _format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "01:02:03")
    """
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _result_to_row(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to CSV row dict.

    Args:
        result: Benchmark result

    Returns:
        Dict with REQUIRED_COLUMNS keys
    """
    seconds_per_case = (
        result.duration_seconds / result.num_traces if result.num_traces > 0 else 0.0
    )

    return {
        "Algorithm": result.algorithm,
        "Log Source": result.log_source,
        "Log": result.log_name,
        "Detected Changepoints": str(result.detected_changepoints),
        "Actual Changepoints for Log": str(result.actual_changepoints),
        "F1-Score": result.f1_score,
        "Average Lag": result.average_lag,
        "Duration": _format_duration(result.duration_seconds),
        "Duration (Seconds)": result.duration_seconds,
        "Seconds per Case": seconds_per_case,
        "Window Size": result.window_size,
        "Threshold": result.threshold,
        "Step Size": result.step_size,
    }


def write_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Write benchmark results to CSV in cdrift format.

    Args:
        results: List of benchmark results
        output_path: Output CSV file path
    """
    rows = [_result_to_row(r) for r in results]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df.to_csv(output_path, index=False)


def append_result(result: BenchmarkResult, output_path: Path) -> None:
    """Append single result to existing CSV.

    Creates file with headers if doesn't exist.

    Args:
        result: Benchmark result to append
        output_path: CSV file path
    """
    row = _result_to_row(result)

    if output_path.exists():
        # Append to existing file
        df_existing = pd.read_csv(output_path)
        df_new = pd.DataFrame([row], columns=REQUIRED_COLUMNS)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
    else:
        # Create new file
        df = pd.DataFrame([row], columns=REQUIRED_COLUMNS)
        df.to_csv(output_path, index=False)
