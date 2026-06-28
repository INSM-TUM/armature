"""Benchmark runner for cdrift-evaluation.

Runs ARM drift detector on all benchmark logs with parameter sweep.
Writes results incrementally to CSV for crash recovery.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import List, Optional
import sys

from armature.drift.arm_detector import ARMDriftDetector
from armature.drift.hybrid_detector import HybridDriftDetector
from armature.drift.cdrift_adapter import CdriftDataset
from armature.drift.csv_writer import BenchmarkResult, append_result
from armature.drift.cdrift_metrics import F1_Score, get_avg_lag
from armature.discovery.xes_parser import parse_xes


class BenchmarkRunner:
    """Run ARM detector on cdrift benchmark with parameter sweep."""

    def __init__(
        self,
        cdrift_path: Path,
        output_path: Path,
        window_sizes: Optional[List[int]] = None,
        thresholds: Optional[List[float]] = None,
        step_sizes: Optional[List[int]] = None,
        lag_window: int = 200,
        mode: str = "peaks",
        peak_prominence: float = 0.02,
        detector_type: str = "arm",
    ):
        """Initialize benchmark runner.

        Args:
            cdrift_path: Path to cdrift-evaluation repository
            output_path: Path to output CSV file
            window_sizes: List of window sizes to test (default: [50, 100, 200, 500])
            thresholds: List of thresholds to test (default: [0.05, 0.1, 0.15, 0.2])
            step_sizes: List of step sizes to test (default: [1])
            lag_window: Lag tolerance for F1 computation (default: 200)
            mode: Detection mode - 'threshold', 'adaptive', or 'peaks'
            peak_prominence: Minimum peak prominence for 'peaks' mode
            detector_type: 'arm' for ARM-only, 'hybrid' for DFG chi-squared + ARM
        """
        self.cdrift_path = Path(cdrift_path)
        self.output_path = Path(output_path)
        self.window_sizes = window_sizes or [50, 100, 200, 500]
        self.thresholds = thresholds or [0.05, 0.1, 0.15, 0.2]
        self.step_sizes = step_sizes or [1]
        self.lag_window = lag_window
        self.mode = mode
        self.peak_prominence = peak_prominence
        self.detector_type = detector_type
        self.dataset = CdriftDataset(cdrift_path)

    def run_single_log(
        self, log_path: Path, window_size: int, threshold: float, step_size: int
    ) -> BenchmarkResult:
        """Run ARM detector on single log with given parameters.

        Args:
            log_path: Path to XES log file
            window_size: Window size parameter
            threshold: Threshold parameter
            step_size: Step size parameter

        Returns:
            BenchmarkResult with detection results and metrics
        """
        # Get log metadata and ground truth
        log_info = self.dataset.get_log_info(log_path)
        log_source = log_info["log_source"]
        log_name = log_info["log_name"]
        actual_changepoints = log_info["ground_truth"]

        # Parse XES to get traces
        start_time = perf_counter()
        traces = parse_xes(log_path)
        num_traces = len(traces)

        # Run detector based on type
        if self.detector_type == "hybrid":
            detector = HybridDriftDetector(
                window_size=window_size,
                step_size=step_size,
                prominence=threshold,  # Use threshold as prominence for hybrid
                min_gap=window_size,
                explain=False,  # Skip explanations for benchmark speed
            )
            result = detector.detect(traces)
            algorithm = "ARM-Hybrid"
        else:
            detector = ARMDriftDetector(
                window_size=window_size,
                threshold=threshold,
                step_size=step_size,
                mode=self.mode,
                peak_prominence=self.peak_prominence,
                min_gap=window_size,
            )
            result = detector.detect(traces)
            algorithm = "ARM"
        duration_seconds = perf_counter() - start_time

        # Compute evaluation metrics using cdrift's LP-based matching
        detected_changepoints = result.drift_indices
        f1_score = F1_Score(detected_changepoints, actual_changepoints, lag=self.lag_window)

        # Compute average lag for matched pairs
        average_lag = get_avg_lag(detected_changepoints, actual_changepoints, lag=self.lag_window)

        return BenchmarkResult(
            algorithm=algorithm,
            log_source=log_source,
            log_name=log_name,
            detected_changepoints=detected_changepoints,
            actual_changepoints=actual_changepoints,
            f1_score=f1_score,
            average_lag=average_lag,
            duration_seconds=duration_seconds,
            num_traces=num_traces,
            window_size=window_size,
            threshold=threshold,
            step_size=step_size,
        )

    def run_all(
        self, progress: bool = True, log_subset: Optional[List[Path]] = None
    ) -> List[BenchmarkResult]:
        """Run benchmark on all logs with all parameter combinations.

        Args:
            progress: Show progress bar (requires tqdm)
            log_subset: Optional list of logs to run (default: all logs)

        Returns:
            List of BenchmarkResult objects
        """
        # Get logs to process
        logs = log_subset if log_subset is not None else self.dataset.list_logs()

        # Generate all parameter combinations
        param_combinations = list(product(self.window_sizes, self.thresholds, self.step_sizes))

        # Total iterations for progress bar
        total = len(logs) * len(param_combinations)

        # Setup progress bar if requested
        if progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=total, desc="Running benchmark")
            except ImportError:
                print("Warning: tqdm not installed, progress bar disabled", file=sys.stderr)
                progress = False

        results = []
        for log_path in logs:
            for window_size, threshold, step_size in param_combinations:
                try:
                    result = self.run_single_log(log_path, window_size, threshold, step_size)
                    results.append(result)

                    # Write result incrementally for crash recovery
                    append_result(result, self.output_path)

                    if progress:
                        pbar.update(1)
                        pbar.set_postfix(
                            log=log_path.stem,
                            w=window_size,
                            t=f"{threshold:.2f}",
                            f1=f"{result.f1_score:.3f}",
                        )
                except Exception as e:
                    print(
                        f"Error processing {log_path.name} (w={window_size}, t={threshold}): {e}",
                        file=sys.stderr,
                    )
                    if progress:
                        pbar.update(1)
                    continue

        if progress:
            pbar.close()

        return results

    def run_parameter_sweep(
        self, log_subset: Optional[List[Path]] = None, progress: bool = True
    ) -> List[BenchmarkResult]:
        """Run parameter sweep on subset of logs for testing.

        Args:
            log_subset: Logs to test (default: first 5)
            progress: Show progress bar

        Returns:
            List of BenchmarkResult objects
        """
        if log_subset is None:
            log_subset = self.dataset.list_logs()[:5]

        return self.run_all(progress=progress, log_subset=log_subset)


def run_arm_benchmark(
    cdrift_path: Path,
    output_path: Path,
    window_sizes: List[int] = None,
    thresholds: List[float] = None,
    step_sizes: List[int] = None,
    lag_window: int = 200,
    progress: bool = True,
) -> List[BenchmarkResult]:
    """Convenience function to run ARM benchmark.

    Args:
        cdrift_path: Path to cdrift-evaluation repository
        output_path: Path to output CSV file
        window_sizes: List of window sizes to test
        thresholds: List of thresholds to test
        step_sizes: List of step sizes to test
        lag_window: Lag tolerance for F1 computation
        progress: Show progress bar

    Returns:
        List of BenchmarkResult objects
    """
    runner = BenchmarkRunner(
        cdrift_path=cdrift_path,
        output_path=output_path,
        window_sizes=window_sizes,
        thresholds=thresholds,
        step_sizes=step_sizes,
        lag_window=lag_window,
    )
    return runner.run_all(progress=progress)
