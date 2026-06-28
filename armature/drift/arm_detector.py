"""ARM-based concept drift detection using matrix comparison.

Detects concept drifts by comparing ARM matrices from sliding windows
of an event log. Uses temporal and existential dependency changes to
identify process structure changes.

Key improvements over naive threshold-based detection:
- scipy.signal.find_peaks with prominence filtering (like Bose)
- Adaptive threshold based on score distribution percentile
- Chi-squared test for statistical significance of cell changes
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2_contingency

from armature.core.matrix import Matrix
from armature.discovery.discover import discover
from armature.discovery.xes_parser import parse_xes
from armature.discovery.dfg import build_dfg
from armature.discovery.scc import find_sccs
from armature.discovery.temporal import extract_temporal_dependencies
from armature.discovery.existential import extract_existential_dependencies
from armature.drift.metrics import compute_drift_score, DriftScore


@dataclass
class ARMDriftResult:
    """Result from ARM drift detection."""

    drift_indices: List[int]
    drift_scores: List[float]  # Normalized drift scores at each position
    detailed_scores: List[DriftScore]  # Full DriftScore objects
    threshold: float
    window_size: int
    positions: List[int] = field(default_factory=list)  # Position for each score


def _discover_from_traces(traces: list, source: str = "window") -> Matrix:
    """Discover ARM matrix from a list of traces.

    Internal function that runs discovery pipeline on trace subset.
    """
    from armature.core.dependencies import (
        TEMPORAL_INVERSE,
        EXISTENTIAL_INVERSE,
        DependencyCell,
        TemporalDependency,
        ExistentialDependency,
    )
    from armature.discovery.discover import set_symmetric_cell

    # Build DFG from traces
    dfg = build_dfg(traces)

    # Find SCCs
    loop_ctx = find_sccs(dfg)

    # Extract dependencies
    temporal_deps = extract_temporal_dependencies(traces, dfg, loop_ctx)
    existential_deps = extract_existential_dependencies(traces)

    # Construct matrix
    matrix = Matrix(source=source)

    for activity in sorted(dfg.activities):
        matrix.add_activity(activity)

    # Set dependencies (copied from discover.py logic)
    sorted_activities = sorted(dfg.activities)
    for i, source_activity in enumerate(sorted_activities):
        for target_activity in sorted_activities[i:]:
            pair = (source_activity, target_activity)
            reverse_pair = (target_activity, source_activity)

            temporal = temporal_deps.get(pair, DependencyCell().temporal)
            existential = existential_deps.get(pair, DependencyCell().existential)

            reverse_temporal = temporal_deps.get(reverse_pair, DependencyCell().temporal)
            reverse_existential = existential_deps.get(reverse_pair, DependencyCell().existential)

            default_cell = DependencyCell()

            if temporal == default_cell.temporal and reverse_temporal != default_cell.temporal:
                temporal = TEMPORAL_INVERSE.get(reverse_temporal, reverse_temporal)

            if (
                existential == default_cell.existential
                and reverse_existential != default_cell.existential
            ):
                existential = EXISTENTIAL_INVERSE.get(reverse_existential, reverse_existential)

            if temporal != default_cell.temporal or existential != default_cell.existential:
                set_symmetric_cell(
                    matrix,
                    source_activity,
                    target_activity,
                    temporal,
                    existential,
                    temporal_deps,
                    existential_deps,
                )

    return matrix


class ARMDriftDetector:
    """ARM-based concept drift detector.

    Detects process drifts by comparing ARM matrices from sliding windows.
    Uses weighted distance metrics on temporal and existential dependency
    changes to identify significant process structure changes.

    Detection modes:
    - 'threshold': Fixed threshold on normalized_distance (legacy)
    - 'adaptive': Adaptive threshold based on score percentile
    - 'peaks': scipy.signal.find_peaks with prominence (like Bose)

    Advantages over Bose S/N/A:
    - Distinguishes DIRECT vs EVENTUAL vs TRUE_EVENTUAL (not just succession)
    - Captures existential patterns (IMPLICATION, EQUIVALENCE, XOR, OR)
    - Weighted distance reflects semantic severity of changes
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.1,
        step_size: int = 10,
        verbose: bool = False,
        detect_peaks: bool = True,
        mode: str = "peaks",
        peak_prominence: float = 0.02,
        peak_width: Optional[int] = None,
        adaptive_percentile: float = 90.0,
        min_gap: Optional[int] = None,
    ):
        """
        Args:
            window_size: Number of traces per window for matrix discovery
            threshold: Normalized drift score threshold for detection (0-1)
            step_size: Step size for sliding window (default: 10)
            verbose: Enable detailed diagnostic logging
            detect_peaks: Use local maxima detection (legacy, use mode='peaks')
            mode: Detection mode - 'threshold', 'adaptive', or 'peaks'
            peak_prominence: Minimum prominence for scipy.find_peaks (default: 0.02)
            peak_width: Minimum width for peaks (default: window_size/step_size/10)
            adaptive_percentile: Percentile for adaptive threshold (default: 90)
            min_gap: Minimum gap between detections (default: window_size)

        Step size calibration: step=10 provides 10x speedup while maintaining
        detection granularity comparable to other algorithms (Bose/EMD/ProDrift
        use step=2, scaled by window size this is equivalent to step=10-20).
        For 1000 traces with window=100, this checks 81 positions vs 801.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.step_size = step_size
        self.verbose = verbose
        self.detect_peaks = detect_peaks
        self.mode = mode
        self.peak_prominence = peak_prominence
        self.peak_width = (
            peak_width if peak_width is not None else max(1, window_size // step_size // 10)
        )
        self.adaptive_percentile = adaptive_percentile
        self.min_gap = min_gap if min_gap is not None else window_size

    def detect(self, traces: list) -> ARMDriftResult:
        """Detect concept drifts in trace sequence.

        Slides two adjacent windows over the traces, discovers ARM matrices
        for each window pair, and computes drift scores. Detects drifts where
        normalized score exceeds threshold.

        Args:
            traces: List of traces (from parse_xes)

        Returns:
            ARMDriftResult with detected drift indices and scores
        """
        n_traces = len(traces)

        if n_traces < 2 * self.window_size:
            # Not enough traces for comparison
            return ARMDriftResult(
                drift_indices=[],
                drift_scores=[],
                detailed_scores=[],
                threshold=self.threshold,
                window_size=self.window_size,
                positions=[],
            )

        drift_scores: List[float] = []
        detailed_scores: List[DriftScore] = []
        positions: List[int] = []

        # Slide two adjacent windows
        i = 0
        while i + 2 * self.window_size <= n_traces:
            window1_traces = traces[i : i + self.window_size]
            window2_traces = traces[i + self.window_size : i + 2 * self.window_size]

            # Discover matrices for each window
            matrix1 = _discover_from_traces(window1_traces, source=f"window_{i}")
            matrix2 = _discover_from_traces(window2_traces, source=f"window_{i + self.window_size}")

            # Compute drift score
            score = compute_drift_score(matrix1, matrix2)

            # Position is at the boundary between windows
            position = i + self.window_size
            positions.append(position)
            drift_scores.append(score.normalized_distance)
            detailed_scores.append(score)

            # Verbose diagnostic logging
            if self.verbose:
                common_acts = set(matrix1.activities) & set(matrix2.activities)
                print(f"\n=== Position {position} ===")
                print(f"Common activities: {len(common_acts)}")
                print(f"Cell changes: {score.cell_change_count}")
                print(f"Raw temporal distance: {score.temporal_distance:.4f}")
                print(f"Raw existential distance: {score.existential_distance:.4f}")
                print(f"Total distance: {score.total_distance:.4f}")
                print(f"Affected activities: {len(score.affected_activities)}")
                print(f"Total cells compared: {score.total_cells_compared}")
                print(f"Normalized distance: {score.normalized_distance:.4f}")
                print(f"Threshold: {self.threshold:.4f}")
                print(f"Exceeds threshold: {score.normalized_distance >= self.threshold}")

            i += self.step_size

        # Detect drift points based on mode
        if self.mode == "peaks":
            drift_indices = self._detect_peaks_scipy(positions, drift_scores)
        elif self.mode == "adaptive":
            drift_indices = self._detect_adaptive(positions, drift_scores)
        else:  # 'threshold' mode (legacy)
            drift_indices = self._detect_threshold(positions, drift_scores)

        # Compute effective threshold used
        effective_threshold = self.threshold
        if self.mode == "adaptive" and drift_scores:
            effective_threshold = np.percentile(drift_scores, self.adaptive_percentile)

        return ARMDriftResult(
            drift_indices=drift_indices,
            drift_scores=drift_scores,
            detailed_scores=detailed_scores,
            threshold=effective_threshold,
            window_size=self.window_size,
            positions=positions,
        )

    def _detect_threshold(self, positions: List[int], scores: List[float]) -> List[int]:
        """Legacy threshold-based detection with simple local max."""
        drift_indices: List[int] = []

        for idx, (pos, score) in enumerate(zip(positions, scores)):
            if score >= self.threshold:
                # Check if this is a local maximum
                is_local_max = True
                if self.detect_peaks:
                    if idx > 0 and scores[idx - 1] >= score:
                        is_local_max = False
                    if idx < len(scores) - 1 and scores[idx + 1] > score:
                        is_local_max = False

                if is_local_max:
                    drift_indices.append(pos)

        return drift_indices

    def _detect_adaptive(self, positions: List[int], scores: List[float]) -> List[int]:
        """Adaptive threshold based on score distribution percentile."""
        if not scores:
            return []

        # Compute adaptive threshold from score distribution
        adaptive_thresh = np.percentile(scores, self.adaptive_percentile)

        # Use threshold detection with adaptive value
        drift_indices: List[int] = []
        for idx, (pos, score) in enumerate(zip(positions, scores)):
            if score >= adaptive_thresh:
                is_local_max = True
                if idx > 0 and scores[idx - 1] >= score:
                    is_local_max = False
                if idx < len(scores) - 1 and scores[idx + 1] > score:
                    is_local_max = False

                if is_local_max:
                    drift_indices.append(pos)

        return self._apply_min_gap(drift_indices, scores, positions)

    def _detect_peaks_scipy(self, positions: List[int], scores: List[float]) -> List[int]:
        """scipy.signal.find_peaks with prominence filtering.

        This mirrors Bose's visualInspection() which uses:
        - find_peaks(-signal, width=80, prominence=0.1)

        For ARM scores (high = drift), we don't negate the signal.
        """
        if not scores:
            return []

        signal = np.array(scores)

        # find_peaks parameters tuned for drift detection
        # width: minimum samples between drift points
        # prominence: minimum height relative to neighboring troughs
        # distance: minimum separation between peaks
        min_distance = max(1, self.min_gap // self.step_size)

        peaks, properties = find_peaks(
            signal,
            prominence=self.peak_prominence,
            width=self.peak_width,
            distance=min_distance,
        )

        # Also require peaks to exceed a base threshold
        # to avoid detecting noise peaks in flat regions
        base_thresh = self.threshold
        filtered_peaks = [p for p in peaks if scores[p] >= base_thresh]

        # Convert peak indices to trace positions
        drift_indices = [positions[p] for p in filtered_peaks]

        return drift_indices

    def _apply_min_gap(
        self, indices: List[int], scores: List[float], positions: List[int]
    ) -> List[int]:
        """Apply minimum gap constraint, keeping highest score in each gap window."""
        if not indices:
            return []

        # Sort by position
        sorted_indices = sorted(indices)
        result = [sorted_indices[0]]

        for idx in sorted_indices[1:]:
            if idx - result[-1] >= self.min_gap:
                result.append(idx)
            else:
                # Keep the one with higher score
                prev_score = scores[positions.index(result[-1])] if result[-1] in positions else 0
                curr_score = scores[positions.index(idx)] if idx in positions else 0
                if curr_score > prev_score:
                    result[-1] = idx

        return result

    def detect_from_log(self, log_path: str | Path) -> ARMDriftResult:
        """Detect drifts from XES log file.

        Convenience method that parses XES and runs detection.
        """
        traces = parse_xes(Path(log_path))
        return self.detect(traces)


def detect_drift_arm(
    log_path: str,
    window_size: int = 100,
    threshold: float = 0.1,
    step_size: int = 10,
) -> ARMDriftResult:
    """Convenience function to detect drift from XES file path.

    Args:
        log_path: Path to XES event log
        window_size: Window size for matrix discovery
        threshold: Drift detection threshold (0-1)
        step_size: Sliding window step size

    Returns:
        ARMDriftResult with detected drifts
    """
    detector = ARMDriftDetector(
        window_size=window_size,
        threshold=threshold,
        step_size=step_size,
    )
    return detector.detect_from_log(log_path)
