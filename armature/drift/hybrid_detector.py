"""Hybrid drift detector: DFG chi-squared for detection, ARM for explanation.

Combines the detection power of trace-level DFG features with the
interpretability of ARM matrix diffs.

Detection uses chi-squared test on aggregated DFG counts between windows,
which captures subtle changes that ARM matrices miss.

Explanation uses ARM matrix diff to show WHAT temporal/existential
relationships changed, providing semantic insight into the drift.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy.stats import chi2_contingency
from scipy.signal import find_peaks

from armature.core.matrix import Matrix
from armature.core.dependencies import DependencyCell
from armature.discovery.xes_parser import parse_xes, Trace
from armature.drift.arm_detector import _discover_from_traces
from armature.drift.metrics import compute_drift_score, DriftScore


@dataclass
class DriftExplanation:
    """Explanation of what changed at a drift point."""

    position: int
    chi2_score: float
    arm_score: Optional[DriftScore]
    temporal_changes: List[Tuple[str, str, str, str]]  # (act1, act2, before, after)
    existential_changes: List[Tuple[str, str, str, str]]
    affected_activities: Set[str]

    def summary(self) -> str:
        """Human-readable summary of the drift."""
        lines = [f"Drift at position {self.position}:"]
        lines.append(f"  Chi-squared score: {self.chi2_score:.2f}")
        if self.arm_score:
            lines.append(f"  ARM cell changes: {self.arm_score.cell_change_count}")
        lines.append(f"  Affected activities: {len(self.affected_activities)}")

        if self.temporal_changes:
            lines.append("  Temporal changes:")
            for a, b, before, after in self.temporal_changes[:5]:
                lines.append(f"    ({a}, {b}): {before} -> {after}")
            if len(self.temporal_changes) > 5:
                lines.append(f"    ... and {len(self.temporal_changes) - 5} more")

        if self.existential_changes:
            lines.append("  Existential changes:")
            for a, b, before, after in self.existential_changes[:5]:
                lines.append(f"    ({a}, {b}): {before} -> {after}")
            if len(self.existential_changes) > 5:
                lines.append(f"    ... and {len(self.existential_changes) - 5} more")

        return "\n".join(lines)


@dataclass
class HybridDriftResult:
    """Result from hybrid drift detection."""

    drift_indices: List[int]
    chi2_scores: List[float]  # Chi-squared scores at each position
    positions: List[int]  # Position for each score
    explanations: List[DriftExplanation]  # Detailed explanation per detected drift
    window_size: int
    prominence: float


def _window_dfg(traces: List[Trace]) -> Dict[Tuple[str, str], int]:
    """Compute aggregated DFG counts for a window of traces."""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for trace in traces:
        for i in range(len(trace.events) - 1):
            a = trace.events[i].activity
            b = trace.events[i + 1].activity
            counts[(a, b)] += 1
    return dict(counts)


def _compute_chi2(
    dfg1: Dict[Tuple[str, str], int],
    dfg2: Dict[Tuple[str, str], int],
    all_pairs: List[Tuple[str, str]],
) -> float:
    """Compute normalized chi-squared statistic between two DFG distributions."""
    counts1 = np.array([dfg1.get(p, 0) for p in all_pairs])
    counts2 = np.array([dfg2.get(p, 0) for p in all_pairs])

    # Filter to pairs with some activity
    active = (counts1 + counts2) > 0
    if active.sum() <= 1:
        return 0.0

    contingency = np.array([counts1[active], counts2[active]])
    try:
        chi2, pval, dof, expected = chi2_contingency(contingency)
        # Normalize by degrees of freedom for comparability
        return chi2 / max(dof, 1)
    except Exception:
        return 0.0


def _explain_drift(
    traces: List[Trace],
    position: int,
    window_size: int,
    chi2_score: float,
) -> DriftExplanation:
    """Generate ARM-based explanation for a detected drift."""
    # Get windows around drift point
    start1 = max(0, position - window_size)
    end1 = position
    start2 = position
    end2 = min(len(traces), position + window_size)

    window1 = traces[start1:end1]
    window2 = traces[start2:end2]

    if len(window1) < 10 or len(window2) < 10:
        return DriftExplanation(
            position=position,
            chi2_score=chi2_score,
            arm_score=None,
            temporal_changes=[],
            existential_changes=[],
            affected_activities=set(),
        )

    # Discover ARM matrices
    matrix1 = _discover_from_traces(window1, source=f"before_{position}")
    matrix2 = _discover_from_traces(window2, source=f"after_{position}")

    # Compute ARM drift score
    arm_score = compute_drift_score(matrix1, matrix2)

    # Extract specific changes
    temporal_changes = []
    existential_changes = []
    affected = set()

    common_acts = sorted(set(matrix1.activities) & set(matrix2.activities))
    for a in common_acts:
        for b in common_acts:
            cell1 = matrix1[a, b]
            cell2 = matrix2[a, b]

            if cell1.temporal != cell2.temporal:
                temporal_changes.append((a, b, cell1.temporal.name, cell2.temporal.name))
                affected.add(a)
                affected.add(b)

            if cell1.existential != cell2.existential:
                existential_changes.append((a, b, cell1.existential.name, cell2.existential.name))
                affected.add(a)
                affected.add(b)

    return DriftExplanation(
        position=position,
        chi2_score=chi2_score,
        arm_score=arm_score,
        temporal_changes=temporal_changes,
        existential_changes=existential_changes,
        affected_activities=affected,
    )


class HybridDriftDetector:
    """Hybrid drift detector using DFG chi-squared + ARM explanation.

    Detection: Chi-squared test on aggregated DFG counts between windows.
    This captures trace-level variation that ARM matrices miss.

    Explanation: ARM matrix diff shows what temporal/existential
    relationships changed, providing semantic understanding.
    """

    def __init__(
        self,
        window_size: int = 200,
        step_size: int = 10,
        prominence: float = 0.5,
        min_gap: Optional[int] = None,
        explain: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            window_size: Traces per window for DFG aggregation
            step_size: Sliding step between comparisons
            prominence: Minimum prominence for peak detection (default: 0.5)
            min_gap: Minimum gap between detections (default: window_size)
            explain: Generate ARM explanations for detected drifts
            verbose: Print diagnostic info
        """
        self.window_size = window_size
        self.step_size = step_size
        self.prominence = prominence
        self.min_gap = min_gap if min_gap is not None else window_size
        self.explain = explain
        self.verbose = verbose

    def detect(self, traces: List[Trace]) -> HybridDriftResult:
        """Detect concept drifts using hybrid approach.

        Args:
            traces: List of traces from parse_xes

        Returns:
            HybridDriftResult with detected drifts and explanations
        """
        n_traces = len(traces)

        if n_traces < 2 * self.window_size:
            return HybridDriftResult(
                drift_indices=[],
                chi2_scores=[],
                positions=[],
                explanations=[],
                window_size=self.window_size,
                prominence=self.prominence,
            )

        # Get all DF pairs from entire log
        all_dfg = _window_dfg(traces)
        all_pairs = sorted(all_dfg.keys())

        if self.verbose:
            print(f"Found {len(all_pairs)} unique DF pairs")

        # Compute chi-squared scores with sliding windows
        chi2_scores = []
        positions = []

        i = 0
        while i + 2 * self.window_size <= n_traces:
            dfg1 = _window_dfg(traces[i : i + self.window_size])
            dfg2 = _window_dfg(traces[i + self.window_size : i + 2 * self.window_size])

            chi2 = _compute_chi2(dfg1, dfg2, all_pairs)

            position = i + self.window_size
            positions.append(position)
            chi2_scores.append(chi2)

            if self.verbose and chi2 > 2.0:
                print(f"Position {position}: chi2={chi2:.2f}")

            i += self.step_size

        # Detect peaks in chi-squared signal
        chi2_array = np.array(chi2_scores)
        min_distance = max(1, self.min_gap // self.step_size)

        peaks, properties = find_peaks(
            chi2_array,
            prominence=self.prominence,
            distance=min_distance,
        )

        drift_indices = [positions[p] for p in peaks]

        if self.verbose:
            print(f"Detected {len(drift_indices)} drifts: {drift_indices}")

        # Generate explanations if requested
        explanations = []
        if self.explain:
            for p in peaks:
                pos = positions[p]
                chi2 = chi2_scores[p]
                explanation = _explain_drift(traces, pos, self.window_size, chi2)
                explanations.append(explanation)

        return HybridDriftResult(
            drift_indices=drift_indices,
            chi2_scores=chi2_scores,
            positions=positions,
            explanations=explanations,
            window_size=self.window_size,
            prominence=self.prominence,
        )

    def detect_from_log(self, log_path: str | Path) -> HybridDriftResult:
        """Detect drifts from XES log file."""
        traces = parse_xes(Path(log_path))
        return self.detect(traces)


def detect_drift_hybrid(
    log_path: str,
    window_size: int = 200,
    prominence: float = 0.5,
    step_size: int = 10,
) -> HybridDriftResult:
    """Convenience function for hybrid drift detection.

    Args:
        log_path: Path to XES event log
        window_size: Window size for DFG aggregation
        prominence: Peak prominence threshold
        step_size: Sliding window step

    Returns:
        HybridDriftResult with detections and explanations
    """
    detector = HybridDriftDetector(
        window_size=window_size,
        step_size=step_size,
        prominence=prominence,
    )
    return detector.detect_from_log(log_path)
