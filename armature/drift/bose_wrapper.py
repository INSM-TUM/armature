"""Bose S/N/A concept drift detection wrapper.

Implements the Bose drift detection method using causal footprint matrix features
(Sometimes/Never/Always follows) with sliding window statistical tests.

Reference: Bose et al. "Dealing With Concept Drifts in Process Mining"
DOI:10.1109/TNNLS.2013.2278313
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pm4py
import pm4py.util.xes_constants as xes
from pm4py.objects.log.obj import EventLog
from scipy import stats
from scipy.signal import find_peaks


@dataclass
class DriftDetectionResult:
    """Result from drift detection."""

    drift_indices: List[int]  # Trace indices where drift detected
    p_values: np.ndarray  # P-values for each trace position
    method: str  # Detection method used


class BoseDriftDetector:
    """Bose S/N/A concept drift detector.

    Implements drift detection using causal footprint matrix features
    (Sometimes/Never/Always follows) with sliding window statistical tests.

    Reference: Bose et al. "Dealing With Concept Drifts in Process Mining"
    DOI:10.1109/TNNLS.2013.2278313
    """

    def __init__(
        self,
        window_size: int = 100,
        measure: str = "j",  # "j" for J-Measure, "wc" for Window Count
        stat_test: str = "mu",  # "mu" for Mann-Whitney U, "ks" for Kolmogorov-Smirnov
        measure_window: Optional[int] = None,
        activity_key: str = xes.DEFAULT_NAME_KEY,
    ):
        """Initialize Bose drift detector.

        Args:
            window_size: Fixed window size for sliding window statistical testing
            measure: Feature measure to use ("j" or "wc")
            stat_test: Statistical test to use ("mu" or "ks")
            measure_window: Window size for measure extraction (defaults to avg trace length)
            activity_key: Key for activity name in event log
        """
        self.window_size = window_size
        self.measure = measure.lower()
        self.stat_test = stat_test.lower()
        self.measure_window = measure_window
        self.activity_key = activity_key

    def detect(self, log: EventLog) -> DriftDetectionResult:
        """Detect concept drifts in event log.

        Args:
            log: Event log to analyze

        Returns:
            DriftDetectionResult with detected drift indices and p-values
        """
        # Get activity names
        activities = _get_activity_names(log, self.activity_key)

        # Calculate p-values for all activity pairs
        pvals = np.zeros(len(log))

        for act1 in activities:
            for act2 in activities:
                # Extract measure time series
                if self.measure == "j":
                    m = _extract_j_measure(log, act1, act2, self.measure_window, self.activity_key)
                elif self.measure == "wc":
                    m = _extract_window_count(
                        log, act1, act2, self.measure_window, self.activity_key
                    )
                else:
                    raise ValueError(f"Invalid measure: {self.measure}")

                # Apply statistical test
                if self.stat_test == "ks":
                    pvals_ = _ks_test_sliding_window(m, self.window_size)
                elif self.stat_test in ["u", "mu"]:
                    pvals_ = _mann_whitney_sliding_window(m, self.window_size)
                else:
                    raise ValueError(f"Invalid stat test: {self.stat_test}")

                # Accumulate p-values
                pvals += pvals_

        # Average p-values across all activity pairs
        pvals = pvals / (len(activities) ** 2)

        # Detect peaks (drift points) using visual inspection
        drift_indices = _visual_inspection(pvals, trim=self.window_size)

        method = f"{self.measure.upper()}_{self.stat_test.upper()}"

        return DriftDetectionResult(
            drift_indices=drift_indices, p_values=pvals, method=method
        )


def detect_drift_bose(
    log_path: str,
    window_size: int = 100,
    measure: str = "j",
    stat_test: str = "mu",
) -> DriftDetectionResult:
    """Convenience function to detect drift from XES file path.

    Args:
        log_path: Path to XES event log file
        window_size: Window size for statistical testing
        measure: Feature measure ("j" or "wc")
        stat_test: Statistical test ("mu" or "ks")

    Returns:
        DriftDetectionResult with detected drifts
    """
    log = pm4py.read_xes(log_path, return_legacy_log_object=True)
    detector = BoseDriftDetector(window_size=window_size, measure=measure, stat_test=stat_test)
    return detector.detect(log)


# Helper functions ported from cd-soa/bose.py


def _get_activity_names(log: EventLog, activity_key: str = xes.DEFAULT_NAME_KEY) -> List[str]:
    """Extract unique activity names from event log."""
    return list(set(event[activity_key] for trace in log for event in trace))


def _get_causal_footprint(
    log: EventLog, activities: List[str] = None, activity_key: str = xes.DEFAULT_NAME_KEY
) -> np.ndarray:
    """Get the Causal Footprint Matrix of event log.

    Contains for every pair of activities, whether the second always (A),
    sometimes (S), or never (N) follows the other in a trace.

    Args:
        log: Event log
        activities: Activity list (if None, extracted from log)
        activity_key: Key for activity name

    Returns:
        Causal footprint matrix (S/N/A values)
    """
    if activities is None:
        activities = _get_activity_names(log, activity_key)

    # Initialize dictionary with % (not yet seen)
    d = {(act1, act2): "%" for act1 in activities for act2 in activities}

    for trace in log:
        # Track which activities we have seen in this trace
        seen = set()
        a_touched = set()  # Pairs where Always was updated

        for event in trace:
            name = event[activity_key]

            # Update follows relations
            for s in seen:
                valnow = d[(s, name)]
                if valnow in ["%", "A"]:
                    d[(s, name)] = "A"
                    a_touched.add((s, name))
                elif valnow == "N":
                    d[(s, name)] = "S"
                # Else value is S, stays S

            seen.add(name)

        # Demote A to S if activity occurred but relation wasn't touched
        for act1 in seen:
            for act2 in activities:
                if d[(act1, act2)] == "A" and (act1, act2) not in a_touched:
                    d[(act1, act2)] = "S"
                elif d[(act1, act2)] == "%":
                    d[(act1, act2)] = "N"

    # Convert to 2D matrix
    output = np.empty((len(activities), len(activities)), dtype='U1')
    for act1 in activities:
        i1 = activities.index(act1)
        for act2 in activities:
            i2 = activities.index(act2)
            output[i1][i2] = d[(act1, act2)] if d[(act1, act2)] != "%" else "N"

    return output


def _calculate_sf(
    log: EventLog,
    act1: str,
    act2: str,
    windowsize: int = None,
    activity_key: str = xes.DEFAULT_NAME_KEY,
) -> List[tuple]:
    """Calculate S and F sets for Window Count measure.

    Args:
        log: Event log
        act1: First activity
        act2: Second activity
        windowsize: Window size (defaults to avg trace length)
        activity_key: Key for activity name

    Returns:
        List of (S, F) tuples for each trace
    """
    if windowsize is None:
        windowsize = sum([len(t) for t in log]) // len(log)

    output = np.empty(len(log), dtype="O")

    for i, trace in enumerate(log):
        # Build S^{l,t}(act1) - multiset of subtraces starting with act1
        S = []
        for j, event in enumerate(trace):
            if event[activity_key] == act1:
                S.append([act[activity_key] for act in trace[j : j + windowsize]])

        # Build F^{l,t}(act1, act2) - subtraces where act2 eventually follows act1
        F = [s for s in S if act2 in s[1:]]  # Exclude first activity

        output[i] = (S, F)

    return output


def _extract_window_count(
    log: EventLog,
    act1: str,
    act2: str,
    windowsize: int = None,
    activity_key: str = xes.DEFAULT_NAME_KEY,
) -> np.ndarray:
    """Extract Window Count time series.

    Args:
        log: Event log
        act1: First activity
        act2: Second activity
        windowsize: Window size
        activity_key: Key for activity name

    Returns:
        Window count values per trace
    """
    sf = _calculate_sf(log, act1, act2, windowsize, activity_key)
    return np.array([len(f) for s, f in sf])


def _extract_j_measure(
    log: EventLog,
    act1: str,
    act2: str,
    windowsize: int = None,
    activity_key: str = xes.DEFAULT_NAME_KEY,
) -> np.ndarray:
    """Extract J-Measure time series.

    Args:
        log: Event log
        act1: First activity
        act2: Second activity
        windowsize: Window size
        activity_key: Key for activity name

    Returns:
        J-Measure values per trace
    """
    output = np.empty(len(log))
    sf = _calculate_sf(log, act1, act2, windowsize, activity_key)

    for i, trace in enumerate(log):
        S, F = sf[i]

        # p(a,b) - probability that b follows a in window
        p_ab = len(F) / len(S) if len(S) != 0 and len(F) != 0 else 0

        # p(a) and p(b) - probabilities in trace
        p_a = len([act for act in trace if act[activity_key] == act1]) / len(trace)
        p_b = len([act for act in trace if act[activity_key] == act2]) / len(trace)

        # Cross entropy term
        ct = p_ab * (0 if p_ab == 0 or p_b == 0 else math.log2(p_ab / p_b)) + (1 - p_ab) * (
            0 if (1 - p_ab) == 0 or (1 - p_b) == 0 else math.log2((1 - p_ab) / (1 - p_b))
        )

        output[i] = p_a * ct

    return output


def _ks_test_sliding_window(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply Kolmogorov-Smirnov test with sliding windows.

    Args:
        signal: Time series signal
        window_size: Window size

    Returns:
        P-values for each position
    """
    pvals = np.ones(len(signal))

    for i in range(len(signal) - (2 * window_size)):
        window1 = signal[i : i + window_size]
        window2 = signal[i + window_size : i + (2 * window_size)]
        ks = stats.ks_2samp(window1, window2)
        pvals[i + window_size] = ks.pvalue

    return pvals


def _mann_whitney_sliding_window(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply Mann-Whitney U test with sliding windows.

    Args:
        signal: Time series signal
        window_size: Window size

    Returns:
        P-values for each position
    """
    pvals = np.ones(len(signal))

    for i in range(len(signal) - (2 * window_size)):
        window1 = signal[i : i + window_size]
        window2 = signal[i + window_size : i + (2 * window_size)]
        u = stats.mannwhitneyu(window1, window2)
        pvals[i + window_size] = u.pvalue

    return pvals


def _visual_inspection(signal: np.ndarray, trim: int = 0) -> List[int]:
    """Automated visual inspection using peak detection.

    Args:
        signal: P-value signal
        trim: Number of values to trim from edges

    Returns:
        List of detected change point indices
    """
    # Detect peaks in inverted signal (low p-values = high peaks)
    peaks = find_peaks(-signal[trim : len(signal) - trim], width=80, prominence=0.1)[0]
    return [x + trim for x in peaks]
