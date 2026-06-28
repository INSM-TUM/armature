"""Evaluation metrics for drift detection with bipartite matching.

Provides F1-score, precision, recall computation using bipartite matching
between detected and actual changepoints with lag tolerance.
"""

from typing import List, Tuple


def bipartite_match_changepoints(
    detected: List[int], actual: List[int], lag: int
) -> Tuple[int, int]:
    """Match detected changepoints to actual using bipartite matching.

    Uses greedy bipartite matching algorithm: sort all valid pairs by distance,
    assign closest pairs first, ensuring each actual matches at most one detected.

    Args:
        detected: Detected changepoint indices
        actual: Ground truth changepoint indices
        lag: Lag tolerance window (traces)

    Returns:
        Tuple of (true_positives, false_positives)
    """
    if not detected:
        return 0, 0

    if not actual:
        return 0, len(detected)

    # Build all valid pairs (within lag) with distances
    pairs = []
    for d in detected:
        for a in actual:
            distance = abs(d - a)
            if distance <= lag:
                pairs.append((distance, d, a))

    # Sort by distance (greedy: assign closest pairs first)
    pairs.sort()

    # Bipartite matching
    matched_detected = set()
    matched_actual = set()
    tp = 0

    for dist, d, a in pairs:
        if d not in matched_detected and a not in matched_actual:
            matched_detected.add(d)
            matched_actual.add(a)
            tp += 1

    fp = len(detected) - tp

    return tp, fp


def compute_metrics(detected: List[int], actual: List[int], lag: int = 200) -> dict:
    """Compute F1-score, precision, recall with lag tolerance.

    Args:
        detected: Detected changepoint indices
        actual: Ground truth changepoint indices
        lag: Lag tolerance window (default: 200 traces, cdrift standard)

    Returns:
        Dict with: tp, fp, fn, precision, recall, f1
    """
    tp, fp = bipartite_match_changepoints(detected, actual, lag)
    fn = len(actual) - tp

    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_average_lag(detected: List[int], actual: List[int], lag: int = 200) -> float:
    """Compute average lag for matched changepoint pairs.

    Args:
        detected: Detected changepoint indices
        actual: Ground truth changepoint indices
        lag: Lag tolerance window

    Returns:
        Average absolute distance for matched pairs, 0.0 if no matches
    """
    if not detected or not actual:
        return 0.0

    # Build all valid pairs (within lag) with distances
    pairs = []
    for d in detected:
        for a in actual:
            distance = abs(d - a)
            if distance <= lag:
                pairs.append((distance, d, a))

    # Sort by distance
    pairs.sort()

    # Bipartite matching
    matched_detected = set()
    matched_actual = set()
    lags = []

    for dist, d, a in pairs:
        if d not in matched_detected and a not in matched_actual:
            matched_detected.add(d)
            matched_actual.add(a)
            lags.append(dist)

    return sum(lags) / len(lags) if lags else 0.0
