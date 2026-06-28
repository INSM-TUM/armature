"""Adapter for cdrift-evaluation metrics.

Provides patched functions to work with NumPy 2.x (np.nan vs np.NaN).
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add cdrift to path
cdrift_path = Path("/home/choky/kerstin/cdrift-evaluation")
sys.path.insert(0, str(cdrift_path))

# Import pulp first
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

# Import needed components from cdrift
import numpy as np


def assign_changepoints(
    detected_changepoints: List[int], actual_changepoints: List[int], lag_window: int = 200
) -> List[Tuple[int, int]]:
    """Assigns detected changepoints to actual changepoints using LP.

    Copied and patched from cdrift/evaluation.py to work with NumPy 2.x.
    Uses same two-phase optimization:
    1. Maximize number of assignments
    2. Minimize sum of squared distances

    Args:
        detected_changepoints: List of locations of detected changepoints.
        actual_changepoints: List of locations of actual changepoints.
        lag_window: Max distance for valid assignment (default: 200)

    Returns:
        List of tuples (detected, actual) representing optimal assignments
    """

    def buildProb_NoObjective(sense):
        prob = LpProblem("Changepoint_Assignment", sense)
        vars = LpVariable.dicts("x", (detected_changepoints, actual_changepoints), 0, 1, LpBinary)
        x = {(dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in actual_changepoints}

        # Only assign at most one changepoint to each actual changepoint
        for ap in actual_changepoints:
            prob += (
                lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}",
            )
        # Each detected changepoint is assigned to at most one actual changepoint
        for dp in detected_changepoints:
            prob += (
                lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}",
            )
        # Distance between chosen changepoints must be within lag window
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                prob += (
                    x[dp, ap] * abs(dp - ap) <= lag_window,
                    f"Distance_Within_Lag_Window : {dp}_{ap}",
                )
        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    # Multi-objective optimization: First maximize number of assignments
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (
        lpSum(prob1_vars[dp, ap] for dp in detected_changepoints for ap in actual_changepoints),
        "Maximize number of assignments",
    )
    prob1.solve(solver)

    # Calculate number of TP
    num_tp = len(
        [
            (dp, ap)
            for dp in detected_changepoints
            for ap in actual_changepoints
            if prob1_vars[dp, ap].varValue == 1
        ]
    )

    # Now minimize squared distance with fixed TP count
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(
            prob2_vars[dp, ap] * pow(dp - ap, 2)
            for dp in detected_changepoints
            for ap in actual_changepoints
        ),
        "Squared_Distances",
    )

    prob2 += (
        lpSum(prob2_vars[dp, ap] for dp in detected_changepoints for ap in actual_changepoints)
        == num_tp,
        "Maximize Number of Assignments",
    )
    prob2.solve(solver)

    return [
        (dp, ap)
        for dp in detected_changepoints
        for ap in actual_changepoints
        if prob2_vars[dp, ap].varValue == 1
    ]


def F1_Score(
    detected: List[int],
    known: List[int],
    lag: int,
    zero_division: float = float("nan"),
    verbose: bool = False,
    count_duplicate_detections: bool = True,
) -> float:
    """Calculates the F1 Score for a Changepoint Detection Result.

    Copied and patched from cdrift/evaluation.py to work with NumPy 2.x.

    Args:
        detected: List of detected changepoint locations.
        known: Ground truth changepoint locations.
        lag: Max distance for true positive.
        zero_division: Return value on zero division (default: NaN).
        verbose: Print warnings on zero division.
        count_duplicate_detections: Count extra detections in lag window as FP.

    Returns:
        F1-Score
    """

    assignments = assign_changepoints(detected, known, lag_window=lag)
    TP = len(assignments)

    if count_duplicate_detections:
        FP = len(detected) - TP
    else:
        true_positive_candidates = [
            d for d in detected if any((k - lag <= d and d <= k + lag) for k in known)
        ]
        FP = len(detected) - len(true_positive_candidates)

    try:
        precision = TP / (TP + FP)
        recall = TP / len(known)
        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score
    except ZeroDivisionError:
        if verbose:
            print("Calculation of F1-Score resulted in division by 0.")
        return zero_division


def get_avg_lag(
    detected_changepoints: List[int], actual_changepoints: List[int], lag: int = 200
) -> float:
    """Calculates the average lag between detected and actual changepoints.

    Copied and patched from cdrift/evaluation.py to work with NumPy 2.x.

    Args:
        detected_changepoints: Locations of detected changepoints
        actual_changepoints: Locations of actual changepoints
        lag: Max distance for valid assignment

    Returns:
        Average distance for matched changepoints
    """
    assignments = assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag)
    avg_lag = 0
    for dc, ap in assignments:
        avg_lag += abs(dc - ap)

    try:
        return avg_lag / len(assignments)
    except ZeroDivisionError:
        return float("nan")
