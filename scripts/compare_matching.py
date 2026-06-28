"""Compare greedy vs LP matching algorithms on test cases.

Tests whether our greedy bipartite matching produces different F1 scores
compared to cdrift's LP-based optimal matching.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.drift.evaluation import compute_metrics as greedy_metrics

# Import LP implementation directly (avoid cdrift NumPy issue)
from typing import List, Tuple
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD


def assign_changepoints_lp(
    detected: List[int], actual: List[int], lag: int
) -> List[Tuple[int, int]]:
    """LP-based assignment (copied from cdrift to avoid NumPy 2.x issues)."""
    if not detected or not actual:
        return []

    def buildProb_NoObjective(sense):
        prob = LpProblem("Changepoint_Assignment", sense)
        vars = LpVariable.dicts("x", (detected, actual), 0, 1, LpBinary)
        x = {(dc, ap): vars[dc][ap] for dc in detected for ap in actual}

        # At most one detected per actual
        for ap in actual:
            prob += (lpSum(x[dp, ap] for dp in detected) <= 1, f"One_Per_Actual_{ap}")

        # At most one actual per detected
        for dp in detected:
            prob += (lpSum(x[dp, ap] for ap in actual) <= 1, f"One_Per_Detected_{dp}")

        # Within lag window
        for dp in detected:
            for ap in actual:
                prob += (x[dp, ap] * abs(dp - ap) <= lag, f"Lag_{dp}_{ap}")

        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    # Step 1: Maximize number of assignments
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (lpSum(prob1_vars[dp, ap] for dp in detected for ap in actual), "Max_Assignments")
    prob1.solve(solver)

    num_tp = len([(dp, ap) for dp in detected for ap in actual if prob1_vars[dp, ap].varValue == 1])

    # Step 2: Minimize squared distance with fixed TP count
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(prob2_vars[dp, ap] * pow(dp - ap, 2) for dp in detected for ap in actual),
        "Min_Squared_Distance",
    )
    prob2 += (
        lpSum(prob2_vars[dp, ap] for dp in detected for ap in actual) == num_tp,
        "Fixed_TP_Count",
    )
    prob2.solve(solver)

    return [(dp, ap) for dp in detected for ap in actual if prob2_vars[dp, ap].varValue == 1]


def lp_f1(detected: List[int], actual: List[int], lag: int) -> float:
    """Compute F1 using LP matching."""
    if not detected and not actual:
        return 1.0
    if not detected or not actual:
        return 0.0

    assignments = assign_changepoints_lp(detected, actual, lag)
    tp = len(assignments)
    fp = len(detected) - tp
    fn = len(actual) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(actual) if actual else 0.0

    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def compare_matching(detected, actual, lag, description):
    """Compare greedy vs LP matching on a single test case."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"{'='*60}")
    print(f"Detected: {detected}")
    print(f"Actual:   {actual}")
    print(f"Lag:      {lag}")

    # Our greedy approach
    greedy = greedy_metrics(detected, actual, lag)

    # LP approach
    lp = lp_f1(detected, actual, lag)

    print(
        f"\nGreedy F1: {greedy['f1']:.4f} (TP={greedy['tp']}, FP={greedy['fp']}, FN={greedy['fn']})"
    )
    print(f"LP F1:     {lp:.4f}")

    diff = abs(greedy["f1"] - lp)
    if diff > 0.001:
        print(f"⚠️  DIFFERENCE: {diff:.4f}")
        return True
    else:
        print("✓ MATCH")
        return False


def main():
    """Run comparison tests."""
    print("Comparing Greedy vs LP Bipartite Matching")
    print("==========================================")

    differences = []

    # Test 1: Perfect detection (should match)
    diff = compare_matching(
        detected=[1101, 2399, 3789, 4789],
        actual=[1099, 2399, 3789, 4789],
        lag=200,
        description="Perfect detection (off by 2 on first)",
    )
    differences.append(("Perfect detection", diff))

    # Test 2: Example from cdrift docs (greedy may fail)
    # Expected LP assignment: (934→1000), (1050→1149), (2100→2000)
    # Greedy might assign: (934→1000), (1050→1000), (2100→2000)
    diff = compare_matching(
        detected=[1050, 934, 2100],
        actual=[1000, 1149, 2000],
        lag=200,
        description="cdrift example (greedy should fail)",
    )
    differences.append(("cdrift example", diff))

    # Test 3: Multiple detections for same drift
    diff = compare_matching(
        detected=[1000, 1001, 2000],
        actual=[1000, 2000],
        lag=200,
        description="Duplicate detection (should match)",
    )
    differences.append(("Duplicate detection", diff))

    # Test 4: No detections
    diff = compare_matching(
        detected=[], actual=[1000, 2000], lag=200, description="No detections (should both be 0)"
    )
    differences.append(("No detections", diff))

    # Test 5: No actual drifts
    diff = compare_matching(
        detected=[1000, 2000], actual=[], lag=200, description="No actual drifts (should both be 0)"
    )
    differences.append(("No actual drifts", diff))

    # Test 6: Overlapping ranges - critical test
    # If detected=[500, 600] and actual=[550, 650], lag=100
    # Both can match either drift
    # LP should optimize: (500→550), (600→650) - both TP
    # Greedy might: (500→550), (600→550) - only 1 TP
    diff = compare_matching(
        detected=[500, 600],
        actual=[550, 650],
        lag=100,
        description="Overlapping ranges (greedy may be suboptimal)",
    )
    differences.append(("Overlapping ranges", diff))

    # Test 7: Three-way ambiguity
    diff = compare_matching(
        detected=[1000, 1100, 1200],
        actual=[1050, 1150],
        lag=150,
        description="Three-way ambiguity (all can match both)",
    )
    differences.append(("Three-way ambiguity", diff))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    different = sum(1 for _, d in differences if d)
    total = len(differences)

    print(f"Total tests: {total}")
    print(f"Matching: {total - different}")
    print(f"Different: {different}")

    if different > 0:
        print("\n⚠️  GREEDY PRODUCES DIFFERENT F1 SCORES")
        print("Recommendation: Implement LP matching for fair comparison")
    else:
        print("\n✓ GREEDY MATCHES LP ON ALL TEST CASES")
        print("Current evaluation is valid")

    print("\nTests with differences:")
    for name, diff in differences:
        if diff:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
