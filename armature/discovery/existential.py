"""Existential dependency extraction (co-occurrence patterns)."""

from __future__ import annotations

from armature.core.dependencies import ExistentialDependency
from armature.discovery.models import Trace


def extract_existential_dependencies(
    traces: list[Trace],
    threshold: float = 1.0,
) -> dict[tuple[str, str], ExistentialDependency]:
    """Extract existential dependencies between activities.

    Algorithm per DEPENDENCE_TYPES.md:
    For each activity pair (A, B), count traces where:
    - both A and B occur
    - only A occurs (B does not)
    - only B occurs (A does not)
    - neither A nor B occur

    Classification rules (in order, with threshold support):
    1. EQUIVALENCE: count_only_a/total < (1-threshold) AND count_only_b/total < (1-threshold) AND count_both > 0
    2. IMPLICATION A=>B: count_only_a/total < (1-threshold) AND count_both > 0 (and not EQUIVALENCE)
    3. IMPLICATION B=>A: count_only_b/total < (1-threshold) AND count_both > 0 (and not EQUIVALENCE)
    4. NEGATED_EQUIVALENCE: count_both/total < (1-threshold) AND count_neither/total < (1-threshold) AND count_only_a > 0 AND count_only_b > 0
    5. NAND: count_both/total < (1-threshold)
    6. OR: count_both > 0 AND (count_only_a > 0 OR count_only_b > 0)
    7. INDEPENDENCE: fallback (all combinations present)

    Args:
        traces: List of process traces
        threshold: Minimum fraction of traces that must satisfy the pattern (0.0-1.0).
                   Default 1.0 means pattern must hold in 100% of traces (deterministic).
                   Lower values allow noise/exceptions.

    Returns:
        Dict mapping (source, target) -> ExistentialDependency
    """
    # Collect all unique activities
    all_activities = set()
    for trace in traces:
        for event in trace.events:
            all_activities.add(event.activity)

    all_activities_list = sorted(all_activities)  # Deterministic ordering

    # Build occurrence matrix: which activities appear in each trace
    # Store as lists to preserve repetition counts for self-loop detection
    trace_occurrences: list[list[str]] = []
    for trace in traces:
        trace_activities_list = [event.activity for event in trace.events]
        trace_occurrences.append(trace_activities_list)

    # For each activity pair, compute counts
    result: dict[tuple[str, str], ExistentialDependency] = {}

    for activity_a in all_activities_list:
        for activity_b in all_activities_list:
            pair = (activity_a, activity_b)

            # Handle diagonal cells (self-loops) specially
            if activity_a == activity_b:
                # Check if activity ever repeats in any trace
                has_repetition = False
                for trace_activities_list in trace_occurrences:
                    # Count occurrences in this trace
                    count = sum(1 for act in trace_activities_list if act == activity_a)
                    if count > 1:
                        has_repetition = True
                        break

                if has_repetition:
                    # Self-loop: second occurrence implies first occurrence
                    result[pair] = ExistentialDependency.IMPLICATION
                # Otherwise skip (no relationship)
                continue

            # Count traces for each combination
            count_both = 0
            count_only_a = 0
            count_only_b = 0
            count_neither = 0

            for trace_activities_list in trace_occurrences:
                trace_activities_set = set(trace_activities_list)
                has_a = activity_a in trace_activities_set
                has_b = activity_b in trace_activities_set

                if has_a and has_b:
                    count_both += 1
                elif has_a and not has_b:
                    count_only_a += 1
                elif not has_a and has_b:
                    count_only_b += 1
                else:  # not has_a and not has_b
                    count_neither += 1

            # Apply classification rules with threshold support
            pair = (activity_a, activity_b)
            total_traces = len(trace_occurrences)

            # Calculate ratios for threshold checking
            # Threshold represents minimum fraction that must satisfy the pattern
            # E.g., threshold=0.9 means 90% of traces must satisfy the pattern
            # Violation tolerance = 1 - threshold (e.g., 0.1 = 10% can violate)
            
            # Rule 1: EQUIVALENCE - always together when they occur (within threshold)
            # For equivalence: both count_only_a and count_only_b should be rare (below tolerance)
            if count_only_a <= (1 - threshold) * total_traces and count_only_b <= (1 - threshold) * total_traces and count_both > 0:
                result[pair] = ExistentialDependency.EQUIVALENCE

            # Rule 2: IMPLICATION A=>B - whenever A occurs, B also occurs (within threshold)
            # count_only_a should be rare (below tolerance) means: A usually has B
            elif count_only_a <= (1 - threshold) * total_traces and count_both > 0:
                result[pair] = ExistentialDependency.IMPLICATION  # A => B

            # Rule 3: IMPLICATION B=>A - whenever B occurs, A also occurs (within threshold)
            # count_only_b should be rare (below tolerance) means: B usually has A
            elif count_only_b <= (1 - threshold) * total_traces and count_both > 0:
                result[pair] = ExistentialDependency.IMPLICATION_BACKWARD  # A <= B (meaning B=>A)

            # Rule 4: NEGATED_EQUIVALENCE - exactly one per trace, never both, never neither (within threshold)
            elif count_both <= (1 - threshold) * total_traces and count_neither <= (1 - threshold) * total_traces and count_only_a > 0 and count_only_b > 0:
                result[pair] = ExistentialDependency.NEGATED_EQUIVALENCE

            # Rule 5: NAND - never together (within threshold).
            # NEGATED_EQUIVALENCE already failed, so count_neither > tolerance is guaranteed,
            # making any additional disjunction vacuously true.
            elif count_both <= (1 - threshold) * total_traces:
                result[pair] = ExistentialDependency.NAND

            # Rule 6: INDEPENDENCE - all four combinations present (most independent)
            elif count_both > 0 and count_only_a > 0 and count_only_b > 0 and count_neither > 0:
                result[pair] = ExistentialDependency.INDEPENDENCE

            # Rule 7: OR - sometimes together, sometimes not (but not all combinations)
            elif count_both > 0 and (count_only_a > 0 or count_only_b > 0):
                result[pair] = ExistentialDependency.OR

            # Rule 8: INDEPENDENCE - fallback (shouldn't reach here normally)
            else:
                result[pair] = ExistentialDependency.INDEPENDENCE

    return result
