"""Temporal dependency extraction with loop awareness."""

from __future__ import annotations

from armature.core.dependencies import TemporalDependency
from armature.discovery.dfg import DFG
from armature.discovery.models import Trace
from armature.discovery.scc import LoopContext


def has_self_loop(dfg: DFG, activity: str) -> bool:
    """Check if activity has self-edge in DFG (actually repeats in some trace)."""
    return activity in dfg.edges and activity in dfg.edges.get(activity, {})


def is_direct_relationship(dfg: DFG, source: str, target: str) -> bool:
    """Check if source->target edge exists in DFG (direct succession observed)."""
    return source in dfg.edges and target in dfg.edges.get(source, {})


def has_multiple_paths(dfg: DFG, source: str, target: str) -> bool:
    """Check if multiple paths exist from source to target.

    Returns True if both:
    1. Direct edge source→target exists in DFG
    2. Alternative path source→...→target exists (via intermediate nodes)

    Algorithm: BFS from source, track if target reachable via paths other than direct edge.

    Args:
        dfg: Directly-follows graph with edges
        source: Source activity
        target: Target activity

    Returns:
        True if alternate path found (multiple paths exist), False otherwise
    """
    # First check if direct edge exists
    if not is_direct_relationship(dfg, source, target):
        return False

    # Build adjacency dict excluding direct edge to target
    from collections import deque

    visited = set()
    queue = deque([source])
    visited.add(source)

    # BFS excluding direct edge source->target
    while queue:
        current = queue.popleft()

        if current not in dfg.edges:
            continue

        for successor in dfg.edges[current]:
            # Skip direct edge to target
            if current == source and successor == target:
                continue

            # If we reach target via alternate path, multiple paths exist
            if successor == target:
                return True

            # Continue BFS
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    return False


def is_deterministic_succession(
    ordering_data: dict[tuple[str, str], dict[str, bool]],
    source: str,
    target: str,
    dfg: DFG,
) -> bool:
    """Check if source->target succession is deterministic (no parallelism).

    A succession is non-deterministic if:
    - Source has multiple direct successors in DFG
    - Those successors can occur in different orders (show INDEPENDENCE)

    Example: If a->b and a->c both exist in DFG, and b/c show both orderings,
    then neither a->b nor a->c is deterministic DIRECT.

    Args:
        ordering_data: Dict mapping (A, B) -> {a_before_b, b_before_a}
        source: Source activity
        target: Target activity
        dfg: Directly-follows graph

    Returns:
        True if succession is deterministic, False if parallel activities exist
    """
    # Get all direct successors of source from DFG
    if source not in dfg.edges:
        return True  # No edges, treat as deterministic if DFG edge exists

    successors = set(dfg.edges[source].keys())

    # If only one successor, it's deterministic
    if len(successors) <= 1:
        return True

    # If target not in successors, not a DFG edge (shouldn't happen)
    if target not in successors:
        return False

    # Check if target and other successors show parallelism
    # If any other successor has both orderings with target, it's non-deterministic
    for other in successors:
        if other == target:
            continue

        # Check if target and other show INDEPENDENCE (both orderings)
        pair = (target, other)
        data = ordering_data.get(pair, {})

        a_before_b = data.get("a_before_b", False)
        b_before_a = data.get("b_before_a", False)

        # Both orderings observed = parallel = non-deterministic
        if a_before_b and b_before_a:
            return False

    return True


def extract_temporal_dependencies(
    traces: list[Trace], dfg: DFG, loop_ctx: LoopContext, threshold: float = 1.0
) -> dict[tuple[str, str], TemporalDependency]:
    """Extract temporal dependencies between activities with loop awareness.

    Algorithm per DEPENDENCE_TYPES.md:
    1. Check co-occurrence: If A and B never appear in same trace -> NO_ORDERING
    2. Track orderings: A->B and/or B->A across traces where both occur
    3. Classify:
       - INDEPENDENCE: Both orderings observed (A->B and B->A)
       - DIRECT: A immediately precedes B in at least one trace
       - EVENTUAL: A before B (not direct), A and B in same SCC
       - TRUE_EVENTUAL: A before B (not direct), A and B NOT in same SCC

    Args:
        traces: List of process traces
        dfg: Directly-follows graph with edges
        loop_ctx: SCC membership mapping (activity -> SCC index or None)
        threshold: Minimum fraction of traces that must satisfy patterns (0.0-1.0).
                   Default 1.0 means patterns must hold in 100% of traces (deterministic).
                   Lower values allow noise/exceptions:
                   - threshold=0.9: 90% must show pattern (10% noise allowed)
                   - threshold=0.8: 80% must show pattern (20% noise allowed)
                   For INDEPENDENCE: both ratios must be > (1 - threshold)
                   For directional patterns: ratio must be >= threshold

    Returns:
        Dict mapping (source, target) -> TemporalDependency
    """
    # Build co-occurrence map: which activities appear together in traces
    cooccurrence: dict[tuple[str, str], bool] = {}
    cooccurrence_count: dict[tuple[str, str], int] = {}  # Count traces where both occur
    # Track orderings: (A, B) -> {a_before_b, b_before_a}
    ordering_data: dict[tuple[str, str], dict[str, bool]] = {}

    # Get all unique activities
    all_activities = list(dfg.activities)

    # Initialize tracking for all pairs
    for i, activity_a in enumerate(all_activities):
        for activity_b in all_activities:
            pair = (activity_a, activity_b)
            cooccurrence[pair] = False
            cooccurrence_count[pair] = 0
            ordering_data[pair] = {
                "a_before_b": False,
                "b_before_a": False,
            }

    # Process each trace to gather ordering data
    # Track per-trace orderings to detect independence
    # Independence requires orderings in DIFFERENT traces
    trace_orderings: dict[tuple[str, str], dict[str, int]] = {}
    for pair in cooccurrence:
        trace_orderings[pair] = {
            "a_only_before_b_count": 0,  # Traces where ONLY A->B
            "b_only_before_a_count": 0,  # Traces where ONLY B->A
        }

    for trace in traces:
        if not trace.events:
            continue

        # Get activities in this trace
        trace_activities = [event.activity for event in trace.events]
        trace_activities_set = set(trace_activities)

        # Mark co-occurrence for all pairs in this trace
        for activity_a in trace_activities_set:
            for activity_b in trace_activities_set:
                cooccurrence[(activity_a, activity_b)] = True
                cooccurrence_count[(activity_a, activity_b)] += 1

        # Track orderings within this trace
        trace_level_orderings: dict[tuple[str, str], dict[str, bool]] = {}
        for i, event_a in enumerate(trace.events):
            activity_a = event_a.activity

            for j, event_b in enumerate(trace.events):
                activity_b = event_b.activity
                pair = (activity_a, activity_b)

                if pair not in trace_level_orderings:
                    trace_level_orderings[pair] = {
                        "a_before_b": False,
                        "b_before_a": False,
                    }

                if i < j:
                    # A before B in this trace
                    ordering_data[pair]["a_before_b"] = True
                    trace_level_orderings[pair]["a_before_b"] = True

                elif i > j:
                    # B before A in this trace
                    ordering_data[pair]["b_before_a"] = True
                    trace_level_orderings[pair]["b_before_a"] = True

        # Count traces with exclusive ordering directions
        for pair, orderings in trace_level_orderings.items():
            has_a_before_b = orderings["a_before_b"]
            has_b_before_a = orderings["b_before_a"]

            # Only count if exclusive (not both in same trace)
            if has_a_before_b and not has_b_before_a:
                trace_orderings[pair]["a_only_before_b_count"] += 1
            if has_b_before_a and not has_a_before_b:
                trace_orderings[pair]["b_only_before_a_count"] += 1

    # Classify temporal dependencies
    result: dict[tuple[str, str], TemporalDependency] = {}

    for pair in cooccurrence:
        activity_a, activity_b = pair

        # Handle diagonal cells (self-loops) separately
        if activity_a == activity_b:
            # Self-loop: only set if activity actually repeats in some trace
            if has_self_loop(dfg, activity_a):
                # Self-loop: activity repeats — first instance directly follows second
                # Structured non-nested loops assume single-entry-single-exit, so
                # constructs like <a,a,b,a> (nested loop) are not possible.
                result[pair] = TemporalDependency.DIRECT
            # If no self-loop, don't add to result dict (leave as default)
            continue

        # Step 1: Check co-occurrence
        if not cooccurrence[pair]:
            result[pair] = TemporalDependency.NO_ORDERING
            continue

        # Step 2: Check orderings
        data = ordering_data[pair]
        a_before_b = data["a_before_b"]
        b_before_a = data["b_before_a"]

        # Check trace-level orderings for independence
        trace_data = trace_orderings[pair]
        a_only_traces = trace_data["a_only_before_b_count"]
        b_only_traces = trace_data["b_only_before_a_count"]
        
        # Step 3: Classify
        # Priority order:
        # 1. Check for bidirectional DFG edges (indicates INDEPENDENCE)
        # 2. DIRECT (single DFG edge with deterministic succession)
        # 3. If both orderings exist but neither is dominant → INDEPENDENCE
        # 4. EVENTUAL/TRUE_EVENTUAL (non-direct orderings with threshold check)
        
        # Calculate ratios for threshold check
        cooccur_count = cooccurrence_count[pair]
        ratio_a = a_only_traces / cooccur_count if cooccur_count > 0 else 0
        ratio_b = b_only_traces / cooccur_count if cooccur_count > 0 else 0

        # Check if both DFG edges exist (bidirectional)
        has_forward_edge = is_direct_relationship(dfg, activity_a, activity_b)
        has_backward_edge = is_direct_relationship(dfg, activity_b, activity_a)
        
        # If BOTH DFG edges exist, need to determine which (if any) is primary
        # Check determinism for both directions to see if one is clearly primary
        if has_forward_edge and has_backward_edge:
            # Special case: if both ratios are 0, it means both orderings appear in same traces (loop)
            # In this case, prefer the forward edge as primary
            if ratio_a == 0 and ratio_b == 0:
                # Both orderings in same traces (loop) - forward edge is primary
                # Fall through to check forward edge determinism
                pass
            else:
                # Check forward direction (A->B) determinism
                forward_deterministic = is_deterministic_succession(
                    ordering_data, activity_a, activity_b, dfg
                )
                forward_parallel = has_multiple_paths(dfg, activity_a, activity_b)
                
                # Check backward direction (B->A) determinism
                backward_deterministic = is_deterministic_succession(
                    ordering_data, activity_b, activity_a, dfg
                )
                backward_parallel = has_multiple_paths(dfg, activity_b, activity_a)
                
                # If BOTH directions are deterministic (true bidirectional loop)
                # OR if NEITHER is deterministic (parallel paths)
                # → Use ratio to determine dominance
                if (forward_deterministic and backward_deterministic) or \
                   (not forward_deterministic and not backward_deterministic):
                    a_is_dominant = ratio_a >= threshold
                    b_is_dominant = ratio_b >= threshold
                    
                    # If neither meets threshold, or both do → INDEPENDENCE
                    if (not a_is_dominant and not b_is_dominant) or \
                       (a_is_dominant and b_is_dominant):
                        result[pair] = TemporalDependency.INDEPENDENCE
                        continue
                    
                    # One direction is dominant by ratio
                    # Fall through to check which direction below
                
                # If only forward is deterministic → check forward direction below
                # If only backward is deterministic → check backward direction below
                # Fall through to normal edge checking

        # Check DIRECT forward (A->B)
        if has_forward_edge:
            # DFG edge exists - check both determinism AND parallel paths
            is_deterministic = is_deterministic_succession(
                ordering_data, activity_a, activity_b, dfg
            )
            has_parallel = has_multiple_paths(dfg, activity_a, activity_b)

            if is_deterministic and not has_parallel:
                # Truly direct - deterministic and no alternate paths
                result[pair] = TemporalDependency.DIRECT
            else:
                # DFG edge exists but non-deterministic OR parallel paths
                # CAN be direct (edge exists) so it's EVENTUAL not TRUE_EVENTUAL
                result[pair] = TemporalDependency.EVENTUAL
            continue
        
        # Check DIRECT backward (B->A)
        if has_backward_edge:
            # Backward DFG edge exists: B→A (for pair (A,B))
            # Check determinism and parallel paths for backward direction
            is_deterministic = is_deterministic_succession(
                ordering_data, activity_b, activity_a, dfg
            )
            has_parallel = has_multiple_paths(dfg, activity_b, activity_a)

            if is_deterministic and not has_parallel:
                # Direct backward succession
                result[pair] = TemporalDependency.DIRECT_BACKWARD
            else:
                # DFG backward edge exists but non-deterministic OR parallel
                # CAN be direct backward (edge exists) so it's EVENTUAL_BACKWARD not TRUE_EVENTUAL_BACKWARD
                result[pair] = TemporalDependency.EVENTUAL_BACKWARD
            continue
        
        # No DFG edges exist - check if both orderings are present without dominance
        # Check if either direction meets the threshold (dominant pattern)
        a_is_dominant = ratio_a >= threshold
        b_is_dominant = ratio_b >= threshold
        
        # If both orderings exist but neither meets threshold → INDEPENDENCE
        # (neither ordering is strong enough to be called a directional pattern)
        if a_before_b and b_before_a and not a_is_dominant and not b_is_dominant:
            result[pair] = TemporalDependency.INDEPENDENCE
            continue
        
        # No DFG edges - check eventual orderings with threshold
        if a_before_b and ratio_a >= threshold:
            # Only A before B with sufficient ratio - NO DFG edge (already handled above)
            # Activities never consecutive, only eventual
            scc_a = loop_ctx.get(activity_a)
            scc_b = loop_ctx.get(activity_b)
            if scc_a is not None and scc_b is not None and scc_a == scc_b:
                result[pair] = TemporalDependency.EVENTUAL
            else:
                result[pair] = TemporalDependency.TRUE_EVENTUAL
        elif b_before_a and ratio_b >= threshold:
            # Only B before A with sufficient ratio - NO DFG edge
            # We classify this as EVENTUAL_BACKWARD based on SCC context
            scc_a = loop_ctx.get(activity_a)
            scc_b = loop_ctx.get(activity_b)
            if scc_a is not None and scc_b is not None and scc_a == scc_b:
                result[pair] = TemporalDependency.EVENTUAL_BACKWARD
            else:
                result[pair] = TemporalDependency.TRUE_EVENTUAL_BACKWARD
        else:
            # No significant ordering pattern found
            result[pair] = TemporalDependency.NO_ORDERING

    return result
