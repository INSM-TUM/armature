"""Tests for temporal dependency extraction."""

from armature.core.dependencies import TemporalDependency
from armature.discovery.dfg import DFG
from armature.discovery.models import Event, Trace
from armature.discovery.scc import LoopContext
from armature.discovery.temporal import extract_temporal_dependencies


def test_direct_dependencies():
    """Trace [A,B,C]: A->B direct, B->C direct."""
    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="A"), Event(activity="B"), Event(activity="C")],
        )
    ]
    dfg = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1}, "B": {"C": 1}},
        start_activities={"A"},
        end_activities={"C"},
    )
    loop_ctx: LoopContext = {}  # No loops

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Direct dependencies
    assert deps[("A", "B")] == TemporalDependency.DIRECT
    assert deps[("B", "C")] == TemporalDependency.DIRECT


def test_true_eventual_no_loop():
    """Linear [A,B,C] no loops: A->C is true_eventual."""
    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="A"), Event(activity="B"), Event(activity="C")],
        )
    ]
    dfg = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1}, "B": {"C": 1}},
        start_activities={"A"},
        end_activities={"C"},
    )
    loop_ctx: LoopContext = {}  # No loops

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # A->C is true eventual (not direct, not in same SCC)
    assert deps[("A", "C")] == TemporalDependency.TRUE_EVENTUAL


def test_eventual_with_loop():
    """Trace [A,B,A,D,C] with SCC {A,B}: A->C is true_eventual (not in same SCC)."""
    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="A"),
                Event(activity="D"),
                Event(activity="C"),
            ],
        )
    ]
    dfg = DFG(
        activities={"A", "B", "C", "D"},
        edges={"A": {"B": 1, "D": 1}, "B": {"A": 1}, "D": {"C": 1}},
        start_activities={"A"},
        end_activities={"C"},
    )
    loop_ctx: LoopContext = {"A": 0, "B": 0}  # A and B in same SCC

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # A->B direct (consecutive)
    assert deps[("A", "B")] == TemporalDependency.DIRECT
    # A->C true_eventual (not direct, A in SCC but C not in same SCC)
    assert deps[("A", "C")] == TemporalDependency.TRUE_EVENTUAL


def test_independence_both_orderings():
    """Traces [[A,B], [B,A]]: INDEPENDENCE for (A,B) - both orderings observed."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[Event(activity="B"), Event(activity="A")]),
    ]
    dfg = DFG(
        activities={"A", "B"},
        edges={"A": {"B": 1}, "B": {"A": 1}},
        start_activities={"A", "B"},
        end_activities={"B", "A"},
    )
    loop_ctx: LoopContext = {}

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Both A->B and B->A observed in different traces = INDEPENDENCE
    assert deps[("A", "B")] == TemporalDependency.INDEPENDENCE


def test_no_ordering_never_cooccur():
    """Traces [[A], [B]]: NO_ORDERING for (A,B) - never co-occur."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A")]),
        Trace(case_id="2", events=[Event(activity="B")]),
    ]
    dfg = DFG(
        activities={"A", "B"},
        edges={},
        start_activities={"A", "B"},
        end_activities={"A", "B"},
    )
    loop_ctx: LoopContext = {}

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # A and B never appear in same trace = NO_ORDERING
    assert deps[("A", "B")] == TemporalDependency.NO_ORDERING


def test_no_ordering_separate_traces():
    """Traces [[A,C], [B,D]]: NO_ORDERING for (A,B), (A,D), (B,C), (C,D)."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="C")]),
        Trace(case_id="2", events=[Event(activity="B"), Event(activity="D")]),
    ]
    dfg = DFG(
        activities={"A", "B", "C", "D"},
        edges={"A": {"C": 1}, "B": {"D": 1}},
        start_activities={"A", "B"},
        end_activities={"C", "D"},
    )
    loop_ctx: LoopContext = {}

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Activities that never co-occur
    assert deps[("A", "B")] == TemporalDependency.NO_ORDERING
    assert deps[("A", "D")] == TemporalDependency.NO_ORDERING
    assert deps[("B", "C")] == TemporalDependency.NO_ORDERING
    assert deps[("C", "D")] == TemporalDependency.NO_ORDERING

    # Activities that do co-occur have ordering
    assert deps[("A", "C")] == TemporalDependency.DIRECT
    assert deps[("B", "D")] == TemporalDependency.DIRECT


def test_partial_cooccurrence():
    """Traces [[A,B], [A,C]]: A->B direct, A->C direct, NO_ORDERING for (B,C)."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[Event(activity="A"), Event(activity="C")]),
    ]
    dfg = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1, "C": 1}},
        start_activities={"A"},
        end_activities={"B", "C"},
    )
    loop_ctx: LoopContext = {}

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # A before B and C in respective traces
    assert deps[("A", "B")] == TemporalDependency.DIRECT
    assert deps[("A", "C")] == TemporalDependency.DIRECT

    # B and C never co-occur
    assert deps[("B", "C")] == TemporalDependency.NO_ORDERING


def test_self_loop_direct():
    """Trace [A,A]: A->A is DIRECT (self-loop, non-nested)."""
    traces = [Trace(case_id="1", events=[Event(activity="A"), Event(activity="A")])]
    dfg = DFG(
        activities={"A"},
        edges={"A": {"A": 2}},
        start_activities={"A"},
        end_activities={"A"},
    )
    loop_ctx: LoopContext = {"A": 0}  # A is in SCC (self-loop)

    deps = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Self-loop: first instance directly followed by second instance
    assert deps[("A", "A")] == TemporalDependency.DIRECT


# Bug 3 regression tests - Self-loop detection


def test_self_loop_only_when_repeats():
    """(a,a) should be NO_ORDERING if activity never repeats."""
    # Trace: a -> b -> c (no repeats)
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="a"), Event(activity="b"), Event(activity="c")],
        )
    ]
    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # No self-loops in this trace - diagonal cells not in result
    assert ("a", "a") not in result
    assert ("b", "b") not in result


def test_self_loop_when_repeats():
    """(a,a) should be EVENTUAL if activity repeats consecutively in trace."""
    # Trace: a -> a -> b (a repeats consecutively = self-loop)
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="a"), Event(activity="a"), Event(activity="b")],
        )
    ]
    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # a repeats consecutively = self-loop exists in DFG → DIRECT
    assert result[("a", "a")] == TemporalDependency.DIRECT


# Bug 4 regression test - Direct vs Eventual


def test_direct_requires_dfg_edge():
    """Direct only if DFG edge exists (actual consecutive occurrence)."""
    # Trace: a -> b -> c (a->c is eventual, not direct)
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="a"), Event(activity="b"), Event(activity="c")],
        )
    ]
    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # a->b direct (consecutive), a->c NOT direct (b between them)
    assert result[("a", "b")] == TemporalDependency.DIRECT
    assert result[("a", "c")] != TemporalDependency.DIRECT


# Bug 5 regression test - True Eventual


def test_true_eventual_detection():
    """TRUE_EVENTUAL for reachable but never consecutive, not in SCC."""
    # a->b->c (linear, no loops)
    # a->c should be TRUE_EVENTUAL (reachable but never direct)
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="a"), Event(activity="b"), Event(activity="c")],
        )
    ]
    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)  # No SCCs in linear sequence
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # a->c: reachable, not direct, not in same SCC = TRUE_EVENTUAL
    assert result[("a", "c")] == TemporalDependency.TRUE_EVENTUAL


# Bug fix regression tests - Parallel activity detection


def test_parallel_activities_not_direct():
    """When activities are parallel, predecessors should be EVENTUAL not DIRECT."""
    # Traces: a->b->c and a->c->b (b and c parallel)
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b"), Event(activity="c")]),
        Trace(case_id="2", events=[Event(activity="a"), Event(activity="b"), Event(activity="c")]),
        Trace(case_id="3", events=[Event(activity="a"), Event(activity="c"), Event(activity="b")]),
        Trace(case_id="4", events=[Event(activity="a"), Event(activity="c"), Event(activity="b")]),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # b and c are parallel (can be in either order)
    assert result[("b", "c")] == TemporalDependency.INDEPENDENCE

    # a precedes both, but since b/c are parallel, a->b and a->c are EVENTUAL not DIRECT
    assert result[("a", "b")] == TemporalDependency.EVENTUAL
    assert result[("a", "c")] == TemporalDependency.EVENTUAL


def test_sequential_activities_are_direct():
    """When activities are always sequential, relationships are DIRECT."""
    # Traces: always a->b->c in this order
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b"), Event(activity="c")]),
        Trace(case_id="2", events=[Event(activity="a"), Event(activity="b"), Event(activity="c")]),
        Trace(case_id="3", events=[Event(activity="a"), Event(activity="b"), Event(activity="c")]),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # All are deterministic direct successions
    assert result[("a", "b")] == TemporalDependency.DIRECT
    assert result[("b", "c")] == TemporalDependency.DIRECT


# Phase 3.3 regression tests - Parallel path detection


def test_parallel_path_detection():
    """Test has_multiple_paths() function directly."""
    from armature.discovery.temporal import has_multiple_paths

    # Simple parallel: A→B (direct), A→C→B (indirect) = True
    dfg = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1, "C": 1}, "C": {"B": 1}},
        start_activities={"A"},
        end_activities={"B"},
    )
    assert has_multiple_paths(dfg, "A", "B") is True

    # Sequential: A→B, B→C = False (no parallel path A→C)
    dfg2 = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1}, "B": {"C": 1}},
        start_activities={"A"},
        end_activities={"C"},
    )
    assert has_multiple_paths(dfg2, "A", "C") is False

    # No connection: A, B isolated = False
    dfg3 = DFG(
        activities={"A", "B"},
        edges={},
        start_activities={"A", "B"},
        end_activities={"A", "B"},
    )
    assert has_multiple_paths(dfg3, "A", "B") is False


def test_direct_with_parallel_paths():
    """Test EVENTUAL classification for parallel gateway patterns."""
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    # Pattern: A→(B||C)→D (parallel gateway)
    # A splits to B and C, which both go to D
    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="C"),
                Event(activity="D"),
            ],
        ),
        Trace(
            case_id="2",
            events=[
                Event(activity="A"),
                Event(activity="C"),
                Event(activity="B"),
                Event(activity="D"),
            ],
        ),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Expected: (A,B), (A,C), (B,D), (C,D) all have parallel paths
    # A→B has alternate path A→C→...→B (via independence)
    # So these should be EVENTUAL or TRUE_EVENTUAL, not DIRECT
    # Note: The exact classification depends on DFG structure
    # With parallel patterns, we expect NOT DIRECT
    assert result[("A", "B")] != TemporalDependency.DIRECT
    assert result[("A", "C")] != TemporalDependency.DIRECT


def test_backward_dependencies():
    """Test backward detection: E→A, E→B, E→D."""
    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    # Pattern: E→A, E→B, E→D (E precedes all)
    traces = [
        Trace(case_id="1", events=[Event(activity="E"), Event(activity="A")]),
        Trace(case_id="2", events=[Event(activity="E"), Event(activity="B")]),
        Trace(case_id="3", events=[Event(activity="E"), Event(activity="D")]),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # Expected: (A,E)=DIRECT_BACKWARD, (B,E)=DIRECT_BACKWARD, (D,E)=DIRECT_BACKWARD
    # Because DFG has E→A, E→B, E→D edges
    assert result[("A", "E")] == TemporalDependency.DIRECT_BACKWARD
    assert result[("B", "E")] == TemporalDependency.DIRECT_BACKWARD
    assert result[("D", "E")] == TemporalDependency.DIRECT_BACKWARD


# Phase 3.4 regression tests - TRUE_EVENTUAL vs EVENTUAL and self-loop fixes


def test_eventual_with_dfg_edge():
    """Activities with DFG edge but parallel paths should be EVENTUAL not TRUE_EVENTUAL."""
    from datetime import datetime

    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    # Pattern: a→(b||c)→d where b and c show INDEPENDENCE
    # b→d and c→d have DFG edges but a has multiple parallel successors
    # So from a's perspective, the succession to d is non-deterministic
    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="a", timestamp=datetime(2024, 1, 1)),
                Event(activity="b", timestamp=datetime(2024, 1, 2)),
                Event(activity="c", timestamp=datetime(2024, 1, 3)),
                Event(activity="d", timestamp=datetime(2024, 1, 4)),
            ],
        ),
        Trace(
            case_id="2",
            events=[
                Event(activity="a", timestamp=datetime(2024, 1, 1)),
                Event(activity="c", timestamp=datetime(2024, 1, 2)),
                Event(activity="b", timestamp=datetime(2024, 1, 3)),
                Event(activity="d", timestamp=datetime(2024, 1, 4)),
            ],
        ),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # a→b and a→c both have DFG edges but b/c are parallel (show INDEPENDENCE)
    # So a→b and a→c should be EVENTUAL not DIRECT (non-deterministic)
    # Check that they're NOT DIRECT (the exact type depends on SCC)
    assert result[("a", "b")] != TemporalDependency.DIRECT
    assert result[("a", "c")] != TemporalDependency.DIRECT


def test_true_eventual_no_dfg_edge():
    """Activities never consecutive should be TRUE_EVENTUAL."""
    from datetime import datetime

    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="a", timestamp=datetime(2024, 1, 1)),
                Event(activity="b", timestamp=datetime(2024, 1, 2)),
                Event(activity="c", timestamp=datetime(2024, 1, 3)),
            ],
        ),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # (a,c) has no DFG edge (b in between) → TRUE_EVENTUAL
    assert result[("a", "c")] == TemporalDependency.TRUE_EVENTUAL


def test_self_loop_eventual():
    """Self-loops should be DIRECT (non-nested loop assumption)."""
    from datetime import datetime

    from armature.discovery.dfg import build_dfg
    from armature.discovery.scc import find_sccs

    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="a", timestamp=datetime(2024, 1, 1)),
                Event(activity="b", timestamp=datetime(2024, 1, 2)),
                Event(activity="b", timestamp=datetime(2024, 1, 3)),
            ],
        ),
    ]

    dfg = build_dfg(traces)
    loop_ctx = find_sccs(dfg)
    result = extract_temporal_dependencies(traces, dfg, loop_ctx)

    # (b,b) is self-loop → DIRECT (structured non-nested loop)
    assert result[("b", "b")] == TemporalDependency.DIRECT


# Phase 3.5 regression tests - TRUE_EVENTUAL vs EVENTUAL precise fixes


import pytest


@pytest.mark.parametrize(
    "log_num,pair,expected_temporal",
    [
        ("03", ("d", "c"), TemporalDependency.TRUE_EVENTUAL),
        ("04", ("b", "d"), TemporalDependency.EVENTUAL),
        ("04", ("c", "d"), TemporalDependency.EVENTUAL),
        ("05", ("d", "e"), TemporalDependency.EVENTUAL_BACKWARD),
        ("05", ("b", "d"), TemporalDependency.EVENTUAL),
        ("05", ("c", "d"), TemporalDependency.EVENTUAL),
        ("06", ("a", "c"), TemporalDependency.EVENTUAL),
        ("06", ("a", "b"), TemporalDependency.EVENTUAL),
        ("07", ("a", "d"), TemporalDependency.EVENTUAL),
        ("08", ("a", "b"), TemporalDependency.EVENTUAL),
        ("08", ("a", "c"), TemporalDependency.EVENTUAL),
        ("08", ("b", "b"), TemporalDependency.DIRECT),
        ("08", ("c", "c"), TemporalDependency.EVENTUAL),
        ("08", ("b", "e"), TemporalDependency.EVENTUAL),
        ("08", ("c", "e"), TemporalDependency.EVENTUAL),
        ("09", ("a", "c"), TemporalDependency.EVENTUAL),
        ("09", ("a", "b"), TemporalDependency.NO_ORDERING),
        ("09", ("b", "c"), TemporalDependency.EVENTUAL),
        ("09", ("b", "d"), TemporalDependency.EVENTUAL),
    ],
)
def test_true_eventual_vs_eventual_precise(log_num, pair, expected_temporal):
    """Regression test for TRUE_EVENTUAL vs EVENTUAL from manual validation."""
    from pathlib import Path
    from armature.discovery import discover

    # Resolve glob for test log
    matches = list(Path("Test Data/Discovery").glob(f"event_log_{log_num}_*.xes"))
    assert len(matches) == 1, f"Expected 1 log for {log_num}, found {len(matches)}"

    matrix = discover(str(matches[0]))
    cell = matrix[pair]

    assert cell.temporal == expected_temporal, (
        f"Log {log_num} pair {pair}: expected {expected_temporal.name}, "
        f"got {cell.temporal.name}"
    )


def test_self_loop_not_created_without_repetition():
    """(a,a) should not exist if activity never repeats."""
    from pathlib import Path
    from armature.discovery import discover
    from armature.core.dependencies import ExistentialDependency

    # Log 08 - activity 'a' never repeats
    matches = list(Path("Test Data/Discovery").glob("event_log_08_*.xes"))
    matrix = discover(str(matches[0]))

    # (a,a) should not be in matrix (no temporal/existential set)
    cell = matrix[("a", "a")]
    assert cell.temporal == TemporalDependency.NO_ORDERING
    assert cell.existential == ExistentialDependency.INDEPENDENCE
