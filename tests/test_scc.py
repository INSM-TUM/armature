"""Tests for Strongly Connected Component (SCC) detection."""

from armature.discovery.scc import find_sccs

from armature.discovery.dfg import DFG


def test_linear_no_sccs():
    """Linear A->B->C has no SCCs (all single-activity components)."""
    dfg = DFG(
        activities={"A", "B", "C"},
        edges={"A": {"B": 1}, "B": {"C": 1}},
        start_activities={"A"},
        end_activities={"C"},
    )
    loop_ctx = find_sccs(dfg)

    # Linear flow - no loops, all activities have None (not in any SCC)
    assert loop_ctx.get("A") is None
    assert loop_ctx.get("B") is None
    assert loop_ctx.get("C") is None


def test_simple_loop():
    """Simple loop A->B->A creates one SCC {A, B}."""
    dfg = DFG(
        activities={"A", "B"},
        edges={"A": {"B": 1}, "B": {"A": 1}},
        start_activities={"A"},
        end_activities={"B"},
    )
    loop_ctx = find_sccs(dfg)

    # Both A and B should be in same SCC
    assert loop_ctx.get("A") is not None
    assert loop_ctx.get("B") is not None
    assert loop_ctx["A"] == loop_ctx["B"]  # Same SCC index


def test_self_loop():
    """Self-loop A->A creates SCC {A}."""
    dfg = DFG(
        activities={"A"},
        edges={"A": {"A": 1}},
        start_activities={"A"},
        end_activities={"A"},
    )
    loop_ctx = find_sccs(dfg)

    # Self-loop is an SCC
    assert loop_ctx.get("A") is not None


def test_complex_partial_loop():
    """A->B->C->B with D->E: SCC {B, C}, no SCC for A, D, E."""
    dfg = DFG(
        activities={"A", "B", "C", "D", "E"},
        edges={
            "A": {"B": 1},
            "B": {"C": 1},
            "C": {"B": 1},
            "D": {"E": 1},
        },
        start_activities={"A", "D"},
        end_activities={"C", "E"},
    )
    loop_ctx = find_sccs(dfg)

    # B and C form an SCC
    assert loop_ctx.get("B") is not None
    assert loop_ctx.get("C") is not None
    assert loop_ctx["B"] == loop_ctx["C"]  # Same SCC

    # A, D, E are not in any SCC
    assert loop_ctx.get("A") is None
    assert loop_ctx.get("D") is None
    assert loop_ctx.get("E") is None


def test_multiple_sccs():
    """Multiple separate loops: A->B->A and C->D->C."""
    dfg = DFG(
        activities={"A", "B", "C", "D"},
        edges={
            "A": {"B": 1},
            "B": {"A": 1},
            "C": {"D": 1},
            "D": {"C": 1},
        },
        start_activities={"A", "C"},
        end_activities={"B", "D"},
    )
    loop_ctx = find_sccs(dfg)

    # A and B in one SCC
    assert loop_ctx.get("A") is not None
    assert loop_ctx.get("B") is not None
    assert loop_ctx["A"] == loop_ctx["B"]

    # C and D in different SCC
    assert loop_ctx.get("C") is not None
    assert loop_ctx.get("D") is not None
    assert loop_ctx["C"] == loop_ctx["D"]

    # A-B SCC different from C-D SCC
    assert loop_ctx["A"] != loop_ctx["C"]


def test_empty_dfg():
    """Empty DFG returns empty LoopContext."""
    dfg = DFG(
        activities=set(),
        edges={},
        start_activities=set(),
        end_activities=set(),
    )
    loop_ctx = find_sccs(dfg)

    assert len(loop_ctx) == 0
