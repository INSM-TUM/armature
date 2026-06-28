"""Tests for Directly-Follows Graph (DFG) construction."""

from armature.discovery.dfg import build_dfg

from armature.discovery.models import Event, Trace


def test_linear_sequence():
    """Test DFG construction from linear sequence A->B->C."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="C"),
            ],
        )
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A", "B", "C"}
    assert dfg.edges == {"A": {"B": 1}, "B": {"C": 1}}
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"C"}


def test_loop():
    """Test DFG captures loop A->B->A."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="A"),
            ],
        )
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A", "B"}
    assert dfg.edges == {"A": {"B": 1}, "B": {"A": 1}}
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"A"}


def test_self_loop():
    """Test DFG captures self-loop A->A."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="A"),
            ],
        )
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A"}
    assert dfg.edges == {"A": {"A": 1}}
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"A"}


def test_parallel_from_multiple_traces():
    """Test parallel branches A->(B,C)->D from multiple traces."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="D"),
            ],
        ),
        Trace(
            case_id="case_2",
            events=[
                Event(activity="A"),
                Event(activity="C"),
                Event(activity="D"),
            ],
        ),
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A", "B", "C", "D"}
    assert dfg.edges == {
        "A": {"B": 1, "C": 1},
        "B": {"D": 1},
        "C": {"D": 1},
    }
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"D"}


def test_multiple_traces_increase_counts():
    """Test transition counts increase with multiple identical traces."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
            ],
        ),
        Trace(
            case_id="case_2",
            events=[
                Event(activity="A"),
                Event(activity="B"),
            ],
        ),
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A", "B"}
    assert dfg.edges == {"A": {"B": 2}}
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"B"}


def test_empty_trace_list():
    """Test DFG from empty trace list returns empty structures."""
    traces = []

    dfg = build_dfg(traces)

    assert dfg.activities == set()
    assert dfg.edges == {}
    assert dfg.start_activities == set()
    assert dfg.end_activities == set()


def test_single_activity_trace():
    """Test DFG from trace with single activity."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
            ],
        )
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A"}
    assert dfg.edges == {}
    assert dfg.start_activities == {"A"}
    assert dfg.end_activities == {"A"}


def test_empty_trace():
    """Test DFG handles trace with no events."""
    traces = [
        Trace(case_id="case_1", events=[]),
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == set()
    assert dfg.edges == {}
    assert dfg.start_activities == set()
    assert dfg.end_activities == set()


def test_mixed_traces():
    """Test DFG with mixed patterns and multiple start/end activities."""
    traces = [
        Trace(
            case_id="case_1",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="C"),
            ],
        ),
        Trace(
            case_id="case_2",
            events=[
                Event(activity="X"),
                Event(activity="Y"),
            ],
        ),
        Trace(
            case_id="case_3",
            events=[
                Event(activity="A"),
                Event(activity="B"),
                Event(activity="D"),
            ],
        ),
    ]

    dfg = build_dfg(traces)

    assert dfg.activities == {"A", "B", "C", "D", "X", "Y"}
    assert dfg.edges == {
        "A": {"B": 2},
        "B": {"C": 1, "D": 1},
        "X": {"Y": 1},
    }
    assert dfg.start_activities == {"A", "X"}
    assert dfg.end_activities == {"C", "D", "Y"}
