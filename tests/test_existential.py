"""Tests for existential dependency extraction."""

from datetime import datetime

from armature.core.dependencies import ExistentialDependency
from armature.discovery.existential import extract_existential_dependencies
from armature.discovery.models import Event, Trace


def test_equivalence_always_together():
    """Traces [[A,B], [A,B]]: EQUIVALENCE for (A,B) - always together."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[Event(activity="A"), Event(activity="B")]),
    ]

    deps = extract_existential_dependencies(traces)

    # A and B always occur together
    assert deps[("A", "B")] == ExistentialDependency.EQUIVALENCE
    assert deps[("B", "A")] == ExistentialDependency.EQUIVALENCE


def test_implication_a_implies_b():
    """Traces [[A,B], [B]]: IMPLICATION A=>B - whenever A occurs, B occurs."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[Event(activity="B")]),
    ]

    deps = extract_existential_dependencies(traces)

    # Whenever A occurs (trace 1), B also occurs. B can occur without A (trace 2).
    assert deps[("A", "B")] == ExistentialDependency.IMPLICATION


def test_implication_b_implies_a():
    """Traces [[A,B], [A]]: IMPLICATION B=>A - whenever B occurs, A occurs."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[Event(activity="A")]),
    ]

    deps = extract_existential_dependencies(traces)

    # Whenever B occurs (trace 1), A also occurs. A can occur without B (trace 2).
    assert deps[("B", "A")] == ExistentialDependency.IMPLICATION


def test_xor_exactly_one():
    """Traces [[A], [B]]: NEGATED_EQUIVALENCE - exactly one per trace, never both, never neither."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A")]),
        Trace(case_id="2", events=[Event(activity="B")]),
    ]

    deps = extract_existential_dependencies(traces)

    # Exactly one of A or B per trace, never together, never neither
    assert deps[("A", "B")] == ExistentialDependency.NEGATED_EQUIVALENCE
    assert deps[("B", "A")] == ExistentialDependency.NEGATED_EQUIVALENCE


def test_nand_never_together():
    """Traces [[A,C], [B,C], [C]]: NAND for (A,B) - never together but not NEGATED_EQUIVALENCE."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="C")]),
        Trace(case_id="2", events=[Event(activity="B"), Event(activity="C")]),
        Trace(case_id="3", events=[Event(activity="C")]),
    ]

    deps = extract_existential_dependencies(traces)

    # A and B never together, but there's a trace with neither
    assert deps[("A", "B")] == ExistentialDependency.NAND
    assert deps[("B", "A")] == ExistentialDependency.NAND


def test_or_sometimes_together():
    """Traces [[A], [B], [A,B]]: OR - sometimes together, sometimes not."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A")]),
        Trace(case_id="2", events=[Event(activity="B")]),
        Trace(case_id="3", events=[Event(activity="A"), Event(activity="B")]),
    ]

    deps = extract_existential_dependencies(traces)

    # Sometimes together, sometimes only one
    assert deps[("A", "B")] == ExistentialDependency.OR
    assert deps[("B", "A")] == ExistentialDependency.OR


def test_independence_all_combinations():
    """Traces [[A], [B], [A,B], [C]]: INDEPENDENCE for (A,B) - all combinations present."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A")]),
        Trace(case_id="2", events=[Event(activity="B")]),
        Trace(case_id="3", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="4", events=[Event(activity="C")]),
    ]

    deps = extract_existential_dependencies(traces)

    # All combinations: only A, only B, both, neither (in trace 4)
    assert deps[("A", "B")] == ExistentialDependency.INDEPENDENCE
    assert deps[("B", "A")] == ExistentialDependency.INDEPENDENCE


def test_multiple_activity_pairs():
    """Test multiple pairs in same trace set."""
    traces = [
        Trace(
            case_id="1",
            events=[Event(activity="A"), Event(activity="B"), Event(activity="C")],
        ),
        Trace(case_id="2", events=[Event(activity="A"), Event(activity="B")]),
    ]

    deps = extract_existential_dependencies(traces)

    # A and B always together = EQUIVALENCE
    assert deps[("A", "B")] == ExistentialDependency.EQUIVALENCE
    assert deps[("B", "A")] == ExistentialDependency.EQUIVALENCE

    # A and C: whenever C occurs, A occurs (IMPLICATION C=>A)
    assert deps[("C", "A")] == ExistentialDependency.IMPLICATION
    # B and C: whenever C occurs, B occurs (IMPLICATION C=>B)
    assert deps[("C", "B")] == ExistentialDependency.IMPLICATION


def test_empty_traces():
    """Empty traces should be handled gracefully."""
    traces = [
        Trace(case_id="1", events=[Event(activity="A"), Event(activity="B")]),
        Trace(case_id="2", events=[]),
    ]

    deps = extract_existential_dependencies(traces)

    # Should handle empty trace (counted as neither A nor B)
    # This creates a trace with neither A nor B, plus trace with both
    # count_both=1, count_only_a=0, count_only_b=0, count_neither=1
    # This is INDEPENDENCE (all combinations present but missing only_a and only_b)
    # Actually, let's reconsider: if we have both and neither, that's still not all combinations
    # For INDEPENDENCE we need: only_a, only_b, both, neither
    # Here we have: both, neither -> should be EQUIVALENCE (when they occur, they're together)
    assert deps[("A", "B")] == ExistentialDependency.EQUIVALENCE


def test_implication_forward():
    """A=>B: whenever A occurs, B also occurs (but B can occur without A)."""
    # Traces where:
    # - A always has B (count_only_a = 0)
    # - B can be alone (count_only_b > 0)
    traces = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b")]),  # both
        Trace(case_id="2", events=[Event(activity="a"), Event(activity="b")]),  # both
        Trace(case_id="3", events=[Event(activity="b")]),  # only B
    ]
    result = extract_existential_dependencies(traces)

    # (a, b) should be IMPLICATION (A=>B)
    # count_only_a=0 (A never alone), count_only_b=1 (B can be alone)
    assert result[("a", "b")] == ExistentialDependency.IMPLICATION


def test_implication_backward():
    """B=>A: whenever B occurs, A also occurs (but A can occur without B)."""
    # Traces where:
    # - B always has A (count_only_b = 0)
    # - A can be alone (count_only_a > 0)
    traces = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b")]),  # both
        Trace(case_id="2", events=[Event(activity="a"), Event(activity="b")]),  # both
        Trace(case_id="3", events=[Event(activity="a")]),  # only A
    ]
    result = extract_existential_dependencies(traces)

    # For pair (a, b):
    # count_only_a = 1 (trace 3: A alone)
    # count_only_b = 0 (B never alone)
    # So (a, b) should be IMPLICATION_BACKWARD (B=>A)
    assert result[("a", "b")] == ExistentialDependency.IMPLICATION_BACKWARD


def test_implication_vs_equivalence():
    """Verify IMPLICATION != EQUIVALENCE."""
    # EQUIVALENCE requires count_only_a == 0 AND count_only_b == 0
    # IMPLICATION requires only one of them to be 0
    traces_equiv = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b")]),
        Trace(case_id="2", events=[Event(activity="a"), Event(activity="b")]),
    ]
    result_equiv = extract_existential_dependencies(traces_equiv)
    assert result_equiv[("a", "b")] == ExistentialDependency.EQUIVALENCE

    traces_impl = [
        Trace(case_id="1", events=[Event(activity="a"), Event(activity="b")]),
        Trace(case_id="2", events=[Event(activity="b")]),  # only B
    ]
    result_impl = extract_existential_dependencies(traces_impl)
    assert result_impl[("a", "b")] == ExistentialDependency.IMPLICATION  # A=>B


def test_self_loop_implication():
    """Self-loops should have IMPLICATION existential dependency.

    When an activity repeats in a trace, the second occurrence implies
    the first occurrence (a => a).
    """
    traces = [
        Trace(
            case_id="1",
            events=[
                Event(activity="a", timestamp=datetime(2024, 1, 1)),
                Event(activity="b", timestamp=datetime(2024, 1, 2)),
                Event(activity="b", timestamp=datetime(2024, 1, 3)),  # b repeats
                Event(activity="c", timestamp=datetime(2024, 1, 4)),
                Event(activity="c", timestamp=datetime(2024, 1, 5)),  # c repeats
            ],
        ),
    ]

    result = extract_existential_dependencies(traces)

    # Self-loops with repetition should have IMPLICATION
    assert result[("b", "b")] == ExistentialDependency.IMPLICATION
    assert result[("c", "c")] == ExistentialDependency.IMPLICATION

    # Activity without repetition should not have self-loop entry
    assert ("a", "a") not in result


def test_self_loop_no_repetition():
    """Activities that don't repeat should not have self-loop dependencies."""
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

    result = extract_existential_dependencies(traces)

    # No repetitions = no self-loop dependencies
    assert ("a", "a") not in result
    assert ("b", "b") not in result
    assert ("c", "c") not in result
