"""Validation tests for event log 02 - parallel activities."""

from pathlib import Path

from armature.core.dependencies import TemporalDependency
from armature.discovery import discover


def test_log02_parallel_activities():
    """Validate log 02: parallel activities b/c make a->b and a->c EVENTUAL.

    Log 02 actual traces:
    - 20x: a → b → c → d → e
    - 22x: a → b → c → d → f
    - 30x: a → c → b → d → e
    - 25x: a → c → b → d → f

    Key insight: a has parallel successors b and c (can occur in either order).
    This makes a->b and a->c non-deterministic, so they should be EVENTUAL.

    Expected behavior:
    - (b,c): INDEPENDENCE (both orderings observed)
    - (a,b): TRUE_EVENTUAL (a precedes b, but b and c are parallel)
    - (a,c): TRUE_EVENTUAL (a precedes c, but b and c are parallel)
    - (b,d): DIRECT (b->d is always consecutive when it occurs, b's successors c and d don't show independence)
    - (c,d): DIRECT (c->d is always consecutive when it occurs, c's successors b and d don't show independence)
    """
    log_path = Path("Test Data/Discovery/event_log_02_Parallelism_NegatedEquivalence.xes")
    matrix = discover(log_path)

    # Core fix: b and c are parallel
    assert matrix["b", "c"].temporal == TemporalDependency.INDEPENDENCE

    # Core fix: a precedes both with non-deterministic ordering (parallel successors)
    assert matrix["a", "b"].temporal == TemporalDependency.TRUE_EVENTUAL
    assert matrix["a", "c"].temporal == TemporalDependency.TRUE_EVENTUAL

    # b->d and c->d are DIRECT (deterministic succession, no parallel alternatives)
    # Note: Even though both b and c eventually lead to d, each individual succession
    # is deterministic and immediate, so they remain DIRECT per ARM theory
    assert matrix["b", "d"].temporal == TemporalDependency.DIRECT
    assert matrix["c", "d"].temporal == TemporalDependency.DIRECT
