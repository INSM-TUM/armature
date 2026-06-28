"""Tests for full discovery pipeline."""

import os
import tempfile
from pathlib import Path

from armature.core.dependencies import (
    EXISTENTIAL_INVERSE,
    TEMPORAL_INVERSE,
    ExistentialDependency,
    TemporalDependency,
)
from armature.discovery.discover import discover


def test_discover_simple_sequence(temp_dir):
    """Test discover() with simple A->B->C sequence."""
    # Create minimal XES file
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="case1"/>
    <event>
      <string key="concept:name" value="A"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
    </event>
  </trace>
</log>"""
    xes_path = temp_dir / "simple.xes"
    xes_path.write_text(xes_content)

    # Discover matrix
    matrix = discover(xes_path)

    # Check matrix metadata
    assert matrix.format_version == "2.0"
    assert matrix.source == str(xes_path)

    # Check activities
    assert set(matrix.activities) == {"A", "B", "C"}

    # Check temporal dependencies
    assert matrix["A", "B"].temporal == TemporalDependency.DIRECT
    assert matrix["B", "C"].temporal == TemporalDependency.DIRECT
    assert matrix["A", "C"].temporal == TemporalDependency.TRUE_EVENTUAL

    # Check existential dependencies (all occur together in same trace)
    assert matrix["A", "B"].existential == ExistentialDependency.EQUIVALENCE
    assert matrix["B", "C"].existential == ExistentialDependency.EQUIVALENCE
    assert matrix["A", "C"].existential == ExistentialDependency.EQUIVALENCE


def test_discover_with_choices(temp_dir):
    """Test discover() with choice pattern (A then B or C)."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="case1"/>
    <event><string key="concept:name" value="A"/></event>
    <event><string key="concept:name" value="B"/></event>
  </trace>
  <trace>
    <string key="concept:name" value="case2"/>
    <event><string key="concept:name" value="A"/></event>
    <event><string key="concept:name" value="C"/></event>
  </trace>
</log>"""
    xes_path = temp_dir / "choices.xes"
    xes_path.write_text(xes_content)

    matrix = discover(xes_path)

    # Check activities
    assert set(matrix.activities) == {"A", "B", "C"}

    # Temporal: A before B and C in respective traces
    assert matrix["A", "B"].temporal == TemporalDependency.DIRECT
    assert matrix["A", "C"].temporal == TemporalDependency.DIRECT
    # B and C never co-occur
    assert matrix["B", "C"].temporal == TemporalDependency.NO_ORDERING

    # Existential dependencies (corrected after symmetry fix):
    # Trace 1: A,B  Trace 2: A,C
    # (A,B): count_only_a=1 (trace2), count_only_b=0, count_both=1 -> IMPLICATION_BACKWARD (B=>A)
    # (A,C): count_only_a=1 (trace1), count_only_c=0, count_both=1 -> IMPLICATION_BACKWARD (C=>A)
    # (B,C): count_only_b=1, count_only_c=1, count_both=0 -> NEGATED_EQUIVALENCE
    assert matrix["A", "B"].existential == ExistentialDependency.IMPLICATION_BACKWARD
    assert matrix["A", "C"].existential == ExistentialDependency.IMPLICATION_BACKWARD
    assert matrix["B", "C"].existential == ExistentialDependency.NEGATED_EQUIVALENCE

    # Verify inverse relationships for symmetry
    assert matrix["B", "A"].existential == ExistentialDependency.IMPLICATION  # A appears with B
    assert matrix["C", "A"].existential == ExistentialDependency.IMPLICATION  # A appears with C


def test_discover_with_loop(temp_dir):
    """Test discover() with loop pattern."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="case1"/>
    <event><string key="concept:name" value="A"/></event>
    <event><string key="concept:name" value="B"/></event>
    <event><string key="concept:name" value="A"/></event>
    <event><string key="concept:name" value="C"/></event>
  </trace>
</log>"""
    xes_path = temp_dir / "loop.xes"
    xes_path.write_text(xes_content)

    matrix = discover(xes_path)

    # Check activities
    assert set(matrix.activities) == {"A", "B", "C"}

    # Temporal: A and B form loop, then C
    assert matrix["A", "B"].temporal == TemporalDependency.DIRECT
    assert matrix["B", "A"].temporal == TemporalDependency.DIRECT
    # A->C is DIRECT (second A directly followed by C)
    # This is INDEPENDENCE since we have both A->C and (via loop) other orderings
    # Actually, in trace [A,B,A,C], we only have A->C ordering, so it's DIRECT
    assert matrix["A", "C"].temporal == TemporalDependency.DIRECT

    # Existential: all in same trace
    assert matrix["A", "B"].existential == ExistentialDependency.EQUIVALENCE
    assert matrix["A", "C"].existential == ExistentialDependency.EQUIVALENCE


def test_discover_custom_source():
    """Test discover() with custom source identifier."""
    # Use a valid XES file from fixtures if available, or create one

    # For this test, we'll use a StringIO-like approach or skip if no fixtures
    # Since we need a file path, let's create a simple test that checks source metadata
    # This test will be simpler - just verify source can be set

    # Actually, let's use temp_dir properly
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xes", delete=False) as f:
        f.write(
            """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="case1"/>
    <event><string key="concept:name" value="A"/></event>
  </trace>
</log>"""
        )
        temp_path = f.name

    try:
        # Test with custom source
        matrix = discover(temp_path, source="custom_source_id")
        assert matrix.source == "custom_source_id"

        # Test with default source (file path)
        matrix2 = discover(temp_path)
        assert matrix2.source == temp_path
    finally:
        Path(temp_path).unlink()


def test_discover_empty_log(temp_dir):
    """Test discover() with empty log."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
</log>"""
    xes_path = temp_dir / "empty.xes"
    xes_path.write_text(xes_content)

    matrix = discover(xes_path)

    # Empty log should produce empty matrix
    assert len(matrix.activities) == 0
    assert len(matrix.dependencies) == 0


def test_matrix_symmetry():
    """Matrix (a,b) and (b,a) must have inverse relationships."""
    # Use a simple log with known relationships
    # a->b->c linear
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
    <log>
      <trace>
        <event><string key="concept:name" value="a"/></event>
        <event><string key="concept:name" value="b"/></event>
        <event><string key="concept:name" value="c"/></event>
      </trace>
    </log>"""

    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False, mode="w") as f:
        f.write(xes_content)
        path = f.name

    try:
        matrix = discover(path)

        # Check all non-diagonal pairs for symmetry
        activities = list(matrix.activities)
        for a in activities:
            for b in activities:
                if a != b:
                    cell_ab = matrix[a, b]
                    cell_ba = matrix[b, a]

                    # Temporal inverse check
                    expected_temporal_ba = TEMPORAL_INVERSE.get(cell_ab.temporal, cell_ab.temporal)
                    assert (
                        cell_ba.temporal == expected_temporal_ba
                    ), f"Temporal symmetry failed for ({a},{b}): {cell_ab.temporal} -> {cell_ba.temporal}, expected {expected_temporal_ba}"

                    # Existential inverse check
                    expected_exist_ba = EXISTENTIAL_INVERSE.get(
                        cell_ab.existential, cell_ab.existential
                    )
                    assert (
                        cell_ba.existential == expected_exist_ba
                    ), f"Existential symmetry failed for ({a},{b}): {cell_ab.existential} -> {cell_ba.existential}, expected {expected_exist_ba}"
    finally:
        os.unlink(path)


def test_symmetry_with_implication():
    """IMPLICATION on (a,b) means IMPLICATION_BACKWARD on (b,a)."""
    # Create log where A=>B (A always has B, but B can be alone)
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
    <log>
      <trace>
        <event><string key="concept:name" value="a"/></event>
        <event><string key="concept:name" value="b"/></event>
      </trace>
      <trace>
        <event><string key="concept:name" value="b"/></event>
      </trace>
    </log>"""

    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False, mode="w") as f:
        f.write(xes_content)
        path = f.name

    try:
        matrix = discover(path)

        # Verify (a,b).existential == IMPLICATION (A always has B)
        assert (
            matrix["a", "b"].existential == ExistentialDependency.IMPLICATION
        ), f"Expected IMPLICATION for (a,b), got {matrix['a', 'b'].existential}"

        # Verify (b,a).existential == IMPLICATION_BACKWARD (B can occur without A)
        assert (
            matrix["b", "a"].existential == ExistentialDependency.IMPLICATION_BACKWARD
        ), f"Expected IMPLICATION_BACKWARD for (b,a), got {matrix['b', 'a'].existential}"
    finally:
        os.unlink(path)
