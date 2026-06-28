"""Tests for Matrix class with activity management."""

import pytest
from armature.core.matrix import Matrix
from armature.core.dependencies import (
    DependencyCell,
    TemporalDependency,
    ExistentialDependency,
)


def test_matrix_creation_empty():
    """Test that Matrix() creates an empty matrix with format_version 2.0."""
    matrix = Matrix()
    assert matrix.format_version == "2.0"
    assert matrix.activities == []
    assert matrix.dependencies == {}
    assert matrix.description == ""


def test_add_activity():
    """Test that add_activity() adds to activities list."""
    matrix = Matrix()
    matrix.add_activity("A")
    assert "A" in matrix.activities
    assert matrix.activities == ["A"]

    matrix.add_activity("B")
    assert matrix.activities == ["A", "B"]


def test_add_activity_raises_on_duplicate():
    """Test that add_activity() raises ValueError on duplicate."""
    matrix = Matrix()
    matrix.add_activity("A")

    with pytest.raises(ValueError, match="Activity 'A' already exists"):
        matrix.add_activity("A")


def test_remove_activity():
    """Test that remove_activity() removes from activities list."""
    matrix = Matrix()
    matrix.add_activity("A")
    matrix.add_activity("B")
    matrix.add_activity("C")

    matrix.remove_activity("B")
    assert matrix.activities == ["A", "C"]


def test_remove_activity_raises_on_not_found():
    """Test that remove_activity() raises ValueError if not found."""
    matrix = Matrix()
    with pytest.raises(ValueError, match="Activity 'X' not found"):
        matrix.remove_activity("X")


def test_remove_activity_cleans_dependencies():
    """Test that remove_activity() removes all dependencies for that activity."""
    matrix = Matrix()
    matrix.add_activity("A")
    matrix.add_activity("B")
    matrix.add_activity("C")

    # Set some dependencies
    cell = DependencyCell(temporal=TemporalDependency.DIRECT)
    matrix.set_cell("A", "B", cell)
    matrix.set_cell("B", "C", cell)
    matrix.set_cell("C", "A", cell)

    # Verify dependencies exist
    assert "A" in matrix.dependencies
    assert "B" in matrix.dependencies["A"]
    assert "C" in matrix.dependencies["B"]

    # Remove B
    matrix.remove_activity("B")

    # Check B is removed as source
    assert "B" not in matrix.dependencies

    # Check B is removed as target from A
    assert "B" not in matrix.dependencies.get("A", {})


def test_activities_maintain_insertion_order():
    """Test that activities maintain deterministic insertion order."""
    matrix = Matrix()
    matrix.add_activity("Z")
    matrix.add_activity("A")
    matrix.add_activity("M")
    assert matrix.activities == ["Z", "A", "M"]


def test_get_cell_returns_neutral_for_missing():
    """Test that get_cell() returns neutral cell for missing dependencies."""
    matrix = Matrix()
    matrix.add_activity("A")
    matrix.add_activity("B")

    cell = matrix.get_cell("A", "B")
    assert cell.is_neutral()
    assert cell.temporal == TemporalDependency.NO_ORDERING
    assert cell.existential == ExistentialDependency.INDEPENDENCE


def test_set_cell_stores_correctly():
    """Test that set_cell() stores cell correctly."""
    matrix = Matrix()
    matrix.add_activity("A")
    matrix.add_activity("B")

    cell = DependencyCell(
        temporal=TemporalDependency.DIRECT, existential=ExistentialDependency.IMPLICATION
    )
    matrix.set_cell("A", "B", cell)

    retrieved = matrix.get_cell("A", "B")
    assert retrieved.temporal == TemporalDependency.DIRECT
    assert retrieved.existential == ExistentialDependency.IMPLICATION


def test_set_cell_raises_for_non_existent_source():
    """Test that set_cell() raises ValueError for non-existent source activity."""
    matrix = Matrix()
    matrix.add_activity("B")

    cell = DependencyCell(temporal=TemporalDependency.DIRECT)
    with pytest.raises(ValueError, match="Source activity 'A' not in activities"):
        matrix.set_cell("A", "B", cell)


def test_set_cell_raises_for_non_existent_target():
    """Test that set_cell() raises ValueError for non-existent target activity."""
    matrix = Matrix()
    matrix.add_activity("A")

    cell = DependencyCell(temporal=TemporalDependency.DIRECT)
    with pytest.raises(ValueError, match="Target activity 'B' not in activities"):
        matrix.set_cell("A", "B", cell)


def test_structural_integrity_validator_rejects_invalid_keys():
    """Test that structural integrity validator rejects invalid dependency keys."""
    # Try to create a matrix with dependencies referencing non-existent activities
    with pytest.raises(ValueError, match="Source activity 'X' not in activities list"):
        Matrix(
            activities=["A", "B"],
            dependencies={"X": {"A": DependencyCell()}},
        )

    with pytest.raises(ValueError, match="Target activity 'Y' not in activities list"):
        Matrix(
            activities=["A", "B"],
            dependencies={"A": {"Y": DependencyCell()}},
        )


def test_matrix_with_metadata():
    """Test that Matrix accepts metadata fields."""
    matrix = Matrix(
        description="Test matrix",
        source="test.xes",
        created_at="2026-01-22T00:00:00Z",
        classification="structured",
    )
    assert matrix.description == "Test matrix"
    assert matrix.source == "test.xes"
    assert matrix.created_at == "2026-01-22T00:00:00Z"
    assert matrix.classification == "structured"
