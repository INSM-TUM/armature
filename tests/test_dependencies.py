"""Tests for dependency types and DependencyCell model."""

import pytest

from armature.core.dependencies import (
    EXISTENTIAL_INVERSE,
    TEMPORAL_INVERSE,
    DependencyCell,
    ExistentialDependency,
    TemporalDependency,
)


def test_temporal_enum_string_values():
    """Test that temporal dependency enum values match expected strings."""
    assert TemporalDependency.DIRECT == "direct"
    assert TemporalDependency.TRUE_EVENTUAL == "true_eventual"
    assert TemporalDependency.EVENTUAL == "eventual"
    assert TemporalDependency.NO_ORDERING == "no_ordering"


def test_existential_enum_string_values():
    """Test that existential dependency enum values match expected strings."""
    assert ExistentialDependency.IMPLICATION == "implication"
    assert ExistentialDependency.EQUIVALENCE == "equivalence"
    assert ExistentialDependency.NEGATED_EQUIVALENCE == "negated_equivalence"
    assert ExistentialDependency.OR == "or"
    assert ExistentialDependency.NAND == "nand"
    assert ExistentialDependency.INDEPENDENCE == "independence"


def test_dependency_cell_defaults_to_neutral():
    """Test that DependencyCell defaults to neutral state."""
    cell = DependencyCell()
    assert cell.temporal == TemporalDependency.NO_ORDERING
    assert cell.existential == ExistentialDependency.INDEPENDENCE
    assert cell.is_neutral() is True


def test_dependency_cell_accepts_enum_values():
    """Test that DependencyCell accepts enum values directly."""
    cell = DependencyCell(
        temporal=TemporalDependency.DIRECT,
        existential=ExistentialDependency.IMPLICATION,
    )
    assert cell.temporal == TemporalDependency.DIRECT
    assert cell.existential == ExistentialDependency.IMPLICATION
    assert cell.is_neutral() is False


def test_dependency_cell_normalizes_string_input():
    """Test that DependencyCell normalizes various string inputs."""
    # Test with exact string
    cell1 = DependencyCell(temporal="direct", existential="implication")
    assert cell1.temporal == TemporalDependency.DIRECT
    assert cell1.existential == ExistentialDependency.IMPLICATION

    # Test with capitalization
    cell2 = DependencyCell(temporal="Direct", existential="Implication")
    assert cell2.temporal == TemporalDependency.DIRECT
    assert cell2.existential == ExistentialDependency.IMPLICATION

    # Test with spaces
    cell3 = DependencyCell(temporal="true eventual", existential="independence")
    assert cell3.temporal == TemporalDependency.TRUE_EVENTUAL
    assert cell3.existential == ExistentialDependency.INDEPENDENCE

    # Test with mixed case and spaces
    cell4 = DependencyCell(temporal="True Eventual", existential="Independence")
    assert cell4.temporal == TemporalDependency.TRUE_EVENTUAL
    assert cell4.existential == ExistentialDependency.INDEPENDENCE


def test_dependency_cell_is_immutable():
    """Test that DependencyCell is immutable (frozen=True)."""
    cell = DependencyCell(temporal=TemporalDependency.DIRECT)

    with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
        cell.temporal = TemporalDependency.EVENTUAL


def test_is_neutral_returns_false_for_non_neutral():
    """Test that is_neutral() returns False for non-neutral cells."""
    cell1 = DependencyCell(temporal=TemporalDependency.DIRECT)
    assert cell1.is_neutral() is False

    cell2 = DependencyCell(existential=ExistentialDependency.IMPLICATION)
    assert cell2.is_neutral() is False

    cell3 = DependencyCell(
        temporal=TemporalDependency.EVENTUAL,
        existential=ExistentialDependency.NEGATED_EQUIVALENCE,
    )
    assert cell3.is_neutral() is False


def test_temporal_backward_values():
    """Test that backward temporal dependency enum values exist."""
    assert TemporalDependency.DIRECT_BACKWARD == "direct_backward"
    assert TemporalDependency.EVENTUAL_BACKWARD == "eventual_backward"
    assert TemporalDependency.TRUE_EVENTUAL_BACKWARD == "true_eventual_backward"


def test_existential_backward_values():
    """Test that backward existential dependency enum values exist."""
    assert ExistentialDependency.IMPLICATION_BACKWARD == "implication_backward"


def test_temporal_inverse_completeness():
    """Test TEMPORAL_INVERSE completeness and symmetry."""
    for dep in TemporalDependency:
        # Every value has an inverse
        assert dep in TEMPORAL_INVERSE, f"{dep} missing from TEMPORAL_INVERSE"

        # Inverse of inverse equals original
        inverse = TEMPORAL_INVERSE[dep]
        assert TEMPORAL_INVERSE[inverse] == dep, f"TEMPORAL_INVERSE[{inverse}] != {dep}"


def test_temporal_inverse_symmetric_values():
    """Test that symmetric temporal values map to themselves."""
    assert TEMPORAL_INVERSE[TemporalDependency.INDEPENDENCE] == TemporalDependency.INDEPENDENCE
    assert TEMPORAL_INVERSE[TemporalDependency.NO_ORDERING] == TemporalDependency.NO_ORDERING


def test_existential_inverse_completeness():
    """Test EXISTENTIAL_INVERSE completeness and symmetry."""
    for dep in ExistentialDependency:
        # Every value has an inverse
        assert dep in EXISTENTIAL_INVERSE, f"{dep} missing from EXISTENTIAL_INVERSE"

        # Inverse of inverse equals original
        inverse = EXISTENTIAL_INVERSE[dep]
        assert EXISTENTIAL_INVERSE[inverse] == dep, f"EXISTENTIAL_INVERSE[{inverse}] != {dep}"


def test_existential_inverse_symmetric_values():
    """Test that symmetric existential values map to themselves."""
    assert EXISTENTIAL_INVERSE[ExistentialDependency.EQUIVALENCE] == ExistentialDependency.EQUIVALENCE
    assert EXISTENTIAL_INVERSE[ExistentialDependency.NEGATED_EQUIVALENCE] == ExistentialDependency.NEGATED_EQUIVALENCE
    assert EXISTENTIAL_INVERSE[ExistentialDependency.OR] == ExistentialDependency.OR
    assert EXISTENTIAL_INVERSE[ExistentialDependency.NAND] == ExistentialDependency.NAND
    assert EXISTENTIAL_INVERSE[ExistentialDependency.INDEPENDENCE] == ExistentialDependency.INDEPENDENCE
