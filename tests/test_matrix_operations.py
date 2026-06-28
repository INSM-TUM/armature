"""Tests for Matrix cell access and batch operations."""

import pytest
from armature.core.matrix import Matrix
from armature.core.dependencies import (
    DependencyCell,
    TemporalDependency,
    ExistentialDependency,
)
from armature.core.operations import (
    AddActivity,
    RemoveActivity,
    SetDependency,
    MatrixOperation,
)


class TestCellAccess:
    """Tests for Pythonic cell access syntax."""

    def test_getitem_returns_neutral_for_empty_matrix(self):
        """__getitem__ returns neutral cell for empty matrix."""
        m = Matrix(activities=["A", "B"])
        cell = m["A", "B"]
        assert cell.is_neutral()
        assert cell.temporal == TemporalDependency.NO_ORDERING
        assert cell.existential == ExistentialDependency.INDEPENDENCE

    def test_getitem_returns_correct_cell_after_set_dependency(self):
        """__getitem__ returns correct cell after set_dependency."""
        m = Matrix(activities=["A", "B"])
        m.set_dependency("A", "B", temporal=TemporalDependency.DIRECT)
        cell = m["A", "B"]
        assert cell.temporal == TemporalDependency.DIRECT
        assert cell.existential == ExistentialDependency.INDEPENDENCE

    def test_getitem_with_invalid_key_raises_error(self):
        """__getitem__ with invalid key raises TypeError."""
        m = Matrix(activities=["A", "B"])
        with pytest.raises(TypeError, match="must be a tuple"):
            _ = m["A"]
        with pytest.raises(TypeError, match="must be a tuple"):
            _ = m["A", "B", "C"]

    def test_set_dependency_updates_temporal_only(self):
        """set_dependency with temporal only preserves existential."""
        m = Matrix(activities=["A", "B"])
        # First set both fields
        m.set_dependency(
            "A", "B",
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION
        )
        # Update only temporal
        m.set_dependency("A", "B", temporal=TemporalDependency.EVENTUAL)
        cell = m["A", "B"]
        assert cell.temporal == TemporalDependency.EVENTUAL
        assert cell.existential == ExistentialDependency.IMPLICATION

    def test_set_dependency_updates_existential_only(self):
        """set_dependency with existential only preserves temporal."""
        m = Matrix(activities=["A", "B"])
        # First set both fields
        m.set_dependency(
            "A", "B",
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION
        )
        # Update only existential
        m.set_dependency("A", "B", existential=ExistentialDependency.EQUIVALENCE)
        cell = m["A", "B"]
        assert cell.temporal == TemporalDependency.DIRECT
        assert cell.existential == ExistentialDependency.EQUIVALENCE

    def test_set_dependency_raises_for_nonexistent_source(self):
        """set_dependency raises ValueError for non-existent source."""
        m = Matrix(activities=["A", "B"])
        with pytest.raises(ValueError, match="Source activity 'X' not in activities"):
            m.set_dependency("X", "B", temporal=TemporalDependency.DIRECT)

    def test_set_dependency_raises_for_nonexistent_target(self):
        """set_dependency raises ValueError for non-existent target."""
        m = Matrix(activities=["A", "B"])
        with pytest.raises(ValueError, match="Target activity 'X' not in activities"):
            m.set_dependency("A", "X", temporal=TemporalDependency.DIRECT)


class TestOperationModels:
    """Tests for operation models."""

    def test_add_activity_model_validates(self):
        """AddActivity model validates activity name."""
        op = AddActivity(activity="Test")
        assert op.op == "add_activity"
        assert op.activity == "Test"

    def test_remove_activity_model_validates(self):
        """RemoveActivity model validates activity name."""
        op = RemoveActivity(activity="Test")
        assert op.op == "remove_activity"
        assert op.activity == "Test"

    def test_set_dependency_model_validates(self):
        """SetDependency model validates fields."""
        op = SetDependency(
            source="A",
            target="B",
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION
        )
        assert op.op == "set_dependency"
        assert op.source == "A"
        assert op.target == "B"
        assert op.temporal == TemporalDependency.DIRECT
        assert op.existential == ExistentialDependency.IMPLICATION

    def test_operations_are_immutable(self):
        """Operations are immutable (frozen=True)."""
        op = AddActivity(activity="Test")
        with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
            op.activity = "Modified"

    def test_operations_serialize_correctly(self):
        """Operations serialize correctly via model_dump()."""
        op = AddActivity(activity="Test")
        data = op.model_dump()
        assert data == {"op": "add_activity", "activity": "Test"}

        op2 = SetDependency(
            source="A",
            target="B",
            temporal=TemporalDependency.DIRECT,
            existential=None
        )
        data2 = op2.model_dump()
        assert data2["op"] == "set_dependency"
        assert data2["source"] == "A"
        assert data2["target"] == "B"
        assert data2["temporal"] == "direct"
        assert data2["existential"] is None


class TestBatchOperations:
    """Tests for atomic batch execution with rollback."""

    def test_successful_batch_applies_all_operations(self):
        """Successful batch applies all operations."""
        m = Matrix()
        operations = [
            AddActivity(activity="A"),
            AddActivity(activity="B"),
            SetDependency(source="A", target="B", temporal=TemporalDependency.DIRECT)
        ]
        m.apply_batch(operations)
        
        assert "A" in m.activities
        assert "B" in m.activities
        assert m["A", "B"].temporal == TemporalDependency.DIRECT

    def test_failed_batch_rolls_back_all_changes(self):
        """Failed batch (duplicate activity) rolls back all changes."""
        m = Matrix()
        operations = [
            AddActivity(activity="A"),
            AddActivity(activity="A")  # Duplicate - will fail
        ]
        
        with pytest.raises(ValueError, match="Batch operation failed at operation 1"):
            m.apply_batch(operations)
        
        # Matrix should be empty (rolled back)
        assert len(m.activities) == 0

    def test_failed_batch_preserves_original_state(self):
        """Failed batch with non-existent activity preserves original state."""
        m = Matrix(activities=["X", "Y"])
        m.set_dependency("X", "Y", temporal=TemporalDependency.EVENTUAL)
        
        operations = [
            AddActivity(activity="A"),
            SetDependency(source="A", target="Z", temporal=TemporalDependency.DIRECT)  # Z doesn't exist
        ]
        
        with pytest.raises(ValueError, match="Batch operation failed"):
            m.apply_batch(operations)
        
        # Should still have original state
        assert m.activities == ["X", "Y"]
        assert m["X", "Y"].temporal == TemporalDependency.EVENTUAL

    def test_batch_with_mixed_operations_executes_in_order(self):
        """Batch with mixed operations executes in order."""
        m = Matrix()
        operations = [
            AddActivity(activity="A"),
            AddActivity(activity="B"),
            AddActivity(activity="C"),
            SetDependency(source="A", target="B", temporal=TemporalDependency.DIRECT),
            SetDependency(source="B", target="C", temporal=TemporalDependency.EVENTUAL),
            RemoveActivity(activity="C")
        ]
        m.apply_batch(operations)
        
        assert m.activities == ["A", "B"]
        assert m["A", "B"].temporal == TemporalDependency.DIRECT
        # C was removed, so dependency B->C should be gone
        assert "C" not in m.activities

    def test_empty_batch_does_nothing(self):
        """Empty batch does nothing."""
        m = Matrix(activities=["A"])
        m.apply_batch([])
        assert m.activities == ["A"]

    def test_batch_with_only_set_dependency_works(self):
        """Batch with only SetDependency operations works."""
        m = Matrix(activities=["A", "B", "C"])
        operations = [
            SetDependency(source="A", target="B", temporal=TemporalDependency.DIRECT),
            SetDependency(source="B", target="C", existential=ExistentialDependency.IMPLICATION),
        ]
        m.apply_batch(operations)
        
        assert m["A", "B"].temporal == TemporalDependency.DIRECT
        assert m["B", "C"].existential == ExistentialDependency.IMPLICATION
