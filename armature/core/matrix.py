"""Matrix class for ARM (Activity Relationship Matrix).

This module defines the Matrix class that manages activities and their
dependency relationships in a sparse adjacency list structure.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from armature.core.dependencies import DependencyCell, ExistentialDependency, TemporalDependency


class Matrix(BaseModel):
    """ARM Matrix with temporal and existential dependencies.

    The Matrix uses a sparse adjacency list for efficient storage of
    dependency relationships. Activities are maintained in a deterministic
    list for consistent ordering across serialization.

    Attributes:
        format_version: Matrix format version (default: "2.0")
        description: Optional description of the matrix
        source: Optional source identifier (e.g., originating XES file)
        created_at: Optional ISO timestamp of creation
        classification: Optional cached classification result
        activities: Ordered list of activity names
        dependencies: Sparse storage of dependency cells (source -> target -> cell)
    """

    # Metadata fields
    format_version: str = Field(default="2.0", frozen=True)
    description: str = ""
    source: str | None = None
    created_at: str | None = None
    classification: str | None = None
    num_traces: int | None = None
    num_variants: int | None = None

    # Activity management
    activities: list[str] = Field(default_factory=list, description="Deterministic ordered list of activities")

    # Cell storage (sparse adjacency list)
    dependencies: dict[str, dict[str, DependencyCell]] = Field(
        default_factory=dict,
        description="Sparse storage: dependencies[source][target] = cell",
    )

    # Configuration
    model_config = {
        "validate_assignment": True,  # Re-validate on attribute changes
        "arbitrary_types_allowed": False,
    }

    @model_validator(mode="after")
    def validate_structural_integrity(self):
        """Ensure all dependency keys reference existing activities.

        This validator enforces structural integrity by checking that every
        source and target in the dependencies dict exists in the activities list.
        This fail-fast approach prevents corrupted internal state.

        Raises:
            ValueError: If any dependency key is not in activities list

        Returns:
            self for chaining
        """
        activity_set = set(self.activities)
        for source, targets in self.dependencies.items():
            if source not in activity_set:
                raise ValueError(f"Source activity '{source}' not in activities list")
            for target in targets.keys():
                if target not in activity_set:
                    raise ValueError(f"Target activity '{target}' not in activities list")
        return self

    def add_activity(self, name: str) -> None:
        """Add an activity to the matrix.

        Args:
            name: Name of the activity to add

        Raises:
            ValueError: If activity already exists
        """
        if name in self.activities:
            raise ValueError(f"Activity '{name}' already exists")
        self.activities.append(name)

    def remove_activity(self, name: str) -> None:
        """Remove an activity and all its dependencies.

        This method removes the activity from the activities list and cleans
        up all dependency cells where this activity appears as either source
        or target.

        Args:
            name: Name of the activity to remove

        Raises:
            ValueError: If activity not found
        """
        if name not in self.activities:
            raise ValueError(f"Activity '{name}' not found")

        # Remove from activities list
        self.activities.remove(name)

        # Remove from dependencies
        self.dependencies.pop(name, None)  # Remove as source
        for targets in self.dependencies.values():
            targets.pop(name, None)  # Remove as target

    def get_cell(self, source: str, target: str) -> DependencyCell:
        """Get dependency cell with neutral default.

        If no dependency exists between the source and target activities,
        returns a neutral cell (NO_ORDERING + INDEPENDENCE).

        Args:
            source: Source activity name
            target: Target activity name

        Returns:
            DependencyCell for the relationship, or neutral cell if not found
        """
        return self.dependencies.get(source, {}).get(target, DependencyCell())

    def set_cell(self, source: str, target: str, cell: DependencyCell) -> None:
        """Set dependency cell with validation.

        Args:
            source: Source activity name
            target: Target activity name
            cell: DependencyCell to set

        Raises:
            ValueError: If source or target activity not in matrix
        """
        if source not in self.activities:
            raise ValueError(f"Source activity '{source}' not in activities")
        if target not in self.activities:
            raise ValueError(f"Target activity '{target}' not in activities")

        if source not in self.dependencies:
            self.dependencies[source] = {}
        self.dependencies[source][target] = cell

    def __getitem__(self, key: tuple[str, str]) -> DependencyCell:
        """Get dependency cell using Pythonic syntax.

        Enables convenient access: cell = matrix["source", "target"]

        Args:
            key: Tuple of (source, target) activity names

        Returns:
            DependencyCell for the relationship, or neutral cell if not found

        Raises:
            TypeError: If key is not a 2-tuple
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Key must be a tuple of (source, target)")
        source, target = key
        return self.get_cell(source, target)

    def set_dependency(
        self,
        source: str,
        target: str,
        temporal: TemporalDependency | None = None,
        existential: ExistentialDependency | None = None,
    ) -> None:
        """Set dependency with granular field updates.

        Updates only the specified fields, preserving others from the current cell.
        This enables partial updates without having to read-modify-write explicitly.

        Args:
            source: Source activity name
            target: Target activity name
            temporal: Temporal dependency to set (preserves existing if None)
            existential: Existential dependency to set (preserves existing if None)

        Raises:
            ValueError: If source or target activity not in matrix
        """
        if source not in self.activities:
            raise ValueError(f"Source activity '{source}' not in activities")
        if target not in self.activities:
            raise ValueError(f"Target activity '{target}' not in activities")

        # Get current cell (or neutral default)
        current = self.get_cell(source, target)

        # Preserve unspecified fields from current cell
        new_temporal = temporal if temporal is not None else current.temporal
        new_existential = existential if existential is not None else current.existential

        # Create new cell with updated fields
        new_cell = DependencyCell(temporal=new_temporal, existential=new_existential)

        # Set the cell
        self.set_cell(source, target, new_cell)

    def apply_batch(self, operations: list) -> None:
        """Apply batch operations atomically with rollback on failure.

        Executes a list of operations in sequence. If any operation fails,
        all changes are rolled back and the matrix returns to its pre-batch state.

        Args:
            operations: List of MatrixOperation to execute

        Raises:
            ValueError: If any operation fails (with context about which one)
        """
        # Import here to avoid circular import
        from armature.core.operations import AddActivity, RemoveActivity, SetDependency

        # Create snapshot for rollback
        snapshot = self.model_copy(deep=True)

        try:
            # Execute each operation
            for idx, operation in enumerate(operations):
                match operation:
                    case AddActivity(activity=name):
                        self.add_activity(name)
                    case RemoveActivity(activity=name):
                        self.remove_activity(name)
                    case SetDependency(source=s, target=t, temporal=temp, existential=exist):
                        self.set_dependency(s, t, temp, exist)
        except Exception as e:
            # Rollback to snapshot
            self.activities = snapshot.activities
            self.dependencies = snapshot.dependencies
            # Re-raise with context
            raise ValueError(f"Batch operation failed at operation {idx}: {e}") from e
