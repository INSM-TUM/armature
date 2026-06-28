"""Dependency types and cell model for ARM matrix.

This module defines the temporal and existential dependency enums along with
the DependencyCell model that represents a single relationship in the matrix.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class TemporalDependency(StrEnum):
    """Temporal dependency types representing ordering relationships.

    Forward dependencies: A before B (DIRECT, TRUE_EVENTUAL, EVENTUAL)
    Backward dependencies: B before A (DIRECT_BACKWARD, TRUE_EVENTUAL_BACKWARD, EVENTUAL_BACKWARD)
    Symmetric: INDEPENDENCE (both orderings in different traces), NO_ORDERING (no evidence)
    """

    DIRECT = "direct"
    TRUE_EVENTUAL = "true_eventual"
    EVENTUAL = "eventual"
    DIRECT_BACKWARD = "direct_backward"
    TRUE_EVENTUAL_BACKWARD = "true_eventual_backward"
    EVENTUAL_BACKWARD = "eventual_backward"
    INDEPENDENCE = "independence"
    NO_ORDERING = "no_ordering"


class ExistentialDependency(StrEnum):
    """Existential dependency types representing co-occurrence relationships.

    IMPLICATION: A=>B (whenever A occurs, B also occurs)
    IMPLICATION_BACKWARD: B=>A (whenever B occurs, A also occurs)
    Symmetric: EQUIVALENCE, NEGATED_EQUIVALENCE, OR, NAND, INDEPENDENCE
    """

    IMPLICATION = "implication"
    IMPLICATION_BACKWARD = "implication_backward"
    EQUIVALENCE = "equivalence"
    NEGATED_EQUIVALENCE = "negated_equivalence"
    OR = "or"
    NAND = "nand"
    INDEPENDENCE = "independence"


# Inverse mappings for matrix symmetry
TEMPORAL_INVERSE = {
    TemporalDependency.DIRECT: TemporalDependency.DIRECT_BACKWARD,
    TemporalDependency.DIRECT_BACKWARD: TemporalDependency.DIRECT,
    TemporalDependency.TRUE_EVENTUAL: TemporalDependency.TRUE_EVENTUAL_BACKWARD,
    TemporalDependency.TRUE_EVENTUAL_BACKWARD: TemporalDependency.TRUE_EVENTUAL,
    TemporalDependency.EVENTUAL: TemporalDependency.EVENTUAL_BACKWARD,
    TemporalDependency.EVENTUAL_BACKWARD: TemporalDependency.EVENTUAL,
    TemporalDependency.INDEPENDENCE: TemporalDependency.INDEPENDENCE,
    TemporalDependency.NO_ORDERING: TemporalDependency.NO_ORDERING,
}

EXISTENTIAL_INVERSE = {
    ExistentialDependency.IMPLICATION: ExistentialDependency.IMPLICATION_BACKWARD,
    ExistentialDependency.IMPLICATION_BACKWARD: ExistentialDependency.IMPLICATION,
    ExistentialDependency.EQUIVALENCE: ExistentialDependency.EQUIVALENCE,
    ExistentialDependency.NEGATED_EQUIVALENCE: ExistentialDependency.NEGATED_EQUIVALENCE,
    ExistentialDependency.OR: ExistentialDependency.OR,
    ExistentialDependency.NAND: ExistentialDependency.NAND,
    ExistentialDependency.INDEPENDENCE: ExistentialDependency.INDEPENDENCE,
}


class DependencyCell(BaseModel):
    """Represents a single cell in the ARM matrix.

    A DependencyCell combines temporal (ordering) and existential (co-occurrence)
    dependencies between two activities. The neutral state (default) represents
    no evidence of relationship.

    Attributes:
        temporal: Temporal dependency type (default: NO_ORDERING)
        existential: Existential dependency type (default: INDEPENDENCE)
    """

    temporal: TemporalDependency = Field(
        default=TemporalDependency.NO_ORDERING,
        description="Temporal relationship (ordering)",
    )
    existential: ExistentialDependency = Field(
        default=ExistentialDependency.INDEPENDENCE,
        description="Existential relationship (co-occurrence)",
    )

    model_config = {
        "frozen": True,  # Immutable cells
        "str_strip_whitespace": True,  # Auto-strip whitespace
    }

    @field_validator("temporal", "existential", mode="before")
    @classmethod
    def normalize_strings(cls, v):
        """Normalize string input to lowercase with underscores.

        Allows flexible input like "Direct" or "true eventual" to be
        automatically converted to the correct enum values.

        Args:
            v: The input value (string or enum)

        Returns:
            Normalized string or original enum value
        """
        if isinstance(v, str):
            return v.lower().replace(" ", "_").replace("-", "_")
        return v

    def is_neutral(self) -> bool:
        """Check if this cell represents a neutral (default) state.

        A neutral cell has no evidence of ordering (NO_ORDERING) and
        no evidence of co-occurrence (INDEPENDENCE).

        Returns:
            True if both temporal and existential are in their neutral state
        """
        return (
            self.temporal == TemporalDependency.NO_ORDERING and self.existential == ExistentialDependency.INDEPENDENCE
        )
