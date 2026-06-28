"""Drift distance metrics for ARM matrix comparison.

Provides weighted distance metrics for comparing ARM matrices across
different windows of an event log. Distances capture both temporal
ordering changes and existential co-occurrence changes.
"""
from dataclasses import dataclass
from typing import Dict, Set, Tuple

from armature.core.dependencies import (
    DependencyCell,
    TemporalDependency,
    ExistentialDependency,
)
from armature.core.matrix import Matrix


# Temporal change weights (higher = more severe change)
# DIRECT -> NONE is worse than DIRECT -> EVENTUAL
TEMPORAL_WEIGHTS: Dict[Tuple[TemporalDependency, TemporalDependency], float] = {
    # Complete loss of relationship
    (TemporalDependency.DIRECT, TemporalDependency.NO_ORDERING): 3.0,
    (TemporalDependency.EVENTUAL, TemporalDependency.NO_ORDERING): 2.5,
    (TemporalDependency.TRUE_EVENTUAL, TemporalDependency.NO_ORDERING): 2.0,
    # Backward equivalents
    (TemporalDependency.DIRECT_BACKWARD, TemporalDependency.NO_ORDERING): 3.0,
    (TemporalDependency.EVENTUAL_BACKWARD, TemporalDependency.NO_ORDERING): 2.5,
    (TemporalDependency.TRUE_EVENTUAL_BACKWARD, TemporalDependency.NO_ORDERING): 2.0,
    # Directness changes (less severe)
    (TemporalDependency.DIRECT, TemporalDependency.EVENTUAL): 1.5,
    (TemporalDependency.DIRECT, TemporalDependency.TRUE_EVENTUAL): 2.0,
    (TemporalDependency.EVENTUAL, TemporalDependency.TRUE_EVENTUAL): 1.0,
    (TemporalDependency.DIRECT_BACKWARD, TemporalDependency.EVENTUAL_BACKWARD): 1.5,
    (TemporalDependency.DIRECT_BACKWARD, TemporalDependency.TRUE_EVENTUAL_BACKWARD): 2.0,
    (TemporalDependency.EVENTUAL_BACKWARD, TemporalDependency.TRUE_EVENTUAL_BACKWARD): 1.0,
    # Direction reversal (significant)
    (TemporalDependency.DIRECT, TemporalDependency.DIRECT_BACKWARD): 2.5,
    (TemporalDependency.EVENTUAL, TemporalDependency.EVENTUAL_BACKWARD): 2.0,
    # Independence gained/lost
    (TemporalDependency.DIRECT, TemporalDependency.INDEPENDENCE): 2.0,
    (TemporalDependency.EVENTUAL, TemporalDependency.INDEPENDENCE): 1.5,
    (TemporalDependency.INDEPENDENCE, TemporalDependency.DIRECT): 2.0,
    (TemporalDependency.INDEPENDENCE, TemporalDependency.EVENTUAL): 1.5,
}

# Existential change weights
EXISTENTIAL_WEIGHTS: Dict[Tuple[ExistentialDependency, ExistentialDependency], float] = {
    # Implication lost (significant)
    (ExistentialDependency.IMPLICATION, ExistentialDependency.INDEPENDENCE): 2.5,
    (ExistentialDependency.IMPLICATION_BACKWARD, ExistentialDependency.INDEPENDENCE): 2.5,
    (ExistentialDependency.EQUIVALENCE, ExistentialDependency.INDEPENDENCE): 3.0,
    # NEGATED_EQUIVALENCE/NAND/OR changes (structural)
    (ExistentialDependency.NEGATED_EQUIVALENCE, ExistentialDependency.OR): 2.0,
    (ExistentialDependency.NAND, ExistentialDependency.OR): 1.5,
    (ExistentialDependency.NEGATED_EQUIVALENCE, ExistentialDependency.INDEPENDENCE): 2.0,
    (ExistentialDependency.NAND, ExistentialDependency.INDEPENDENCE): 1.5,
    # Implication direction change
    (ExistentialDependency.IMPLICATION, ExistentialDependency.IMPLICATION_BACKWARD): 1.5,
}


@dataclass
class DriftScore:
    """Drift score between two ARM matrices."""

    cell_change_count: int
    temporal_distance: float
    existential_distance: float
    total_distance: float
    affected_activities: Set[str]
    activity_coverage: float
    total_cells_compared: int

    @property
    def normalized_distance(self) -> float:
        """Normalized total distance (0-1 scale).

        Uses change ratio: proportion of cells that changed.
        This is more robust than raw distance as it's independent of
        matrix size and provides consistent threshold interpretation.
        """
        if self.total_cells_compared == 0:
            return 0.0
        # Change ratio: what fraction of matrix cells changed
        # This ranges 0-1 and is comparable across different log sizes
        return self.cell_change_count / self.total_cells_compared

    @property
    def raw_distance_per_change(self) -> float:
        """Average distance per changed cell (unnormalized)."""
        if self.cell_change_count == 0:
            return 0.0
        return self.total_distance / self.cell_change_count


def compute_drift_score(matrix1: Matrix, matrix2: Matrix) -> DriftScore:
    """Compute drift score between two ARM matrices.

    Compares matrices cell-by-cell, computing weighted distances for
    temporal and existential dependency changes.

    Args:
        matrix1: First ARM matrix (typically from earlier window)
        matrix2: Second ARM matrix (typically from later window)

    Returns:
        DriftScore with detailed change metrics
    """
    # Get common activities
    common_acts = sorted(set(matrix1.activities) & set(matrix2.activities))

    cell_changes = 0
    temporal_dist = 0.0
    existential_dist = 0.0
    affected_activities: Set[str] = set()

    for a in common_acts:
        for b in common_acts:
            cell1 = matrix1[a, b]
            cell2 = matrix2[a, b]

            # Temporal comparison
            if cell1.temporal != cell2.temporal:
                cell_changes += 1
                affected_activities.add(a)
                affected_activities.add(b)

                # Get weight from mapping (or default)
                key = (cell1.temporal, cell2.temporal)
                reverse_key = (cell2.temporal, cell1.temporal)
                weight = TEMPORAL_WEIGHTS.get(key, TEMPORAL_WEIGHTS.get(reverse_key, 1.0))
                temporal_dist += weight

            # Existential comparison
            if cell1.existential != cell2.existential:
                cell_changes += 1
                affected_activities.add(a)
                affected_activities.add(b)

                key = (cell1.existential, cell2.existential)
                reverse_key = (cell2.existential, cell1.existential)
                weight = EXISTENTIAL_WEIGHTS.get(key, EXISTENTIAL_WEIGHTS.get(reverse_key, 1.0))
                existential_dist += weight

    total_cells = len(common_acts) * len(common_acts)
    activity_coverage = len(affected_activities) / len(common_acts) if common_acts else 0.0

    return DriftScore(
        cell_change_count=cell_changes,
        temporal_distance=temporal_dist,
        existential_distance=existential_dist,
        total_distance=temporal_dist + existential_dist,
        affected_activities=affected_activities,
        activity_coverage=activity_coverage,
        total_cells_compared=total_cells,
    )
