"""Granular percentage calculations for ARM classification.

Implements the percentage-based approach from the original Rust implementation:
https://github.com/INSM-TUM/automated-process-classification

Instead of aggregated ratios (direct_ratio, eventual_ratio, etc.), this computes
percentages for specific combinations of temporal and existential dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.core.matrix import Matrix


@dataclass
class CalculatedPercentages:
    """Percentages of specific temporal+existential dependency combinations.

    Matches the Rust implementation's CalculatedPercentages struct.
    Each field represents a specific combination of temporal and existential types.
    """

    # Primary Rule related percentages
    independence_none: float  # Temporal INDEPENDENCE AND no existential
    no_ordering_none: float  # Temporal NO_ORDERING AND no existential
    none_implication: float  # No temporal AND implication existential
    none_equivalence: float  # No temporal AND equivalence existential
    eventual_equivalence: float  # Eventual temporal AND equivalence existential
    eventual_implication: float  # Eventual temporal AND implication existential

    # Secondary Rule specific percentages
    none_negated_equivalence: float  # No temporal AND negated equivalence existential

    # Unstructured Rule related percentages
    eventual_any_existential: float  # Eventual temporal AND any existential
    direct_any_existential: float  # Direct temporal AND any existential
    direct_none: float  # Direct temporal AND no existential

    # Detailed metrics for differentiating Structured vs Semi-Structured
    true_eventual_ratio: float  # Ratio of TRUE_EVENTUAL / (TRUE_EVENTUAL + EVENTUAL) among eventual dependencies
    eventual_or: float  # Eventual temporal AND OR existential

    @property
    def none_none(self) -> float:
        """Backward compatibility for none_none (sum of independence and no_ordering)."""
        return self.independence_none + self.no_ordering_none

    @staticmethod
    def from_matrix(matrix: Matrix) -> CalculatedPercentages:
        """Calculate percentages from ARM matrix.

        Args:
            matrix: ARM matrix to analyze

        Returns:
            CalculatedPercentages with computed percentages

        Raises:
            ValueError: If matrix is empty
        """
        if len(matrix.activities) == 0:
            raise ValueError("Input matrix is empty")

        # Count all dependency combinations
        # Total = all non-neutral cells (excluding diagonal)
        total_entries = 0
        counts_independence_none = 0
        counts_no_ordering_none = 0
        counts_none_implication = 0
        counts_none_equivalence = 0
        counts_none_negated_equivalence = 0
        counts_eventual_equivalence = 0
        counts_eventual_implication = 0
        counts_eventual_any = 0
        counts_direct_any = 0
        counts_direct_none = 0

        # Detailed metrics
        counts_true_eventual_existential = 0
        counts_all_eventual_existential = 0
        counts_eventual_or = 0

        for source in matrix.activities:
            for target in matrix.activities:
                if source == target:
                    continue  # Skip diagonal

                cell = matrix.get_cell(source, target)
                if cell.is_neutral():
                    continue  # Skip neutral cells

                total_entries += 1

                # Classify dependency by temporal type
                # "None" temporal = INDEPENDENCE (no temporal ordering evidence)
                is_temporal_independence = cell.temporal == TemporalDependency.INDEPENDENCE
                is_temporal_no_ordering = cell.temporal == TemporalDependency.NO_ORDERING
                is_temporal_none = is_temporal_independence or is_temporal_no_ordering
                
                is_temporal_direct = cell.temporal in [
                    TemporalDependency.DIRECT,
                    TemporalDependency.DIRECT_BACKWARD,
                ]
                is_temporal_eventual = cell.temporal in [
                    TemporalDependency.TRUE_EVENTUAL,
                    TemporalDependency.TRUE_EVENTUAL_BACKWARD,
                    TemporalDependency.EVENTUAL,
                    TemporalDependency.EVENTUAL_BACKWARD,
                ]

                # Classify by existential type
                has_implication = cell.existential in [
                    ExistentialDependency.IMPLICATION,
                    ExistentialDependency.IMPLICATION_BACKWARD,
                ]
                has_equivalence = cell.existential == ExistentialDependency.EQUIVALENCE
                has_negated_equivalence = cell.existential == ExistentialDependency.NEGATED_EQUIVALENCE
                has_or = cell.existential == ExistentialDependency.OR
                has_any_existential = has_implication or has_equivalence

                # Count combinations
                if is_temporal_none:
                    if has_implication:
                        counts_none_implication += 1
                    elif has_equivalence:
                        counts_none_equivalence += 1
                    elif has_negated_equivalence:
                        counts_none_negated_equivalence += 1
                    else:
                        if is_temporal_independence:
                            counts_independence_none += 1
                        else:
                            counts_no_ordering_none += 1

                elif is_temporal_eventual:
                    if has_any_existential:
                        counts_eventual_any += 1
                        
                        counts_all_eventual_existential += 1
                        if cell.temporal in [
                            TemporalDependency.TRUE_EVENTUAL, 
                            TemporalDependency.TRUE_EVENTUAL_BACKWARD
                        ]:
                            counts_true_eventual_existential += 1
                        
                        if has_equivalence:
                            counts_eventual_equivalence += 1
                        elif has_implication:
                            counts_eventual_implication += 1
                    elif has_or:
                        counts_eventual_or += 1
                        # We also count OR as an eventual existential for structure purposes?
                        # Rust code treated OR as "none_none". 
                        # But for p03 we need to see it.
                        counts_all_eventual_existential += 1

                elif is_temporal_direct:
                    if has_any_existential:
                        counts_direct_any += 1
                    else:
                        counts_direct_none += 1

                # Other temporal types (INDEPENDENCE) or other existential types
                # (NAND, OR, XOR) count as none_none
                elif not has_any_existential:
                    if is_temporal_independence:
                        counts_independence_none += 1
                    else:
                        # Fallback for NO_ORDERING or other non-direct/non-eventual
                        counts_no_ordering_none += 1

        if total_entries == 0:
            raise ValueError("Matrix has no dependencies")

        total_f = float(total_entries)

        true_eventual_ratio = (
            counts_true_eventual_existential / float(counts_all_eventual_existential)
            if counts_all_eventual_existential > 0
            else 0.0
        )

        return CalculatedPercentages(
            independence_none=counts_independence_none / total_f,
            no_ordering_none=counts_no_ordering_none / total_f,
            none_implication=counts_none_implication / total_f,
            none_equivalence=counts_none_equivalence / total_f,
            eventual_equivalence=counts_eventual_equivalence / total_f,
            eventual_implication=counts_eventual_implication / total_f,
            none_negated_equivalence=counts_none_negated_equivalence / total_f,
            eventual_any_existential=counts_eventual_any / total_f,
            direct_any_existential=counts_direct_any / total_f,
            direct_none=counts_direct_none / total_f,
            true_eventual_ratio=true_eventual_ratio,
            eventual_or=counts_eventual_or / total_f,
        )
