"""Classification engine for ARM matrices.

Implements RCIS2026 paper classification algorithm with transparent rule evaluation.
"""

from __future__ import annotations

from armature.classification.config import EPSILON, ConfigLoader, ThresholdConfig
from armature.classification.result import CategoryEnum, ClassificationResult
from armature.classification.rules import (
    evaluate_loosely_structured_rules,
    evaluate_semi_structured_rules,
    evaluate_structured_rules,
    evaluate_unstructured_rules,
)
from armature.classification.trace import RuleTrace
from armature.core.dependencies import (
    ExistentialDependency,
    TemporalDependency,
)
from armature.core.matrix import Matrix


class ClassificationEngine:
    """Classification engine for ARM matrices.

    Classifies matrices into four categories based on dependency type ratios:
    - Structured: High direct and eventual ratios
    - Semi-structured: High implication, moderate eventual
    - Loosely-structured: High NAND/OR ratio
    - Unstructured: Default category
    """

    def __init__(self, config: ThresholdConfig | None = None):
        """Initialize classification engine.

        Args:
            config: Threshold configuration. If None, loads defaults.
        """
        self.config = config if config is not None else ConfigLoader.load()

    def _compute_dependency_counts(self, matrix: Matrix) -> dict[str, int]:
        """Compute counts of each dependency type from matrix.

        Counts all non-neutral cells. Each cell counted once by temporal type.
        Existential types tracked separately.

        Note: The "eventual" counter tracks EVENTUAL enum (distinct from TRUE_EVENTUAL).
        For ratio calculation, forward orderings include DIRECT + TRUE_EVENTUAL only.

        Args:
            matrix: ARM matrix to analyze

        Returns:
            Dict with counts for DIRECT, EVENTUAL, NAND, OR, IMPLICATION, OTHER
        """
        # Count cells by temporal type (each cell counted once)
        temporal_counts = {
            "direct": 0,
            "eventual": 0,
            "true_eventual": 0,
            "other": 0,  # INDEPENDENCE, backward types, etc.
        }

        # Count cells by existential type (separate tracking)
        existential_counts = {
            "nand": 0,
            "or": 0,
            "implication": 0,
        }

        # Iterate all dependencies in sparse structure
        for source, targets in matrix.dependencies.items():
            for target, cell in targets.items():
                # Skip neutral cells (no dependency)
                if cell.is_neutral():
                    continue

                # Count by temporal type (each cell once)
                if cell.temporal == TemporalDependency.DIRECT:
                    temporal_counts["direct"] += 1
                elif cell.temporal == TemporalDependency.TRUE_EVENTUAL:
                    temporal_counts["true_eventual"] += 1
                elif cell.temporal == TemporalDependency.EVENTUAL:
                    temporal_counts["eventual"] += 1
                else:
                    # INDEPENDENCE, backward types, or other
                    temporal_counts["other"] += 1

                # Track existential types
                if cell.existential == ExistentialDependency.IMPLICATION:
                    existential_counts["implication"] += 1
                elif cell.existential == ExistentialDependency.NAND:
                    existential_counts["nand"] += 1
                elif cell.existential == ExistentialDependency.OR:
                    existential_counts["or"] += 1

        # Combine into single dict
        return {
            "direct": temporal_counts["direct"],
            "eventual": temporal_counts["eventual"],
            "true_eventual": temporal_counts["true_eventual"],
            "other": temporal_counts["other"],
            "nand": existential_counts["nand"],
            "or": existential_counts["or"],
            "implication": existential_counts["implication"],
        }

    def _compute_dependency_ratios(self, counts: dict[str, int], total: int) -> dict[str, float]:
        """Compute ratios of dependency types.

        Args:
            counts: Dependency counts from _compute_dependency_counts
            total: Total number of dependencies

        Returns:
            Dict with ratios for direct, eventual, nand_or, implication

        Raises:
            ValueError: If total is 0 (empty matrix)
        """
        if total == 0:
            raise ValueError("Cannot classify empty matrix - no dependencies found")

        # Compute individual ratios
        direct_ratio = counts["direct"] / total
        # Eventual includes ALL forward orderings (DIRECT, TRUE_EVENTUAL, EVENTUAL)
        # DIRECT ⊂ EVENTUAL semantically in process mining (State decision 05-02)
        eventual_ratio = (counts["direct"] + counts["true_eventual"] + counts["eventual"]) / total
        implication_ratio = counts["implication"] / total
        # NAND/OR combined ratio (non-determinism indicator)
        nand_or_ratio = (counts["nand"] + counts["or"]) / total

        return {
            "direct_ratio": direct_ratio,
            "eventual_ratio": eventual_ratio,
            "implication_ratio": implication_ratio,
            "nand_or_ratio": nand_or_ratio,
        }

    def classify(self, matrix: Matrix) -> ClassificationResult:
        """Classify matrix into one of four categories.

        Applies threshold-based rules in order:
        1. Structured (highest precedence)
        2. Semi-structured
        3. Loosely-structured
        4. Unstructured (default)

        Args:
            matrix: ARM matrix to classify

        Returns:
            ClassificationResult with category, metrics, and full trace

        Raises:
            ValueError: If matrix has no dependencies
        """
        # Step 1: Compute dependency counts
        counts = self._compute_dependency_counts(matrix)

        # Step 2: Compute total (all non-neutral cells)
        total = counts["direct"] + counts["true_eventual"] + counts["eventual"] + counts["other"]

        # Step 3: Compute ratios (will raise if total == 0)
        ratios = self._compute_dependency_ratios(counts, total)

        # Step 4: Initialize trace
        trace = RuleTrace()

        # Step 5: Evaluate rules in precedence order
        category = CategoryEnum.UNSTRUCTURED  # Default
        confidence = "exact"

        if evaluate_structured_rules(ratios, self.config, trace):
            category = CategoryEnum.STRUCTURED
        elif evaluate_semi_structured_rules(ratios, self.config, trace):
            category = CategoryEnum.SEMI_STRUCTURED
        elif evaluate_loosely_structured_rules(ratios, self.config, trace):
            category = CategoryEnum.LOOSELY_STRUCTURED
        else:
            # Unstructured is default
            evaluate_unstructured_rules(ratios, self.config, trace)
            category = CategoryEnum.UNSTRUCTURED

        # Step 6: Check for boundary cases
        # If any ratio is within 2*EPSILON of threshold, mark as boundary
        boundary_detected = self._detect_boundary(ratios)
        if boundary_detected:
            confidence = "boundary"

        # Step 7: Compute matrix metadata
        activity_count = len(matrix.activities)
        # Density = total / (n * n) for directed graph with self-loops
        max_possible = activity_count * activity_count if activity_count > 0 else 1
        density = total / max_possible if max_possible > 0 else 0.0

        # Step 8: Return result
        return ClassificationResult(
            category=category,
            confidence=confidence,
            dependency_counts=counts,
            dependency_ratios=ratios,
            thresholds_applied=self.config.model_dump(),
            rule_trace=trace.to_list(),
            activity_count=activity_count,
            total_dependencies=total,
            density=density,
            loop_count=0,  # TODO: Compute from SCC if needed
        )

    def _detect_boundary(self, ratios: dict[str, float]) -> bool:
        """Detect if any ratio is near a threshold (boundary case).

        Args:
            ratios: Computed dependency ratios

        Returns:
            True if any ratio is within 2*EPSILON of a threshold
        """
        boundary_tolerance = 2 * EPSILON

        # Check structured thresholds
        direct_diff = abs(ratios["direct_ratio"] - self.config.direct_ratio_structured)
        if direct_diff < boundary_tolerance:
            return True
        eventual_diff = abs(ratios["eventual_ratio"] - self.config.eventual_ratio_structured)
        if eventual_diff < boundary_tolerance:
            return True

        # Check semi-structured thresholds
        direct_semi_diff = abs(ratios["direct_ratio"] - self.config.direct_ratio_semi_max)
        if direct_semi_diff < boundary_tolerance:
            return True
        eventual_semi_diff = abs(ratios["eventual_ratio"] - self.config.eventual_ratio_semi_min)
        if eventual_semi_diff < boundary_tolerance:
            return True
        impl_diff = abs(ratios["implication_ratio"] - self.config.implication_ratio_semi)
        if impl_diff < boundary_tolerance:
            return True

        # Check loosely-structured thresholds
        direct_loosely_diff = abs(ratios["direct_ratio"] - self.config.direct_ratio_loosely_max)
        if direct_loosely_diff < boundary_tolerance:
            return True
        nand_or_diff = abs(ratios["nand_or_ratio"] - self.config.nand_or_ratio_loosely)
        if nand_or_diff < boundary_tolerance:
            return True

        return False
