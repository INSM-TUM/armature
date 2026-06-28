"""Classification algorithms for ARM matrices.

Uses definition-based weighted scoring (CLASSIFICATION_WEIGHTS approach).
Public API for ARM matrix classification.
"""

from __future__ import annotations

from armature.classification.config import ConfigLoader, ThresholdConfig
from armature.classification.engine import ClassificationEngine
from armature.classification.result import CategoryEnum, ClassificationResult
from armature.classification.weighted_engine import classify_matrix
from armature.core.matrix import Matrix

__all__ = [
    "classify",
    "ClassificationEngine",
    "ClassificationResult",
    "CategoryEnum",
    "ThresholdConfig",
    "ConfigLoader",
]


def classify(matrix: Matrix) -> ClassificationResult:
    """Classify ARM matrix into structural category.

    Uses definition-based weighted scoring. Computes per-feature ratios over all
    ordered activity pairs, applies structural pre-filters, then resolves via
    weighted dot-product scores across S / SS / LS / U.

    Args:
        matrix: ARM matrix to classify

    Returns:
        ClassificationResult with category, scores, ratios, and decision trace

    Raises:
        ValueError: If matrix has no activity pairs
    """
    return classify_matrix(matrix)
