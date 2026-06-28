"""
Core data structures (Matrix, relationships, YAML serialization).
"""

from armature.core.matrix import Matrix
from armature.core.dependencies import (
    DependencyCell,
    TemporalDependency,
    ExistentialDependency,
)

__all__ = [
    "Matrix",
    "DependencyCell",
    "TemporalDependency",
    "ExistentialDependency",
]
