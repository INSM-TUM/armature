"""Classification result model with full transparency output.

Provides structured result with category, metrics, thresholds, and rule trace.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class CategoryEnum(StrEnum):
    """Process structure categories."""

    UNSTRUCTURED = "unstructured"
    LOOSELY_STRUCTURED = "loosely_structured"
    SEMI_STRUCTURED = "semi_structured"
    STRUCTURED = "structured"


class ClassificationResult(BaseModel):
    """Complete classification result with transparency data.

    Includes category, metrics, thresholds applied, and full rule trace.
    Pattern 3 from 05-RESEARCH.md: complete reasoning output.
    """

    category: CategoryEnum = Field(description="Classification category")

    confidence: Literal["exact", "boundary"] = Field(description="Exact match or boundary/borderline case")

    dependency_counts: dict[str, int] = Field(
        description="Raw dependency counts (DIRECT, EVENTUAL, NAND, OR, IMPLICATION)"
    )

    dependency_ratios: dict[str, float] = Field(description="Computed ratios used in threshold comparisons")

    thresholds_applied: dict[str, float] = Field(description="Thresholds from config used in classification")

    rule_trace: list[dict] = Field(description="Complete rule evaluation trace from RuleTrace.to_list()")

    activity_count: int = Field(description="Number of activities in matrix", ge=0)

    total_dependencies: int = Field(description="Sum of all dependency counts", ge=0)

    density: float = Field(
        description="Dependency density: total / (activities * (activities - 1))",
        ge=0.0,
        le=1.0,
    )

    loop_count: int = Field(default=0, description="Number of loops (SCCs) if computed", ge=0)

    metadata: dict = Field(
        default_factory=dict, description="Optional metadata (e.g., scores, method)"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return self.model_dump_json(indent=indent)
