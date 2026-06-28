"""Rule trace accumulator for transparent classification reasoning.

Provides immutable step recording for every rule evaluation during classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RuleOutcome(StrEnum):
    """Outcome of a rule evaluation."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class TraceStep:
    """Immutable record of a single rule evaluation.

    frozen=True prevents mutation after creation, ensuring trace integrity.
    """

    rule_name: str  # e.g., "direct_for_structured"
    metric_name: str  # e.g., "direct_ratio"
    computed_value: float  # actual ratio computed from matrix
    threshold: float  # config threshold used for comparison
    operator: str  # ">=" | "<=" | ">" | "<"
    outcome: RuleOutcome  # PASSED | FAILED

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "computed_value": self.computed_value,
            "threshold": self.threshold,
            "operator": self.operator,
            "outcome": self.outcome.value,
        }


@dataclass
class RuleTrace:
    """Accumulator for rule evaluation steps.

    Collects TraceStep instances during classification for transparency.
    """

    steps: list[TraceStep] = field(default_factory=list)

    def record_step(
        self,
        rule_name: str,
        metric_name: str,
        computed_value: float,
        threshold: float,
        operator: str,
        outcome: RuleOutcome,
    ) -> None:
        """Record a rule evaluation step.

        Args:
            rule_name: Name of the rule being evaluated
            metric_name: Metric being checked
            computed_value: Computed value from matrix
            threshold: Threshold from config
            operator: Comparison operator
            outcome: Whether rule passed or failed
        """
        step = TraceStep(
            rule_name=rule_name,
            metric_name=metric_name,
            computed_value=computed_value,
            threshold=threshold,
            operator=operator,
            outcome=outcome,
        )
        self.steps.append(step)

    def to_list(self) -> list[dict]:
        """Serialize all steps to list of dicts for JSON."""
        return [step.to_dict() for step in self.steps]
