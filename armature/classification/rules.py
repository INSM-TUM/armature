"""Rule evaluation functions for ARM classification.

Implements threshold-based classification rules with epsilon tolerance and trace recording.
"""

from __future__ import annotations

from armature.classification.config import EPSILON, ThresholdConfig
from armature.classification.trace import RuleOutcome, RuleTrace


def compare_with_tolerance(value: float, threshold: float, operator: str) -> bool:
    """Compare floats with epsilon tolerance to handle rounding errors.

    Pattern from 05-RESEARCH.md Pitfall 2: Floating-point comparison needs epsilon.

    Args:
        value: Computed value from matrix
        threshold: Threshold from config
        operator: Comparison operator (>=, <=, >, <)

    Returns:
        Boolean result of comparison with epsilon tolerance

    Raises:
        ValueError: If operator is not recognized
    """
    if operator == ">=":
        return value >= threshold - EPSILON
    elif operator == "<=":
        return value <= threshold + EPSILON
    elif operator == ">":
        return value > threshold + EPSILON
    elif operator == "<":
        return value < threshold - EPSILON
    else:
        raise ValueError(f"Unknown operator: {operator}")


def evaluate_structured_rules(
    dependency_ratios: dict[str, float], config: ThresholdConfig, trace: RuleTrace
) -> bool:
    """Evaluate structured classification rules.

    Structured processes have high DIRECT and EVENTUAL ratios.

    Args:
        dependency_ratios: Computed ratios from matrix
        config: Threshold configuration
        trace: Rule trace accumulator

    Returns:
        True if all structured rules pass
    """
    direct_ratio = dependency_ratios.get("direct_ratio", 0.0)
    eventual_ratio = dependency_ratios.get("eventual_ratio", 0.0)

    # Rule 1: direct_ratio >= direct_ratio_structured
    direct_passes = compare_with_tolerance(direct_ratio, config.direct_ratio_structured, ">=")
    trace.record_step(
        rule_name="structured_direct_ratio",
        metric_name="direct_ratio",
        computed_value=direct_ratio,
        threshold=config.direct_ratio_structured,
        operator=">=",
        outcome=RuleOutcome.PASSED if direct_passes else RuleOutcome.FAILED,
    )

    # Rule 2: eventual_ratio >= eventual_ratio_structured
    eventual_passes = compare_with_tolerance(eventual_ratio, config.eventual_ratio_structured, ">=")
    trace.record_step(
        rule_name="structured_eventual_ratio",
        metric_name="eventual_ratio",
        computed_value=eventual_ratio,
        threshold=config.eventual_ratio_structured,
        operator=">=",
        outcome=RuleOutcome.PASSED if eventual_passes else RuleOutcome.FAILED,
    )

    return direct_passes and eventual_passes


def evaluate_semi_structured_rules(
    dependency_ratios: dict[str, float], config: ThresholdConfig, trace: RuleTrace
) -> bool:
    """Evaluate semi-structured classification rules.

    Semi-structured processes have high IMPLICATION and moderate EVENTUAL ratios,
    but lower DIRECT than structured.

    Args:
        dependency_ratios: Computed ratios from matrix
        config: Threshold configuration
        trace: Rule trace accumulator

    Returns:
        True if all semi-structured rules pass
    """
    implication_ratio = dependency_ratios.get("implication_ratio", 0.0)
    eventual_ratio = dependency_ratios.get("eventual_ratio", 0.0)
    direct_ratio = dependency_ratios.get("direct_ratio", 0.0)

    # Rule 1: implication_ratio >= implication_ratio_semi
    implication_passes = compare_with_tolerance(
        implication_ratio, config.implication_ratio_semi, ">="
    )
    trace.record_step(
        rule_name="semi_implication_ratio",
        metric_name="implication_ratio",
        computed_value=implication_ratio,
        threshold=config.implication_ratio_semi,
        operator=">=",
        outcome=RuleOutcome.PASSED if implication_passes else RuleOutcome.FAILED,
    )

    # Rule 2: eventual_ratio >= eventual_ratio_semi_min
    eventual_passes = compare_with_tolerance(
        eventual_ratio, config.eventual_ratio_semi_min, ">="
    )
    trace.record_step(
        rule_name="semi_eventual_ratio_min",
        metric_name="eventual_ratio",
        computed_value=eventual_ratio,
        threshold=config.eventual_ratio_semi_min,
        operator=">=",
        outcome=RuleOutcome.PASSED if eventual_passes else RuleOutcome.FAILED,
    )

    # Rule 3: direct_ratio < direct_ratio_semi_max (exclude structured)
    direct_passes = compare_with_tolerance(direct_ratio, config.direct_ratio_semi_max, "<=")
    trace.record_step(
        rule_name="semi_direct_ratio_max",
        metric_name="direct_ratio",
        computed_value=direct_ratio,
        threshold=config.direct_ratio_semi_max,
        operator="<=",
        outcome=RuleOutcome.PASSED if direct_passes else RuleOutcome.FAILED,
    )

    return implication_passes and eventual_passes and direct_passes


def evaluate_loosely_structured_rules(
    dependency_ratios: dict[str, float], config: ThresholdConfig, trace: RuleTrace
) -> bool:
    """Evaluate loosely-structured classification rules.

    Loosely-structured processes have higher NAND/OR ratios and lower DIRECT.

    Args:
        dependency_ratios: Computed ratios from matrix
        config: Threshold configuration
        trace: Rule trace accumulator

    Returns:
        True if all loosely-structured rules pass
    """
    nand_or_ratio = dependency_ratios.get("nand_or_ratio", 0.0)
    direct_ratio = dependency_ratios.get("direct_ratio", 0.0)
    eventual_ratio = dependency_ratios.get("eventual_ratio", 0.0)
    implication_ratio = dependency_ratios.get("implication_ratio", 0.0)

    # Exclude pure nand_or (all other ratios exactly 0) - that's unstructured
    if nand_or_ratio > 0 and direct_ratio == 0 and eventual_ratio == 0 and implication_ratio == 0:
        return False

    # Rule 1: nand_or_ratio >= nand_or_ratio_loosely
    nand_or_passes = compare_with_tolerance(nand_or_ratio, config.nand_or_ratio_loosely, ">=")
    trace.record_step(
        rule_name="loosely_nand_or_ratio",
        metric_name="nand_or_ratio",
        computed_value=nand_or_ratio,
        threshold=config.nand_or_ratio_loosely,
        operator=">=",
        outcome=RuleOutcome.PASSED if nand_or_passes else RuleOutcome.FAILED,
    )

    # Rule 2: direct_ratio < direct_ratio_loosely_max (exclude semi/structured)
    direct_passes = compare_with_tolerance(direct_ratio, config.direct_ratio_loosely_max, "<=")
    trace.record_step(
        rule_name="loosely_direct_ratio_max",
        metric_name="direct_ratio",
        computed_value=direct_ratio,
        threshold=config.direct_ratio_loosely_max,
        operator="<=",
        outcome=RuleOutcome.PASSED if direct_passes else RuleOutcome.FAILED,
    )

    return nand_or_passes and direct_passes


def evaluate_unstructured_rules(dependency_ratios: dict[str, float], config: ThresholdConfig, trace: RuleTrace) -> bool:
    """Evaluate unstructured classification rules.

    Unstructured is the default category when no other category matches.
    This function always returns True but records why other categories failed.

    Args:
        dependency_ratios: Computed ratios from matrix
        config: Threshold configuration
        trace: Rule trace accumulator

    Returns:
        Always True (default category)
    """
    # Record that we're in the default unstructured category
    trace.record_step(
        rule_name="unstructured_default",
        metric_name="category",
        computed_value=0.0,  # No specific metric for default
        threshold=0.0,
        operator="default",
        outcome=RuleOutcome.PASSED,
    )

    return True
