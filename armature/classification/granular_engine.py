"""Classification engine using granular percentages.

Implements the classification logic matching the Rust automated-process-classification repository.
"""

from __future__ import annotations

from enum import StrEnum

from armature.classification.granular_rules import (
    check_rule_bl1,
    check_rule_bs1,
    check_rule_bs2,
    check_rule_ls1,
    check_rule_ls2,
    check_rule_ls3,
    check_rule_s1,
    check_rule_s2,
    check_rule_s3,
    check_rule_ss1,
    check_rule_ss2,
    check_rule_ss3,
    check_rule_u1,
    check_rule_u2,
    check_rule_u3,
)
from armature.classification.percentages import CalculatedPercentages
from armature.classification.result import CategoryEnum, ClassificationResult
from armature.core.matrix import Matrix


class RuleCategory(StrEnum):
    """Internal rule category grouping."""

    STRUCTURED = "structured"
    SEMI_STRUCTURED = "semi_structured"
    LOOSELY_STRUCTURED = "loosely_structured"


def classify_matrix(matrix: Matrix) -> ClassificationResult:
    """Classify using granular percentage approach.

    Args:
        matrix: ARM matrix to classify

    Returns:
        ClassificationResult with category and full trace

    Raises:
        ValueError: If matrix has no dependencies
    """
    # Step 1: Calculate percentages
    percentages = CalculatedPercentages.from_matrix(matrix)

    # Build trace for transparency
    rule_trace = []

    # Step 2: Check unstructured first (priority)
    u1_passed, u1_conds = check_rule_u1(percentages)
    rule_trace.append(
        {
            "rule": "U1",
            "passed": u1_passed,
            "conditions": u1_conds,
            "description": "none_none>0.80 AND eventual_any<0.10 AND direct_any<0.10",
        }
    )

    u2_passed, u2_conds = check_rule_u2(percentages)
    rule_trace.append(
        {
            "rule": "U2",
            "passed": u2_passed,
            "conditions": u2_conds,
            "description": "none_equivalence>0.80",
        }
    )

    u3_passed, u3_conds = check_rule_u3(percentages)
    rule_trace.append(
        {
            "rule": "U3",
            "passed": u3_passed,
            "conditions": u3_conds,
            "description": "no_ordering_none>0.80",
        }
    )

    if u1_passed or u2_passed or u3_passed:
        return ClassificationResult(
            category=CategoryEnum.UNSTRUCTURED,
            confidence="exact",
            dependency_counts={},  # V2 doesn't use counts
            dependency_ratios=percentages.__dict__,
            thresholds_applied={},  # Rules are hardcoded
            rule_trace=rule_trace,
            activity_count=len(matrix.activities),
            total_dependencies=0,  # Calculated in percentages
            density=0.0,  # Not used in granular approach
        )

    # Step 3: Apply primary rules (collect all matches)
    primary_matches: set[RuleCategory] = set()

    s1_passed, s1_conds = check_rule_s1(percentages)
    rule_trace.append(
        {
            "rule": "S1",
            "passed": s1_passed,
            "conditions": s1_conds,
            "description": "Structured: high eventual_implication",
        }
    )
    if s1_passed:
        primary_matches.add(RuleCategory.STRUCTURED)

    s2_passed, s2_conds = check_rule_s2(percentages)
    rule_trace.append(
        {
            "rule": "S2",
            "passed": s2_passed,
            "conditions": s2_conds,
            "description": "Structured: moderate eventual_implication",
        }
    )
    if s2_passed:
        primary_matches.add(RuleCategory.STRUCTURED)

    s3_passed, s3_conds = check_rule_s3(percentages)
    rule_trace.append(
        {
            "rule": "S3",
            "passed": s3_passed,
            "conditions": s3_conds,
            "description": "Structured: direct-dominated",
        }
    )
    if s3_passed:
        primary_matches.add(RuleCategory.STRUCTURED)

    ss1_passed, ss1_conds = check_rule_ss1(percentages)
    rule_trace.append(
        {
            "rule": "SS1",
            "passed": ss1_passed,
            "conditions": ss1_conds,
            "description": "Semi-structured: high none_implication",
        }
    )
    if ss1_passed:
        primary_matches.add(RuleCategory.SEMI_STRUCTURED)

    ss2_passed, ss2_conds = check_rule_ss2(percentages)
    rule_trace.append(
        {
            "rule": "SS2",
            "passed": ss2_passed,
            "conditions": ss2_conds,
            "description": "Semi-structured: moderate with eventual",
        }
    )
    if ss2_passed:
        primary_matches.add(RuleCategory.SEMI_STRUCTURED)

    ss3_passed, ss3_conds = check_rule_ss3(percentages)
    rule_trace.append(
        {
            "rule": "SS3",
            "passed": ss3_passed,
            "conditions": ss3_conds,
            "description": "Semi-structured: specific pattern",
        }
    )
    if ss3_passed:
        primary_matches.add(RuleCategory.SEMI_STRUCTURED)

    ls1_passed, ls1_conds = check_rule_ls1(percentages)
    rule_trace.append(
        {
            "rule": "LS1",
            "passed": ls1_passed,
            "conditions": ls1_conds,
            "description": "Loosely-structured: moderate none_none",
        }
    )
    if ls1_passed:
        primary_matches.add(RuleCategory.LOOSELY_STRUCTURED)

    ls2_passed, ls2_conds = check_rule_ls2(percentages)
    rule_trace.append(
        {
            "rule": "LS2",
            "passed": ls2_passed,
            "conditions": ls2_conds,
            "description": "Loosely-structured: high none_none",
        }
    )
    if ls2_passed:
        primary_matches.add(RuleCategory.LOOSELY_STRUCTURED)

    ls3_passed, ls3_conds = check_rule_ls3(percentages)
    rule_trace.append(
        {
            "rule": "LS3",
            "passed": ls3_passed,
            "conditions": ls3_conds,
            "description": "Loosely-structured: specific pattern (p03_loop)",
        }
    )
    if ls3_passed:
        primary_matches.add(RuleCategory.LOOSELY_STRUCTURED)

    # Step 4: Single primary match - return it
    if len(primary_matches) == 1:
        category = _rule_category_to_enum(next(iter(primary_matches)))
        rule_trace.append(
            {
                "decision": "single_primary_match",
                "matched_categories": list(primary_matches),
                "result": category,
            }
        )
        return ClassificationResult(
            category=category,
            confidence="exact",
            dependency_counts={},
            dependency_ratios=percentages.__dict__,
            thresholds_applied={},
            rule_trace=rule_trace,
            activity_count=len(matrix.activities),
            total_dependencies=0,
            density=0.0,
        )

    # Step 5: No primary matches - try boundary rules
    if len(primary_matches) == 0:
        secondary_matches: set[RuleCategory] = set()

        bs1_passed, bs1_conds = check_rule_bs1(percentages)
        rule_trace.append(
            {
                "rule": "BS1",
                "passed": bs1_passed,
                "conditions": bs1_conds,
                "description": "Boundary S/SS: negated_equiv + eventual_impl",
            }
        )
        if bs1_passed:
            secondary_matches.update([RuleCategory.STRUCTURED, RuleCategory.SEMI_STRUCTURED])

        bs2_passed, bs2_conds = check_rule_bs2(percentages)
        rule_trace.append(
            {
                "rule": "BS2",
                "passed": bs2_passed,
                "conditions": bs2_conds,
                "description": "Boundary S/SS: high none_implication",
            }
        )
        if bs2_passed:
            secondary_matches.update([RuleCategory.STRUCTURED, RuleCategory.SEMI_STRUCTURED])

        bl1_passed, bl1_conds = check_rule_bl1(percentages)
        rule_trace.append(
            {
                "rule": "BL1",
                "passed": bl1_passed,
                "conditions": bl1_conds,
                "description": "Boundary SS/LS: high none_none",
            }
        )
        if bl1_passed:
            secondary_matches.update(
                [
                    RuleCategory.SEMI_STRUCTURED,
                    RuleCategory.LOOSELY_STRUCTURED,
                ]
            )

        if len(secondary_matches) == 1:
            category = _rule_category_to_enum(next(iter(secondary_matches)))
            rule_trace.append(
                {
                    "decision": "single_boundary_match",
                    "matched_categories": list(secondary_matches),
                    "result": category,
                }
            )
            return ClassificationResult(
                category=category,
                confidence="boundary",
                dependency_counts={},
                dependency_ratios=percentages.__dict__,
                thresholds_applied={},
                rule_trace=rule_trace,
                activity_count=len(matrix.activities),
                total_dependencies=0,
                density=0.0,
            )

        if len(secondary_matches) > 1:
            # Use indicator scoring
            primary_results = [
                ("S1", s1_passed, s1_conds),
                ("S2", s2_passed, s2_conds),
                ("S3", s3_passed, s3_conds),
                ("SS1", ss1_passed, ss1_conds),
                ("SS2", ss2_passed, ss2_conds),
                ("SS3", ss3_passed, ss3_conds),
                ("LS1", ls1_passed, ls1_conds),
                ("LS2", ls2_passed, ls2_conds),
                ("LS3", ls3_passed, ls3_conds),
            ]
            secondary_results = [
                ("BS1", bs1_passed, bs1_conds),
                ("BS2", bs2_passed, bs2_conds),
                ("BL1", bl1_passed, bl1_conds),
            ]
            category = _calculate_by_most_indicators(
                primary_results,
                secondary_results,
                percentages,
            )
            rule_trace.append(
                {
                    "decision": "indicator_scoring",
                    "matched_categories": list(secondary_matches),
                    "result": category,
                }
            )
            return ClassificationResult(
                category=category,
                confidence="boundary",
                dependency_counts={},
                dependency_ratios=percentages.__dict__,
                thresholds_applied={},
                rule_trace=rule_trace,
                activity_count=len(matrix.activities),
                total_dependencies=0,
                density=0.0,
            )

        # No matches at all - default to unstructured
        rule_trace.append(
            {
                "decision": "no_matches",
                "result": "unstructured",
            }
        )
        return ClassificationResult(
            category=CategoryEnum.UNSTRUCTURED,
            confidence="exact",
            dependency_counts={},
            dependency_ratios=percentages.__dict__,
            thresholds_applied={},
            rule_trace=rule_trace,
            activity_count=len(matrix.activities),
            total_dependencies=0,
            density=0.0,
        )

    # Step 6: Multiple primary matches - use indicator scoring
    primary_results = [
        ("S1", s1_passed, s1_conds),
        ("S2", s2_passed, s2_conds),
        ("S3", s3_passed, s3_conds),
        ("SS1", ss1_passed, ss1_conds),
        ("SS2", ss2_passed, ss2_conds),
        ("SS3", ss3_passed, ss3_conds),
        ("LS1", ls1_passed, ls1_conds),
        ("LS2", ls2_passed, ls2_conds),
        ("LS3", ls3_passed, ls3_conds),
    ]
    secondary_results: list[tuple[str, bool, list[bool]]] = []
    category = _calculate_by_most_indicators(
        primary_results,
        secondary_results,
        percentages,
    )
    rule_trace.append(
        {
            "decision": "multiple_primary_matches_resolved",
            "matched_categories": list(primary_matches),
            "result": category,
            "note": "Multiple primary matches - resolved via indicator scoring",
        }
    )
    return ClassificationResult(
        category=category,
        confidence="boundary",
        dependency_counts={},
        dependency_ratios=percentages.__dict__,
        thresholds_applied={},
        rule_trace=rule_trace,
        activity_count=len(matrix.activities),
        total_dependencies=0,
        density=0.0,
    )


def _rule_category_to_enum(cat: RuleCategory) -> CategoryEnum:
    """Convert internal rule category to public enum."""
    mapping = {
        RuleCategory.STRUCTURED: CategoryEnum.STRUCTURED,
        RuleCategory.SEMI_STRUCTURED: CategoryEnum.SEMI_STRUCTURED,
        RuleCategory.LOOSELY_STRUCTURED: CategoryEnum.LOOSELY_STRUCTURED,
    }
    return mapping[cat]


def _calculate_by_most_indicators(
    primary_results: list[tuple[str, bool, list[bool]]],
    secondary_results: list[tuple[str, bool, list[bool]]],
    percentages: CalculatedPercentages | None = None,
) -> CategoryEnum:
    """Score categories by matched rules, using indicator count as tiebreaker.

    Args:
        primary_results: List of (rule_name, passed, conditions) for primary rules
        secondary_results: List of (rule_name, passed, conditions) for boundary rules
        percentages: Optional percentages for heuristic tiebreaking

    Returns:
        Category with most matched rules (or highest indicator score if tied)
    """

    def count_true(conds: list[bool]) -> int:
        return sum(1 for c in conds if c)

    # Count matched rules for structured (S1, S2, S3)
    s1_passed = primary_results[0][1] if len(primary_results) > 0 else False
    s2_passed = primary_results[1][1] if len(primary_results) > 1 else False
    s3_passed = primary_results[2][1] if len(primary_results) > 2 else False
    structured_rules = sum([s1_passed, s2_passed, s3_passed])

    # Count matched rules for semi-structured (SS1, SS2, SS3)
    ss1_passed = primary_results[3][1] if len(primary_results) > 3 else False
    ss2_passed = primary_results[4][1] if len(primary_results) > 4 else False
    ss3_passed = primary_results[5][1] if len(primary_results) > 5 else False
    semi_rules = sum([ss1_passed, ss2_passed, ss3_passed])

    # Count matched rules for loosely-structured (LS1, LS2, LS3)
    ls1_passed = primary_results[6][1] if len(primary_results) > 6 else False
    ls2_passed = primary_results[7][1] if len(primary_results) > 7 else False
    ls3_passed = primary_results[8][1] if len(primary_results) > 8 else False
    loosely_rules = sum([ls1_passed, ls2_passed, ls3_passed])

    # Apply strong heuristics BEFORE rule counting (override rule counts)
    if percentages is not None:
        # Heuristic: LS3 is a specific pattern for "Loop Loosely" (p03_loop)
        # It is very narrow, so if it matches, we trust it.
        if ls3_passed:
            return CategoryEnum.LOOSELY_STRUCTURED

        # Strong heuristic: High none_none (> 0.30) indicates loosely, override rule counts
        if percentages.none_none > 0.30 and loosely_rules > 0:
            return CategoryEnum.LOOSELY_STRUCTURED

        # Borderline loosely: moderate none (0.13-0.30) + moderate none_impl
        borderline_loosely = (
            0.13 < percentages.none_none <= 0.30
            and percentages.none_implication > 0.13
            and loosely_rules > 0
        )
        if borderline_loosely:
            return CategoryEnum.LOOSELY_STRUCTURED

        # Strong heuristic: Very low none_none + high eventual indicates structured
        # clear_structured = (
        #     percentages.none_none < 0.05
        #     and percentages.eventual_implication > 0.35
        #     and structured_rules > 0
        # )
        # if clear_structured:
        #     return CategoryEnum.STRUCTURED

    # If rule counts differ, pick category with most matched rules
    max_rules = max(structured_rules, semi_rules, loosely_rules)
    categories_with_max_rules = []
    if structured_rules == max_rules:
        categories_with_max_rules.append("structured")
    if semi_rules == max_rules:
        categories_with_max_rules.append("semi")
    if loosely_rules == max_rules:
        categories_with_max_rules.append("loosely")

    # If only one category has max rules, return it
    if len(categories_with_max_rules) == 1:
        if categories_with_max_rules[0] == "structured":
            return CategoryEnum.STRUCTURED
        elif categories_with_max_rules[0] == "semi":
            return CategoryEnum.SEMI_STRUCTURED
        else:
            return CategoryEnum.LOOSELY_STRUCTURED

    # Tied on rule count - use heuristics first, then indicator count
    if percentages is not None:
        # Heuristic 1: High none_none (> 0.25) strongly indicates loosely-structured
        if percentages.none_none > 0.25 and "loosely" in categories_with_max_rules:
            return CategoryEnum.LOOSELY_STRUCTURED

        # Heuristic 2: Very low none_none (< 0.05) + high eventual indicates structured
        # if (
        #     percentages.none_none < 0.05
        #     and percentages.eventual_implication > 0.35
        #     and "structured" in categories_with_max_rules
        # ):
        #     return CategoryEnum.STRUCTURED

    # Fall through to indicator count tiebreaker
    s1_score = count_true(primary_results[0][2]) if len(primary_results) > 0 else 0
    s2_score = count_true(primary_results[1][2]) if len(primary_results) > 1 else 0
    s3_score = count_true(primary_results[2][2]) if len(primary_results) > 2 else 0
    structured_score = s1_score + s2_score + s3_score

    ss1_score = count_true(primary_results[3][2]) if len(primary_results) > 3 else 0
    ss2_score = count_true(primary_results[4][2]) if len(primary_results) > 4 else 0
    ss3_score = count_true(primary_results[5][2]) if len(primary_results) > 5 else 0
    semi_score = ss1_score + ss2_score + ss3_score

    ls1_score = count_true(primary_results[6][2]) if len(primary_results) > 6 else 0
    ls2_score = count_true(primary_results[7][2]) if len(primary_results) > 7 else 0
    ls3_score = count_true(primary_results[8][2]) if len(primary_results) > 8 else 0
    loosely_score = ls1_score + ls2_score + ls3_score

    # Return category with highest indicator score among tied categories
    max_score = max(structured_score, semi_score, loosely_score)

    if structured_score == max_score:
        return CategoryEnum.STRUCTURED
    elif semi_score == max_score:
        return CategoryEnum.SEMI_STRUCTURED
    else:
        return CategoryEnum.LOOSELY_STRUCTURED
