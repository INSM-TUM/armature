#!/usr/bin/env python3.12
"""Analyze which rule thresholds cause classification failures.

Compares calculated percentages against rule conditions for each test log
to identify which thresholds need adjustment for our discovery model's output distribution.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from armature.classification.percentages import CalculatedPercentages
from armature.classification.rules_v2 import (
    check_rule_bl1,
    check_rule_bs1,
    check_rule_bs2,
    check_rule_ls1,
    check_rule_ls2,
    check_rule_s1,
    check_rule_s2,
    check_rule_s3,
    check_rule_ss1,
    check_rule_ss2,
    check_rule_ss3,
    check_rule_u1,
    check_rule_u2,
)
from armature.discovery import discover
from armature.serialization.yaml_codec import YAMLCodec


def analyze_log(xes_path: Path, ground_truth: str) -> dict[str, Any]:
    """Analyze single log: percentages, rule results, failures.

    Args:
        xes_path: Path to XES file
        ground_truth: Expected category (structured/semi_structured/loosely_structured/unstructured)

    Returns:
        Dict with log info, percentages, rule results, and failures
    """
    # Check if it's YAML or XES
    if xes_path.suffix in [".yaml", ".yml"]:
        codec = YAMLCodec()
        matrix = codec.load(xes_path)
    else:
        matrix = discover(xes_path)
    pct = CalculatedPercentages.from_matrix(matrix)

    # Test all rules
    rules = {
        "S1": (check_rule_s1, "structured"),
        "S2": (check_rule_s2, "structured"),
        "S3": (check_rule_s3, "structured"),
        "SS1": (check_rule_ss1, "semi_structured"),
        "SS2": (check_rule_ss2, "semi_structured"),
        "SS3": (check_rule_ss3, "semi_structured"),
        "LS1": (check_rule_ls1, "loosely_structured"),
        "LS2": (check_rule_ls2, "loosely_structured"),
        "U1": (check_rule_u1, "unstructured"),
        "U2": (check_rule_u2, "unstructured"),
        "BS1": (check_rule_bs1, "boundary_structured_semi"),
        "BS2": (check_rule_bs2, "boundary_structured_semi"),
        "BL1": (check_rule_bl1, "boundary_semi_loosely"),
    }

    results = {}
    for rule_name, (rule_fn, category) in rules.items():
        passed, conditions = rule_fn(pct)
        results[rule_name] = {
            "passed": passed,
            "conditions": conditions,
            "category": category,
            "should_match": category.lower() == ground_truth.lower()
            or (category.startswith("boundary") and ground_truth.lower() in category.lower()),
        }

    # Identify failures: should pass but didn't
    failures = []
    for rule_name, result in results.items():
        if result["should_match"] and not result["passed"]:
            failed_condition_indices = [i for i, c in enumerate(result["conditions"]) if not c]
            failures.append(
                {
                    "rule": rule_name,
                    "failed_conditions": failed_condition_indices,
                    "all_conditions": result["conditions"],
                }
            )

    return {
        "path": xes_path,
        "name": xes_path.stem,
        "ground_truth": ground_truth,
        "percentages": pct,
        "results": results,
        "failures": failures,
    }


def main() -> None:
    """Analyze all test logs and output threshold recommendations."""
    test_data = Path("Test Data/Classification")

    # Load all test logs grouped by category
    categories = {
        "structured": test_data / "structured",
        "semi_structured": test_data / "semi-structured",
        "loosely_structured": test_data / "loosely-structured",
        "unstructured": test_data / "unstructured",
    }

    all_analyses = []
    category_percentages: dict[str, list[CalculatedPercentages]] = defaultdict(list)

    for category, category_dir in categories.items():
        if not category_dir.exists():
            continue

        for xes_path in sorted(category_dir.glob("*.xes")):
            analysis = analyze_log(xes_path, category)
            all_analyses.append(analysis)
            category_percentages[category].append(analysis["percentages"])

    # Report failures by category
    print("=== CLASSIFICATION FAILURES BY CATEGORY ===\n")

    for category in ["structured", "semi_structured", "loosely_structured", "unstructured"]:
        category_analyses = [a for a in all_analyses if a["ground_truth"] == category]
        failures_count = sum(1 for a in category_analyses if len(a["failures"]) > 0)

        print(f"{category.upper()}: {failures_count}/{len(category_analyses)} logs have failures")

        for analysis in category_analyses:
            if analysis["failures"]:
                print(f"  {analysis['name']}:")
                for failure in analysis["failures"]:
                    failed_conds = failure["failed_conditions"]
                    print(f"    {failure['rule']} failed conditions: {failed_conds}")
                    print(f"      Percentages: {analysis['percentages'].__dict__}")

    # Analyze percentage distributions
    print("\n=== PERCENTAGE DISTRIBUTIONS BY CATEGORY ===\n")

    for category, pcts_list in category_percentages.items():
        if not pcts_list:
            continue

        print(f"{category.upper()} (n={len(pcts_list)}):")

        # Calculate ranges for each percentage field
        fields = [
            "none_none",
            "none_implication",
            "none_equivalence",
            "eventual_equivalence",
            "eventual_implication",
            "none_negated_equivalence",
            "eventual_any_existential",
            "direct_any_existential",
            "direct_none",
        ]

        for field in fields:
            values = [getattr(p, field) for p in pcts_list]
            min_val = min(values)
            max_val = max(values)
            median_val = sorted(values)[len(values) // 2]

            stats = f"min={min_val:6.3f}, median={median_val:6.3f}, max={max_val:6.3f}"
            print(f"  {field:30s}: {stats}")

        print()

    # Generate threshold adjustment recommendations
    print("=== THRESHOLD ADJUSTMENT RECOMMENDATIONS ===\n")

    # Analyze structured failures
    structured_analyses = [a for a in all_analyses if a["ground_truth"] == "structured"]
    structured_failures = [a for a in structured_analyses if len(a["failures"]) > 0]

    if structured_failures:
        print(f"STRUCTURED: {len(structured_failures)}/{len(structured_analyses)} logs fail\n")

        # Check S1/S2 failure patterns
        s1_failures = [a for a in structured_failures if any(f["rule"] == "S1" for f in a["failures"])]
        s2_failures = [a for a in structured_failures if any(f["rule"] == "S2" for f in a["failures"])]

        if s1_failures:
            print(f"  S1 failures: {len(s1_failures)} logs")
            none_none_vals = [a["percentages"].none_none for a in s1_failures]
            none_impl_vals = [a["percentages"].none_implication for a in s1_failures]
            eventual_equiv_vals = [a["percentages"].eventual_equivalence for a in s1_failures]
            eventual_impl_vals = [a["percentages"].eventual_implication for a in s1_failures]

            print(f"    none_none range: {min(none_none_vals):.3f}-{max(none_none_vals):.3f}")
            print("    Current threshold: < 0.05")
            if max(none_none_vals) >= 0.05:
                print(f"    RECOMMENDATION: Relax to < {max(none_none_vals) + 0.01:.2f}")

            none_impl_range = f"{min(none_impl_vals):.3f}-{max(none_impl_vals):.3f}"
            print(f"    none_implication range: {none_impl_range}")
            print("    Current threshold: < 0.10")
            if max(none_impl_vals) >= 0.10:
                print(f"    RECOMMENDATION: Relax to < {max(none_impl_vals) + 0.01:.2f}")

            eventual_equiv_range = f"{min(eventual_equiv_vals):.3f}-{max(eventual_equiv_vals):.3f}"
            print(f"    eventual_equivalence range: {eventual_equiv_range}")
            print("    Current threshold: > 0.10")
            if min(eventual_equiv_vals) <= 0.10:
                print(f"    RECOMMENDATION: Lower to > {min(eventual_equiv_vals) - 0.01:.2f}")

            eventual_impl_range = f"{min(eventual_impl_vals):.3f}-{max(eventual_impl_vals):.3f}"
            print(f"    eventual_implication range: {eventual_impl_range}")
            print("    Current threshold: > 0.40")
            if min(eventual_impl_vals) <= 0.40:
                print(f"    RECOMMENDATION: Lower to > {min(eventual_impl_vals) - 0.01:.2f}")

            print()

        if s2_failures:
            print(f"  S2 failures: {len(s2_failures)} logs")
            # Similar analysis for S2
            print()

    # Summary
    total_logs = len(all_analyses)
    failed_logs = sum(1 for a in all_analyses if len(a["failures"]) > 0)
    success_rate = (total_logs - failed_logs) / total_logs * 100

    print("\n=== SUMMARY ===")
    print(f"Total logs: {total_logs}")
    print(f"Failed logs: {failed_logs}")
    print(f"Success rate: {success_rate:.1f}%")
    print("Target: 100%")


if __name__ == "__main__":
    main()
