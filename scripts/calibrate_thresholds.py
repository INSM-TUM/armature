#!/usr/bin/env python3.12
"""Calibrate classification thresholds from ground truth test data.

Discovers matrices from Test Data/Classification/*/*.xes files,
computes dependency ratios, and outputs calibrated threshold values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# isort: off
from armature.classification.engine import ClassificationEngine  # noqa: E402
from armature.discovery.discover import discover  # noqa: E402
# isort: on


def discover_and_classify(xes_path: Path) -> dict:
    """Discover matrix from XES and compute ratios.

    Args:
        xes_path: Path to XES file

    Returns:
        Dict with dependency ratios and counts
    """
    # Discover matrix
    matrix = discover(xes_path)

    # Classify to get ratios
    classifier = ClassificationEngine()
    result = classifier.classify(matrix)

    return {
        "ratios": result.dependency_ratios,
        "counts": result.dependency_counts,
        "category": result.category.value,
        "density": result.density,
    }


def main():
    """Calibrate thresholds from ground truth data."""
    test_data_dir = project_root / "Test Data" / "Classification"

    if not test_data_dir.exists():
        print(f"Error: Test data directory not found: {test_data_dir}", file=sys.stderr)
        sys.exit(1)

    # Category folders to process
    categories = ["structured", "semi-structured", "loosely-structured", "unstructured"]

    # Collect results by category
    results_by_category = {cat: [] for cat in categories}

    print("Discovering matrices and computing ratios...")
    print()

    for category in categories:
        category_dir = test_data_dir / category
        if not category_dir.exists():
            continue

        xes_files = list(category_dir.glob("*.xes"))
        print(f"{category}: {len(xes_files)} files")

        for xes_file in xes_files:
            try:
                result = discover_and_classify(xes_file)
                results_by_category[category].append(
                    {
                        "file": xes_file.name,
                        **result,
                    }
                )
                print(f"  ✓ {xes_file.name}")
            except Exception as e:
                print(f"  ✗ {xes_file.name}: {e}", file=sys.stderr)

    print()
    print("=" * 80)
    print("RATIO ANALYSIS BY CATEGORY")
    print("=" * 80)
    print()

    # Analyze ratios by category
    stats = {}
    for category in categories:
        results = results_by_category[category]
        if not results:
            continue

        print(f"\n{category.upper()} ({len(results)} samples)")
        print("-" * 40)

        # Extract ratio arrays
        direct_ratios = [r["ratios"]["direct_ratio"] for r in results]
        eventual_ratios = [r["ratios"]["eventual_ratio"] for r in results]
        implication_ratios = [r["ratios"]["implication_ratio"] for r in results]
        nand_or_ratios = [r["ratios"]["nand_or_ratio"] for r in results]

        # Compute stats
        def stats_for(values):
            if not values:
                return {"min": 0, "max": 0, "median": 0}
            sorted_vals = sorted(values)
            return {
                "min": min(values),
                "max": max(values),
                "median": sorted_vals[len(sorted_vals) // 2],
            }

        direct_stats = stats_for(direct_ratios)
        eventual_stats = stats_for(eventual_ratios)
        implication_stats = stats_for(implication_ratios)
        nand_or_stats = stats_for(nand_or_ratios)

        stats[category] = {
            "direct": direct_stats,
            "eventual": eventual_stats,
            "implication": implication_stats,
            "nand_or": nand_or_stats,
        }

        print(
            f"  direct_ratio:      min={direct_stats['min']:.3f}, "
            f"max={direct_stats['max']:.3f}, median={direct_stats['median']:.3f}"
        )
        print(
            f"  eventual_ratio:    min={eventual_stats['min']:.3f}, "
            f"max={eventual_stats['max']:.3f}, median={eventual_stats['median']:.3f}"
        )
        print(
            f"  implication_ratio: min={implication_stats['min']:.3f}, "
            f"max={implication_stats['max']:.3f}, median={implication_stats['median']:.3f}"
        )
        print(
            f"  nand_or_ratio:     min={nand_or_stats['min']:.3f}, "
            f"max={nand_or_stats['max']:.3f}, median={nand_or_stats['median']:.3f}"
        )

    print()
    print("=" * 80)
    print("PROPOSED THRESHOLD CALIBRATION")
    print("=" * 80)
    print()

    # Propose thresholds based on data
    if "structured" in stats:
        print("Structured thresholds:")
        print(f"  direct_ratio_structured: {stats['structured']['direct']['median']:.3f}")
        print(f"  eventual_ratio_structured: {stats['structured']['eventual']['median']:.3f}")
        print()

    if "semi-structured" in stats:
        print("Semi-structured thresholds:")
        # Use max direct from semi as the ceiling
        print(f"  direct_ratio_semi_max: {stats['semi-structured']['direct']['max']:.3f}")
        print(f"  eventual_ratio_semi_min: {stats['semi-structured']['eventual']['min']:.3f}")
        print(f"  implication_ratio_semi: {stats['semi-structured']['implication']['median']:.3f}")
        print()

    if "loosely-structured" in stats:
        print("Loosely-structured thresholds:")
        print(f"  direct_ratio_loosely_max: {stats['loosely-structured']['direct']['max']:.3f}")
        print(f"  nand_or_ratio_loosely: {stats['loosely-structured']['nand_or']['median']:.3f}")
        print()

    # Output YAML config
    print()
    print("=" * 80)
    print("YAML CONFIGURATION")
    print("=" * 80)
    print()

    yaml_config = {}
    if "structured" in stats:
        yaml_config["direct_ratio_structured"] = round(stats["structured"]["direct"]["median"], 3)
        yaml_config["eventual_ratio_structured"] = round(
            stats["structured"]["eventual"]["median"], 3
        )

    if "semi-structured" in stats:
        yaml_config["direct_ratio_semi_max"] = round(stats["semi-structured"]["direct"]["max"], 3)
        yaml_config["eventual_ratio_semi_min"] = round(
            stats["semi-structured"]["eventual"]["min"], 3
        )
        yaml_config["implication_ratio_semi"] = round(
            stats["semi-structured"]["implication"]["median"], 3
        )

    if "loosely-structured" in stats:
        yaml_config["direct_ratio_loosely_max"] = round(
            stats["loosely-structured"]["direct"]["max"], 3
        )
        yaml_config["nand_or_ratio_loosely"] = round(
            stats["loosely-structured"]["nand_or"]["median"], 3
        )

    print(yaml.dump(yaml_config, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
