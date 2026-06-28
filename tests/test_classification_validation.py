"""Validation suite for classification against Test Data/Classification/.

Validates classification algorithm correctness using ground truth data
organized in category-specific folders (structured, loosely-structured, etc).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from armature.classification import CategoryEnum, classify
from armature.discovery import discover

# Test Data paths
TEST_DATA_ROOT = Path(__file__).parent.parent / "Test Data" / "Classification"
VARIANTS_FILE = TEST_DATA_ROOT / "Variants.txt"


def parse_variants() -> dict[str, dict[str, Any]]:
    """Parse Variants.txt to extract expected classifications.

    Returns:
        Dict mapping log name to expected category info:
        {
            "Log07": {
                "expected_category": CategoryEnum.SEMI_STRUCTURED,
                "is_boundary": False
            },
            "Log14": {
                "expected_category": [
                    CategoryEnum.LOOSELY_STRUCTURED,
                    CategoryEnum.SEMI_STRUCTURED
                ],
                "is_boundary": True
            },
            ...
        }
    """
    if not VARIANTS_FILE.exists():
        pytest.skip(f"Variants.txt not found at {VARIANTS_FILE}")

    variants = {}
    with open(VARIANTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Variants"):
                continue

            # Parse log name: "Log07_semiStructured:" or boundary case
            if line.startswith("Log"):
                # Extract log name and category part
                parts = line.split(":")
                if not parts:
                    continue

                name_part = parts[0].strip()

                # Split on underscore - first part is log name (Log07, Log14, etc)
                # Rest is category or category_category for boundary cases
                name_components = name_part.split("_")
                if len(name_components) < 2:
                    continue

                log_name = name_components[0]  # "Log07"
                # Category part: "semiStructured" or "loosely_semi" (boundary)
                category_part = "_".join(name_components[1:])

                # Check for loop variant first
                is_loop = "_loop" in name_part.lower()
                # Remove loop suffix from category part
                category_part = category_part.replace("_loop", "").strip()

                # Check for boundary case (two categories separated by underscore)
                # Filter out "loop" from category split
                category_components = [c for c in category_part.split("_") if c and c != "loop"]

                if len(category_components) > 1:
                    # Boundary case
                    is_boundary = True
                    expected_category = [_map_category_name(cat) for cat in category_components]
                else:
                    # Single category
                    is_boundary = False
                    expected_category = _map_category_name(category_components[0])

                variants[log_name] = {
                    "expected_category": expected_category,
                    "is_boundary": is_boundary,
                    "is_loop": is_loop,
                }

    return variants


def _map_category_name(name: str) -> CategoryEnum:
    """Map string category name to CategoryEnum."""
    # Remove "loop" suffix if present
    name = name.replace("_loop", "").strip()

    # Case-insensitive mapping
    mapping = {
        "semistructured": CategoryEnum.SEMI_STRUCTURED,
        "looselystructured": CategoryEnum.LOOSELY_STRUCTURED,
        "structured": CategoryEnum.STRUCTURED,
        "unstructured": CategoryEnum.UNSTRUCTURED,
        "structuered": CategoryEnum.STRUCTURED,  # Typo in Log27
    }

    name_lower = name.lower()
    if name_lower not in mapping:
        raise ValueError(f"Unknown category name: {name}")

    return mapping[name_lower]


def find_xes_file(log_name: str) -> Path:
    """Find XES file for given log name in category subdirectories.

    Args:
        log_name: Log name like "Log07"

    Returns:
        Path to XES file

    Raises:
        FileNotFoundError: If XES file not found
    """
    # Search all category folders
    category_folders = [
        "structured",
        "loosely-structured",
        "semi-structured",
        "unstructured",
        "edge-cases",
    ]

    for folder in category_folders:
        folder_path = TEST_DATA_ROOT / folder
        if not folder_path.exists():
            continue

        # Look for files matching log name
        for xes_path in folder_path.glob("**/*.xes"):
            # Check if filename starts with log name
            if xes_path.stem.startswith(log_name):
                return xes_path

    raise FileNotFoundError(f"XES file not found for {log_name}")


def find_yaml_file(log_name: str) -> Path | None:
    """Find YAML matrix file for given log name.

    Args:
        log_name: Log name like "Log07"

    Returns:
        Path to YAML file, or None if not found
    """
    # Search all category folders
    category_folders = [
        "structured",
        "loosely-structured",
        "semi-structured",
        "unstructured",
        "edge-cases",
    ]

    for folder in category_folders:
        folder_path = TEST_DATA_ROOT / folder
        if not folder_path.exists():
            continue

        # Look for YAML files matching log name
        for yaml_path in folder_path.glob("**/*.yaml"):
            if yaml_path.stem.startswith(log_name):
                return yaml_path
        for yml_path in folder_path.glob("**/*.yml"):
            if yml_path.stem.startswith(log_name):
                return yml_path

    return None


@pytest.fixture
def variants():
    """Pytest fixture providing parsed variants data."""
    return parse_variants()


def test_parse_variants():
    """Test that parse_variants() extracts expected categories correctly."""
    variants = parse_variants()

    # Should have parsed multiple logs
    assert len(variants) > 0

    # Check specific logs
    assert "Log07" in variants
    assert variants["Log07"]["expected_category"] == CategoryEnum.SEMI_STRUCTURED
    assert variants["Log07"]["is_boundary"] is False

    assert "Log08" in variants
    assert variants["Log08"]["expected_category"] == CategoryEnum.LOOSELY_STRUCTURED

    # Boundary case
    assert "Log14" in variants
    assert variants["Log14"]["is_boundary"] is True
    assert CategoryEnum.LOOSELY_STRUCTURED in variants["Log14"]["expected_category"]
    assert CategoryEnum.SEMI_STRUCTURED in variants["Log14"]["expected_category"]

    # Loop variants
    assert "Log26" in variants
    assert variants["Log26"]["is_loop"] is True


def find_all_xes_files() -> list[tuple[str, Path, CategoryEnum]]:
    """Find all XES files in Test Data/Classification/ with expected categories.

    Returns:
        List of tuples: (log_name, xes_path, expected_category)
    """
    xes_files = []

    folder_to_category = {
        "structured": CategoryEnum.STRUCTURED,
        "semi-structured": CategoryEnum.SEMI_STRUCTURED,
        "loosely-structured": CategoryEnum.LOOSELY_STRUCTURED,
        "unstructured": CategoryEnum.UNSTRUCTURED,
        "edge-cases": CategoryEnum.LOOSELY_STRUCTURED,  # edge-cases treated as loosely
    }

    for category_folder, expected_category in folder_to_category.items():
        folder_path = TEST_DATA_ROOT / category_folder
        if not folder_path.exists():
            continue

        for xes_path in folder_path.glob("**/*.xes"):
            # Extract log name from filename (e.g., Log07_semiStructured.xes -> Log07)
            log_name = xes_path.stem.split("_")[0]
            xes_files.append((log_name, xes_path, expected_category))

    return sorted(xes_files, key=lambda x: x[0])  # Sort by log name


@pytest.mark.parametrize("log_name,xes_path,expected_category", find_all_xes_files())
def test_full_pipeline_classification(log_name, xes_path, expected_category):
    """Test XES → discovery → classification pipeline completes successfully."""
    # xes_path and expected_category provided directly by parametrize

    # Discover matrix
    matrix = discover(xes_path)
    assert len(matrix.activities) > 0, f"{log_name}: matrix has no activities"

    # Classify
    result = classify(matrix)

    # Validate result structure
    assert result.category in [
        CategoryEnum.STRUCTURED,
        CategoryEnum.SEMI_STRUCTURED,
        CategoryEnum.LOOSELY_STRUCTURED,
        CategoryEnum.UNSTRUCTURED,
    ]
    assert result.confidence in ["exact", "boundary"]
    assert "direct_ratio" in result.dependency_ratios
    assert "eventual_ratio" in result.dependency_ratios
    assert len(result.rule_trace) > 0

    # Validate category match against ground truth
    if result.category != expected_category:
        ratios_items = [f"  {k}: {v:.3f}" for k, v in result.dependency_ratios.items()]
        ratios_str = "\n".join(ratios_items)
        pytest.fail(
            f"Misclassified {xes_path.name}:\n"
            f"  Expected: {expected_category.value}\n"
            f"  Got: {result.category.value}\n"
            f"Ratios:\n{ratios_str}"
        )


@pytest.mark.parametrize("log_name,expected", list(parse_variants().items())[:5])
def test_isolated_classification(log_name, expected):
    """Test YAML → classification pipeline (skips discovery).

    Tests classification on pre-discovered matrices. Since no YAML files exist
    in Test Data, discovers first then re-classifies to validate isolation.
    """
    # Find XES file and discover matrix
    try:
        xes_path = find_xes_file(log_name)
    except FileNotFoundError:
        pytest.skip(f"XES file not found for {log_name}")

    # Discover once
    matrix = discover(xes_path)

    # Classify from matrix (isolated from discovery)
    result1 = classify(matrix)

    # Classify again - should be deterministic
    result2 = classify(matrix)

    # Validate determinism
    assert result1.category == result2.category
    assert result1.dependency_ratios == result2.dependency_ratios
    assert result1.total_dependencies == result2.total_dependencies


def test_classification_coverage():
    """Verify classification pipeline processes all available logs."""
    variants = parse_variants()

    # Count logs that have XES files
    available_count = 0
    processed_count = 0
    errors = []

    for log_name in variants.keys():
        try:
            xes_path = find_xes_file(log_name)
            available_count += 1

            # Attempt to process
            matrix = discover(xes_path)
            _result = classify(matrix)
            processed_count += 1

        except FileNotFoundError:
            continue  # Expected for missing logs
        except Exception as e:
            errors.append(f"{log_name}: {type(e).__name__}: {e}")

    print(f"\nClassification coverage: {processed_count}/{available_count} logs processed")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    assert len(errors) == 0, f"{len(errors)} logs failed to process"
    assert processed_count >= 15, f"Expected at least 15 logs, processed {processed_count}"


def compute_ratio_statistics() -> dict[str, dict[str, dict[str, float]]]:
    """Compute statistics for each ratio by category.

    Returns:
        Dict mapping category -> ratio_name -> {min, max, mean, median, stdev}
        Example: {
            "structured": {
                "direct_ratio": {"min": 0.071, "max": 0.100, "mean": 0.085, ...},
                ...
            },
            ...
        }
    """
    # Organize logs by category from filename
    category_logs: dict[str, list[tuple[str, Path]]] = {
        "structured": [],
        "semi_structured": [],
        "loosely_structured": [],
        "unstructured": [],
    }

    # Discover all XES files
    for category_folder in ["structured", "semi-structured", "loosely-structured", "unstructured"]:
        folder_path = TEST_DATA_ROOT / category_folder
        if not folder_path.exists():
            continue

        # Map folder name to category key
        category_key = category_folder.replace("-", "_")

        for xes_path in folder_path.glob("**/*.xes"):
            log_name = xes_path.stem
            category_logs[category_key].append((log_name, xes_path))

    # Collect ratios per category
    category_ratios: dict[str, dict[str, list[float]]] = {}

    for category, logs in category_logs.items():
        if not logs:
            continue

        # Initialize ratio lists
        ratios_data = {
            "direct_ratio": [],
            "eventual_ratio": [],
            "implication_ratio": [],
            "nand_or_ratio": [],
        }

        for log_name, xes_path in logs:
            try:
                matrix = discover(xes_path)
                result = classify(matrix)

                # Collect ratios
                for ratio_name in ratios_data.keys():
                    ratios_data[ratio_name].append(result.dependency_ratios[ratio_name])

            except Exception as e:
                print(f"Warning: Failed to process {log_name}: {e}")
                continue

        category_ratios[category] = ratios_data

    # Compute statistics
    stats: dict[str, dict[str, dict[str, float]]] = {}

    for category, ratios_data in category_ratios.items():
        stats[category] = {}

        for ratio_name, values in ratios_data.items():
            if not values:
                continue

            stats[category][ratio_name] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }

    return stats


def analyze_failure(log_name: str, xes_path: Path, expected_category: str) -> dict[str, Any]:
    """Analyze a single log for classification failure.

    Args:
        log_name: Name of log file
        xes_path: Path to XES file
        expected_category: Expected category from folder structure

    Returns:
        Dict with gap analysis: {
            "log_name": str,
            "expected": str,
            "actual": str,
            "ratios": dict,
            "gaps": list of {threshold_name, gap, message},
            "recommendation": str
        }
    """
    from armature.classification.config import ConfigLoader

    try:
        matrix = discover(xes_path)
        result = classify(matrix)
    except Exception as e:
        return {
            "log_name": log_name,
            "error": str(e),
            "expected": expected_category,
            "actual": None,
        }

    # Load current thresholds
    config = ConfigLoader.load()

    # Compute gaps (positive = passed, negative = failed)
    gaps = []

    # STRUCTURED thresholds (min thresholds - need to be >= )
    gaps.append(
        {
            "threshold": "direct_ratio_structured",
            "gap": result.dependency_ratios["direct_ratio"] - config.direct_ratio_structured,
            "value": result.dependency_ratios["direct_ratio"],
            "threshold_value": config.direct_ratio_structured,
            "direction": "min",
        }
    )
    gaps.append(
        {
            "threshold": "eventual_ratio_structured",
            "gap": result.dependency_ratios["eventual_ratio"] - config.eventual_ratio_structured,
            "value": result.dependency_ratios["eventual_ratio"],
            "threshold_value": config.eventual_ratio_structured,
            "direction": "min",
        }
    )

    # SEMI-STRUCTURED thresholds
    # direct_ratio_semi_max is a MAX threshold (need to be <=)
    gaps.append(
        {
            "threshold": "direct_ratio_semi_max",
            "gap": config.direct_ratio_semi_max - result.dependency_ratios["direct_ratio"],
            "value": result.dependency_ratios["direct_ratio"],
            "threshold_value": config.direct_ratio_semi_max,
            "direction": "max",
        }
    )
    gaps.append(
        {
            "threshold": "eventual_ratio_semi_min",
            "gap": result.dependency_ratios["eventual_ratio"] - config.eventual_ratio_semi_min,
            "value": result.dependency_ratios["eventual_ratio"],
            "threshold_value": config.eventual_ratio_semi_min,
            "direction": "min",
        }
    )
    gaps.append(
        {
            "threshold": "implication_ratio_semi",
            "gap": result.dependency_ratios["implication_ratio"] - config.implication_ratio_semi,
            "value": result.dependency_ratios["implication_ratio"],
            "threshold_value": config.implication_ratio_semi,
            "direction": "min",
        }
    )

    # LOOSELY-STRUCTURED thresholds
    # direct_ratio_loosely_max is a MAX threshold
    gaps.append(
        {
            "threshold": "direct_ratio_loosely_max",
            "gap": config.direct_ratio_loosely_max - result.dependency_ratios["direct_ratio"],
            "value": result.dependency_ratios["direct_ratio"],
            "threshold_value": config.direct_ratio_loosely_max,
            "direction": "max",
        }
    )
    gaps.append(
        {
            "threshold": "nand_or_ratio_loosely",
            "gap": result.dependency_ratios["nand_or_ratio"] - config.nand_or_ratio_loosely,
            "value": result.dependency_ratios["nand_or_ratio"],
            "threshold_value": config.nand_or_ratio_loosely,
            "direction": "min",
        }
    )

    # Identify failures (negative gaps)
    failures = [g for g in gaps if g["gap"] < 0]

    # Generate recommendation
    recommendation = ""
    if failures:
        # Find largest failure (by absolute gap)
        largest_failure = max(failures, key=lambda g: abs(g["gap"]))
        threshold_name = largest_failure["threshold"]
        current_val = largest_failure["threshold_value"]
        computed_val = largest_failure["value"]
        gap_val = largest_failure["gap"]
        recommendation = (
            f"Lower {threshold_name} to ~{computed_val:.3f} " f"(current: {current_val:.3f}, gap: {gap_val:.3f})"
        )

    return {
        "log_name": log_name,
        "expected": expected_category,
        "actual": result.category.value,
        "ratios": result.dependency_ratios,
        "gaps": gaps,
        "recommendation": recommendation,
        "match": expected_category == result.category.value,
    }


def generate_dataset_report():
    """Generate comprehensive dataset analysis reports.

    Creates:
    - .planning/phases/06.1-classification-validation/analysis/dataset_report.txt
    - .planning/phases/06.1-classification-validation/analysis/failure_analysis.json
    """
    # Create analysis directory
    base_path = Path(__file__).parent.parent / ".planning" / "phases"
    analysis_dir = base_path / "06.1-classification-validation" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    print("Computing ratio statistics by category...")
    stats = compute_ratio_statistics()

    # Analyze all logs for failures
    print("Analyzing all logs for misclassifications...")
    failures = []
    total_logs = 0
    matched_logs = 0

    for category_folder in ["structured", "semi-structured", "loosely-structured", "unstructured"]:
        folder_path = TEST_DATA_ROOT / category_folder
        if not folder_path.exists():
            continue

        # Map folder name to expected category
        expected_category = category_folder.replace("-", "_")

        for xes_path in folder_path.glob("**/*.xes"):
            log_name = xes_path.stem
            total_logs += 1

            analysis = analyze_failure(log_name, xes_path, expected_category)

            if analysis.get("match"):
                matched_logs += 1
            else:
                failures.append(analysis)

    # Write dataset_report.txt
    report_path = analysis_dir / "dataset_report.txt"
    with open(report_path, "w") as f:
        f.write("Classification Dataset Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total logs: {total_logs}\n")
        f.write(f"Correctly classified: {matched_logs}\n")
        f.write(f"Misclassified: {len(failures)}\n")
        f.write(f"Accuracy: {matched_logs / total_logs * 100:.1f}%\n\n")

        f.write("Statistics by Category\n")
        f.write("-" * 60 + "\n\n")

        for category in ["structured", "semi_structured", "loosely_structured", "unstructured"]:
            if category not in stats:
                continue

            category_stats = stats[category]
            log_count = category_stats.get("direct_ratio", {}).get("count", 0)

            f.write(f"{category.upper().replace('_', ' ')} ({log_count} logs):\n")

            ratio_names = ["direct_ratio", "eventual_ratio", "implication_ratio", "nand_or_ratio"]
            for ratio_name in ratio_names:
                if ratio_name not in category_stats:
                    continue

                ratio_stats = category_stats[ratio_name]
                f.write(f"  {ratio_name:20s}: ")
                f.write(f"min={ratio_stats['min']:.3f}, ")
                f.write(f"max={ratio_stats['max']:.3f}, ")
                f.write(f"mean={ratio_stats['mean']:.3f}, ")
                f.write(f"median={ratio_stats['median']:.3f}, ")
                f.write(f"stdev={ratio_stats['stdev']:.3f}\n")

            f.write("\n")

        if failures:
            f.write("Misclassifications\n")
            f.write("-" * 60 + "\n\n")

            for failure in failures:
                expected = failure["expected"]
                actual = failure["actual"]
                f.write(f"{failure['log_name']}: Expected {expected}, Got {actual}\n")
                f.write("  Ratios:\n")
                for ratio_name, value in failure["ratios"].items():
                    f.write(f"    {ratio_name}: {value:.3f}\n")

                f.write("  Failed thresholds:\n")
                for gap in failure["gaps"]:
                    if gap["gap"] < 0:
                        direction_str = ">=" if gap["direction"] == "min" else "<="
                        f.write(f"    - {gap['threshold']}: {gap['gap']:.3f} ")
                        f.write(f"(needs {direction_str} {gap['threshold_value']:.3f}, ")
                        f.write(f"computed {gap['value']:.3f})\n")

                if failure["recommendation"]:
                    f.write(f"  Recommendation: {failure['recommendation']}\n")

                f.write("\n")

    # Write failure_analysis.json
    json_path = analysis_dir / "failure_analysis.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "total_logs": total_logs,
                "matched": matched_logs,
                "misclassified": len(failures),
                "accuracy": matched_logs / total_logs if total_logs > 0 else 0.0,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print("\nReports generated:")
    print(f"  - {report_path}")
    print(f"  - {json_path}")
    accuracy_pct = matched_logs / total_logs * 100
    print(f"\nSummary: {matched_logs}/{total_logs} logs correctly classified ({accuracy_pct:.1f}%)")


