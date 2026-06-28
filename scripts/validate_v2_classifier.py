#!/usr/bin/env python3.12
"""Validate v2 classifier against all 34 test logs."""

from pathlib import Path

from armature.classification import classify
from armature.discovery import discover

# Test Data paths
TEST_DATA_ROOT = Path(__file__).parent.parent / "Test Data" / "Classification"


def find_all_xes_files() -> list[tuple[str, Path, str]]:
    """Find all XES files with expected categories from folder structure."""
    xes_files = []

    folder_to_category = {
        "structured": "structured",
        "semi-structured": "semi_structured",
        "loosely-structured": "loosely_structured",
        "unstructured": "unstructured",
        "edge-cases": "loosely_structured",  # edge-cases treated as loosely
    }

    for category_folder, expected_category in folder_to_category.items():
        folder_path = TEST_DATA_ROOT / category_folder
        if not folder_path.exists():
            continue

        for xes_path in folder_path.glob("**/*.xes"):
            # Extract log name from filename
            log_name = xes_path.stem.split("_")[0]
            xes_files.append((log_name, xes_path, expected_category))

    return sorted(xes_files, key=lambda x: x[0])


def main():
    """Run v2 classifier on all test logs."""
    test_logs = find_all_xes_files()

    print(f"Running v2 classifier on {len(test_logs)} logs...")
    print("=" * 80)

    passed = 0
    failed = 0
    errors = []

    for log_name, xes_path, expected_category in test_logs:
        try:
            # Discover matrix
            matrix = discover(xes_path)

            # Classify with v2
            result = classify(matrix, use_v2=True)

            # Check match
            actual_category = result.category.value

            if actual_category == expected_category:
                print(f"✓ {log_name:8s} {expected_category:20s} -> {actual_category:20s}")
                passed += 1
            else:
                print(f"✗ {log_name:8s} {expected_category:20s} -> {actual_category:20s} MISMATCH")
                failed += 1

                # Show percentages and rules for debugging
                pct = result.dependency_ratios
                print(
                    f"    none_none={pct.get('none_none', 0):.3f}, "
                    f"none_impl={pct.get('none_implication', 0):.3f}, "
                    f"eventual_impl={pct.get('eventual_implication', 0):.3f}"
                )
                # Show which rules matched
                matched_rules = [r for r in result.rule_trace if r.get("passed")]
                if matched_rules:
                    rules_str = ", ".join([r["rule"] for r in matched_rules])
                    print(f"    Matched rules: {rules_str}")

        except Exception as e:
            print(f"✗ {log_name:8s} ERROR: {e}")
            errors.append((log_name, str(e)))
            failed += 1

    print("=" * 80)
    print(f"\nResults: {passed}/{len(test_logs)} passed ({passed/len(test_logs)*100:.1f}%)")
    print(f"Failed: {failed}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for log_name, error in errors:
            print(f"  {log_name}: {error}")


if __name__ == "__main__":
    main()
