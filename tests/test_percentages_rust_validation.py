"""Validate Python percentage calculation matches Rust implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from armature.classification.percentages import CalculatedPercentages
from armature.discovery import discover
from scripts.extract_rust_percentages import extract_rust_percentages

TEST_LOGS = ["Log06", "Log08", "Log10", "Log12"]
TOLERANCE = 0.001


@pytest.mark.parametrize("log_name", TEST_LOGS)
def test_python_percentages_match_rust(log_name):
    """Verify Python percentages match Rust within 0.001 tolerance.

    Compares Python CalculatedPercentages against Rust implementation output
    for representative logs: Log06 (semi), Log08 (semi), Log10 (loosely), Log12 (structured).

    Expected outcomes:
    - If PASS: Our percentages match Rust, problem is discovery model incompatibility
    - If FAIL: Our percentage calculation has bugs, fix those first
    """
    # Get Rust percentages
    rust_pct = extract_rust_percentages(log_name)
    if rust_pct is None:
        pytest.skip("Rust binary not available")

    # Compute Python percentages
    test_data = Path("/home/choky/kerstin/armature/Test Data/Classification")
    xes_files = list(test_data.rglob(f"{log_name}*.xes"))
    assert xes_files, f"No .xes file found for {log_name}"

    matrix = discover(xes_files[0])
    python_pct = CalculatedPercentages.from_matrix(matrix)

    # Compare all 9 fields
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
        rust_val = rust_pct[field]
        python_val = getattr(python_pct, field)
        diff = abs(rust_val - python_val)
        assert diff < TOLERANCE, (
            f"{log_name}.{field}: Python={python_val:.4f} vs Rust={rust_val:.4f} "
            f"(diff={diff:.4f} > tolerance={TOLERANCE})"
        )


def test_percentage_calculation_deterministic():
    """Percentages should be deterministic for same matrix."""
    test_data = Path("/home/choky/kerstin/armature/Test Data/Classification")
    xes_files = list(test_data.rglob("Log06*.xes"))

    matrix = discover(xes_files[0])
    pct1 = CalculatedPercentages.from_matrix(matrix)
    pct2 = CalculatedPercentages.from_matrix(matrix)

    assert pct1 == pct2, "Percentage calculation not deterministic"
