"""Golden baseline regression test suite.

This module provides dedicated regression testing for discovery output validation.
Unlike test_discovery_validation.py which generates comprehensive validation reports,
this test focuses solely on asserting that discovery output matches golden baselines.

Purpose: Prevent future regressions - any discovery algorithm changes that alter
output will trigger test failures with precise diagnostics.

Run with: python3.12 -m pytest tests/validation/test_golden_regression.py -v
"""

from pathlib import Path

import pytest

from armature.discovery.discover import discover
from armature.serialization.yaml_codec import YAMLCodec


def test_discovery_matches_golden_baseline(xes_file: Path, project_root: Path):
    """Validate that discovery output exactly matches golden baseline.

    For each test log:
    1. Run discover() to generate matrix
    2. Load golden YAML from tests/fixtures/discovery/golden/
    3. Assert activities match
    4. Assert all cells match

    Args:
        xes_file: Path to XES test log
        project_root: Project root directory

    Raises:
        pytest.skip: If golden baseline does not exist yet
        AssertionError: If any cell differs from golden baseline
    """
    golden_dir = project_root / "tests" / "fixtures" / "discovery" / "golden"
    golden_path = golden_dir / f"{xes_file.stem}.yaml"

    # Skip if golden baseline not created yet
    if not golden_path.exists():
        pytest.skip(f"Golden baseline not yet created for {xes_file.name}")

    # Load golden baseline
    golden = YAMLCodec.load(golden_path)

    # Run discovery
    matrix = discover(xes_file)

    # Assert activities match
    if matrix.activities != golden.activities:
        pytest.fail(
            f"Regression detected in {xes_file.name}: Activity set mismatch\n"
            f"Expected activities: {sorted(golden.activities)}\n"
            f"Got activities: {sorted(matrix.activities)}"
        )

    # Assert all cells match
    for source in matrix.activities:
        for target in matrix.activities:
            matrix_cell = matrix.get_cell(source, target)
            golden_cell = golden.get_cell(source, target)

            if matrix_cell != golden_cell:
                pytest.fail(
                    f"Regression detected in {xes_file.name} at cell [{source}, {target}]\n"
                    f"Expected: {golden_cell.temporal.value}/{golden_cell.existential.value}\n"
                    f"Got: {matrix_cell.temporal.value}/{matrix_cell.existential.value}"
                )
