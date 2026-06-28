"""Discovery validation test suite.

Processes all synthetic test logs and generates validation report.
Run with: python3.12 -m pytest tests/validation/ -v
"""

import time
from pathlib import Path

import pytest

from armature.discovery.discover import discover
from armature.serialization.yaml_codec import YAMLCodec

from .report_generator import compare_matrices, generate_report

# ============================================================================
# Clean Log Tests
# ============================================================================


def test_clean_log_discovery(xes_file: Path, validation_results: dict, project_root: Path):
    """Process clean log and store results for report."""
    results_dir = project_root / "tests" / "fixtures" / "discovery" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    golden_dir = project_root / "tests" / "fixtures" / "discovery" / "golden"

    try:
        # Discover matrix
        matrix = discover(xes_file)

        # Save result YAML
        result_path = results_dir / f"{xes_file.stem}.yaml"
        YAMLCodec.save(matrix, result_path)

        # Count non-default relationships
        relationship_count = 0
        for source in matrix.activities:
            for target in matrix.activities:
                cell = matrix.get_cell(source, target)
                # Count if not default (NO_ORDERING + INDEPENDENCE)
                if cell.temporal.value != "no_ordering" or cell.existential.value != "independence":
                    relationship_count += 1

        # Count traces (need to re-parse for this info)
        from armature.discovery.xes_parser import parse_xes

        traces = parse_xes(xes_file)
        trace_count = len(traces)

        # Check for golden output
        golden_path = golden_dir / f"{xes_file.stem}.yaml"
        has_golden = golden_path.exists()

        # Store result
        validation_results["files"].append(
            {
                "name": xes_file.name,
                "activities": len(matrix.activities),
                "relationships": relationship_count,
                "traces": trace_count,
                "has_golden": has_golden,
                "result_path": str(result_path),
                "matrix": matrix,  # Keep for report rendering
            }
        )

        # If golden exists, validate match
        if has_golden:
            golden = YAMLCodec.load(golden_path)
            msg = f"Activities mismatch for {xes_file.name}"
            assert matrix.activities == golden.activities, msg
            # Compare dependencies (cells)
            for source in matrix.activities:
                for target in matrix.activities:
                    msg = f"Cell mismatch at [{source}, {target}] for {xes_file.name}"
                    assert matrix.get_cell(source, target) == golden.get_cell(source, target), msg

    except Exception as e:
        validation_results["errors"].append(
            {
                "file": xes_file.name,
                "error": str(e),
            }
        )
        pytest.fail(f"Discovery failed for {xes_file.name}: {e}")


# ============================================================================
# Noisy Log Comparison Tests
# ============================================================================


class TestNoisyComparison:
    """Compare noisy logs against clean baselines."""

    @pytest.fixture
    def noisy_logs(self, test_data_dir: Path) -> list[Path]:
        """Get all noisy log files."""
        noise_dir = test_data_dir / "noise"
        return sorted(noise_dir.glob("event_log_noise_*.xes"))

    @pytest.fixture
    def clean_logs(self, test_data_dir: Path) -> dict[str, Path]:
        """Map log number to clean log path."""
        clean_files = {}
        for f in test_data_dir.glob("event_log_*.xes"):
            # Skip noise subdirectory
            if f.parent.name == "noise":
                continue
            # Extract number: event_log_01_xxx.xes -> "01"
            parts = f.stem.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                clean_files[parts[2]] = f
        return clean_files

    def test_noisy_log_comparison(
        self,
        noisy_logs: list[Path],
        clean_logs: dict[str, Path],
        validation_results: dict,
        project_root: Path,
    ):
        """Process all noisy logs and compare to clean baselines."""
        results_dir = project_root / "tests" / "fixtures" / "discovery" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for noisy_file in noisy_logs:
            # Extract number: event_log_noise_01.xes -> "01"
            parts = noisy_file.stem.split("_")
            log_num = parts[-1]  # Last part should be number

            clean_file = clean_logs.get(log_num)
            if not clean_file:
                validation_results["errors"].append(
                    {
                        "file": noisy_file.name,
                        "error": f"No matching clean log found for number {log_num}",
                    }
                )
                continue

            try:
                # Discover both matrices
                clean_matrix = discover(clean_file)
                noisy_matrix = discover(noisy_file)

                # Save noisy result
                result_path = results_dir / f"{noisy_file.stem}.yaml"
                YAMLCodec.save(noisy_matrix, result_path)

                # Compare
                changes = compare_matrices(clean_matrix, noisy_matrix)

                # Store comparison
                validation_results["noisy_comparisons"].append(
                    {
                        "clean_name": clean_file.name,
                        "noisy_name": noisy_file.name,
                        "clean_matrix": clean_matrix,
                        "noisy_matrix": noisy_matrix,
                        "changes": changes,
                        "result_path": str(result_path),
                    }
                )

            except Exception as e:
                validation_results["errors"].append(
                    {
                        "file": noisy_file.name,
                        "error": str(e),
                    }
                )


# ============================================================================
# Report Generation (runs after all tests)
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def finalize_report(request, validation_results: dict, project_root: Path):
    """Generate HTML report after all tests complete."""
    yield  # Run all tests first

    # Set end time
    validation_results["end_time"] = time.time()

    # Generate report
    from datetime import datetime

    report_dir = project_root / "tests" / "reports"
    report_path = report_dir / f"validation_{datetime.now():%Y%m%d_%H%M%S}.html"

    generate_report(validation_results, report_path)

    # Print report location
    print(f"\n{'=' * 60}")
    print(f"VALIDATION REPORT: {report_path}")
    print(f"{'=' * 60}")
    print(f"Clean logs: {len(validation_results['files'])}")
    print(f"Errors: {len(validation_results['errors'])}")
    print(f"Noisy comparisons: {len(validation_results['noisy_comparisons'])}")
