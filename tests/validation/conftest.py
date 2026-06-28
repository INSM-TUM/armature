"""Pytest fixtures for validation tests."""

import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory."""
    return project_root / "Test Data" / "Discovery"


@pytest.fixture(scope="session")
def validation_results():
    """Collect results across all validation tests."""
    return {
        "files": [],
        "errors": [],
        "noisy_comparisons": [],
        "start_time": time.time(),
        "end_time": None,
    }


def pytest_generate_tests(metafunc):
    """Parametrize tests with discovered XES files."""
    if "xes_file" not in metafunc.fixturenames:
        return

    # Find project root from conftest location
    conftest_path = Path(__file__).parent
    project_root = conftest_path.parent.parent
    test_data_dir = project_root / "Test Data" / "Discovery"

    # Find all clean test logs (only from root Discovery dir, not noise subdir)
    clean_logs = []
    for f in sorted(test_data_dir.glob("event_log_*.xes")):
        # Exclude noise subdirectory
        if f.parent.name != "noise":
            clean_logs.append(f)

    # Create test IDs from filenames
    ids = [f.stem for f in clean_logs]
    metafunc.parametrize("xes_file", clean_logs, ids=ids)
