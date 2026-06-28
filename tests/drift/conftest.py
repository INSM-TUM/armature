"""Pytest fixtures for drift detection tests."""
import pytest
from pathlib import Path


@pytest.fixture
def drift_logs_dir():
    """Path to counter-example drift logs."""
    return Path(__file__).parent.parent / "fixtures" / "drift_logs"


@pytest.fixture
def reports_dir(tmp_path):
    """Temporary directory for test reports."""
    return tmp_path / "reports"


@pytest.fixture
def arm_detector():
    """Pre-configured ARM drift detector."""
    from armature.drift import ARMDriftDetector
    return ARMDriftDetector(window_size=20, threshold=0.05, step_size=5)


@pytest.fixture
def bose_detector():
    """Pre-configured Bose drift detector."""
    from armature.drift import BoseDriftDetector
    return BoseDriftDetector(window_size=20, measure="j", stat_test="mu")
