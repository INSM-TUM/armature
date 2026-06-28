"""Tests for cdrift-compatible wrapper functions.

Validates that test_arm_hybrid() matches cdrift's standard detector
signature and return format for integration into testAll_reproducibility.py
"""

import inspect
import re
from pathlib import Path

import pytest

from armature.drift.cdrift_wrapper import test_arm_hybrid, calcDurFromSeconds


class TestArmHybridSignature:
    """Validate function signature matches cdrift pattern."""

    def test_signature_matches_cdrift_pattern(self):
        """Verify test_arm_hybrid has correct parameter names and order."""
        sig = inspect.signature(test_arm_hybrid)
        params = list(sig.parameters.keys())
        
        # Check required positional parameters in correct order
        assert params[0] == 'filepath'
        assert params[1] == 'window_size'
        assert params[2] == 'prominence'
        assert params[3] == 'step_size'
        assert params[4] == 'F1_LAG'
        assert params[5] == 'cp_locations'
        assert params[6] == 'position'
        assert params[7] == 'show_progress_bar'
        
        # Check defaults for optional parameters
        assert sig.parameters['position'].default is None
        assert sig.parameters['show_progress_bar'].default is True


class TestArmHybridReturnFormat:
    """Validate return format matches cdrift expectations."""

    @pytest.fixture
    def test_log(self):
        """Path to compressed test log."""
        log_path = Path(__file__).parent.parent / "fixtures" / "compressed.xes.gz"
        assert log_path.exists(), f"Test log not found: {log_path}"
        return str(log_path)

    def test_returns_list_of_dicts(self, test_log):
        """Verify function returns list containing dict."""
        result = test_arm_hybrid(
            filepath=test_log,
            window_size=5,
            prominence=1.0,
            step_size=2,
            F1_LAG=200,
            cp_locations=[],
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_has_all_required_columns(self, test_log):
        """Verify returned dict has all required cdrift columns."""
        result = test_arm_hybrid(
            filepath=test_log,
            window_size=5,
            prominence=1.0,
            step_size=2,
            F1_LAG=200,
            cp_locations=[],
        )
        
        row = result[0]
        
        # Standard columns (all detectors)
        assert 'Algorithm' in row
        assert 'Log' in row
        assert 'Log Source' in row
        assert 'Detected Changepoints' in row
        assert 'Actual Changepoints for Log' in row
        assert 'F1-Score' in row
        assert 'Average Lag' in row
        assert 'Duration' in row
        assert 'Duration (Seconds)' in row
        assert 'Seconds per Case' in row
        
        # Parameter columns (detector-specific)
        assert 'window_size' in row
        assert 'prominence' in row
        assert 'step_size' in row

    def test_column_values_correct_types(self, test_log):
        """Verify column values have correct types."""
        result = test_arm_hybrid(
            filepath=test_log,
            window_size=5,
            prominence=1.0,
            step_size=2,
            F1_LAG=200,
            cp_locations=[],
        )
        
        row = result[0]
        
        assert row['Algorithm'] == 'ARM-Hybrid'
        assert row['Log'] == 'compressed'  # stem without .xes.gz
        assert row['Log Source'] == 'fixtures'  # parent dir name
        
        # Parameters should match input
        assert row['window_size'] == 5
        assert row['prominence'] == 1.0
        assert row['step_size'] == 2
        
        # Metrics should be numeric
        assert isinstance(row['F1-Score'], (int, float))
        assert isinstance(row['Average Lag'], (int, float))
        assert isinstance(row['Duration (Seconds)'], (int, float))
        assert isinstance(row['Seconds per Case'], (int, float))
        
        # Duration should be string in hh:mm:ss format
        assert isinstance(row['Duration'], str)


class TestChangePointFormat:
    """Validate changepoint list format (critical for CSV serialization)."""

    @pytest.fixture
    def test_log(self):
        """Path to compressed test log."""
        log_path = Path(__file__).parent.parent / "fixtures" / "compressed.xes.gz"
        assert log_path.exists(), f"Test log not found: {log_path}"
        return str(log_path)

    def test_changepoints_are_python_list(self, test_log):
        """Verify Detected Changepoints is Python list not numpy array."""
        result = test_arm_hybrid(
            filepath=test_log,
            window_size=5,
            prominence=1.0,
            step_size=2,
            F1_LAG=200,
            cp_locations=[],
        )
        
        detected = result[0]['Detected Changepoints']
        
        # Must be Python list for proper CSV serialization
        assert isinstance(detected, list)
        assert type(detected).__name__ == 'list'  # Not ndarray or other type
        
        # Elements should be Python ints
        if detected:  # If any changepoints detected
            assert all(isinstance(cp, int) for cp in detected)


class TestDurationFormatting:
    """Validate duration formatting matches cdrift convention."""

    @pytest.fixture
    def test_log(self):
        """Path to compressed test log."""
        log_path = Path(__file__).parent.parent / "fixtures" / "compressed.xes.gz"
        assert log_path.exists(), f"Test log not found: {log_path}"
        return str(log_path)

    def test_duration_matches_hhmmss_pattern(self, test_log):
        """Verify Duration field matches HH:MM:SS format."""
        result = test_arm_hybrid(
            filepath=test_log,
            window_size=5,
            prominence=1.0,
            step_size=2,
            F1_LAG=200,
            cp_locations=[],
        )
        
        duration_str = result[0]['Duration']
        
        # Should match HH:MM:SS pattern
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        assert re.match(pattern, duration_str), f"Duration '{duration_str}' doesn't match HH:MM:SS"

    def test_calcdurfromseconds_helper(self):
        """Test helper function with known inputs."""
        # 120 seconds = 2 minutes
        assert calcDurFromSeconds(120) == "00:02:00"
        
        # 3661 seconds = 1 hour, 1 minute, 1 second
        assert calcDurFromSeconds(3661) == "01:01:01"
        
        # 0 seconds
        assert calcDurFromSeconds(0) == "00:00:00"
        
        # Fractional seconds should be floored
        assert calcDurFromSeconds(120.9) == "00:02:00"
