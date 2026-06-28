"""Tests for drift detection evaluation metrics and CSV writer."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from armature.drift.evaluation import (
    bipartite_match_changepoints,
    compute_average_lag,
    compute_metrics,
)
from armature.drift.csv_writer import (
    BenchmarkResult,
    REQUIRED_COLUMNS,
    write_results_csv,
    append_result,
)


class TestBipartiteMatching:
    """Test bipartite matching algorithm."""

    def test_bipartite_match_exact(self):
        """Exact matches within lag window."""
        detected = [100, 200]
        actual = [100, 200]
        lag = 10

        tp, fp = bipartite_match_changepoints(detected, actual, lag)

        assert tp == 2
        assert fp == 0

    def test_bipartite_match_within_lag(self):
        """Matches within lag tolerance."""
        detected = [105, 195]
        actual = [100, 200]
        lag = 10

        tp, fp = bipartite_match_changepoints(detected, actual, lag)

        assert tp == 2
        assert fp == 0

    def test_bipartite_match_outside_lag(self):
        """One match outside lag window."""
        detected = [100, 250]
        actual = [100, 200]
        lag = 10

        tp, fp = bipartite_match_changepoints(detected, actual, lag)

        assert tp == 1
        assert fp == 1

    def test_bipartite_match_no_double_count(self):
        """Prevents double counting of actual changepoints."""
        detected = [100, 101]
        actual = [100]
        lag = 10

        tp, fp = bipartite_match_changepoints(detected, actual, lag)

        # Only one actual to match, so only one can be TP
        assert tp == 1
        assert fp == 1

    def test_bipartite_match_empty_detected(self):
        """Handle empty detected list."""
        tp, fp = bipartite_match_changepoints([], [100, 200], 10)

        assert tp == 0
        assert fp == 0

    def test_bipartite_match_empty_actual(self):
        """Handle empty actual list."""
        tp, fp = bipartite_match_changepoints([100, 200], [], 10)

        assert tp == 0
        assert fp == 2

    def test_bipartite_match_greedy_closest(self):
        """Greedy matching assigns closest pairs first."""
        detected = [100, 110]
        actual = [100, 105]
        lag = 15

        tp, fp = bipartite_match_changepoints(detected, actual, lag)

        # 100->100 (dist 0), 110->105 (dist 5)
        assert tp == 2
        assert fp == 0


class TestComputeMetrics:
    """Test metric computation."""

    def test_compute_metrics_precision_recall(self):
        """Compute precision and recall correctly."""
        detected = [100, 200, 300]
        actual = [100, 200]
        lag = 10

        metrics = compute_metrics(detected, actual, lag)

        assert metrics["tp"] == 2
        assert metrics["fp"] == 1
        assert metrics["fn"] == 0
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == pytest.approx(0.8, abs=0.01)

    def test_compute_metrics_empty_detected(self):
        """Handle empty detected list."""
        metrics = compute_metrics([], [100, 200], 10)

        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 2
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_compute_metrics_empty_actual(self):
        """Handle empty actual list."""
        metrics = compute_metrics([100, 200], [], 10)

        assert metrics["tp"] == 0
        assert metrics["fp"] == 2
        assert metrics["fn"] == 0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_compute_metrics_perfect(self):
        """Perfect detection."""
        detected = [100, 200]
        actual = [100, 200]
        lag = 10

        metrics = compute_metrics(detected, actual, lag)

        assert metrics["tp"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestAverageLag:
    """Test average lag computation."""

    def test_average_lag_exact_match(self):
        """Zero lag for exact matches."""
        detected = [100, 200]
        actual = [100, 200]

        avg_lag = compute_average_lag(detected, actual, lag=10)

        assert avg_lag == 0.0

    def test_average_lag_with_offset(self):
        """Compute average lag with offsets."""
        detected = [105, 195]
        actual = [100, 200]

        avg_lag = compute_average_lag(detected, actual, lag=10)

        assert avg_lag == 5.0  # (5 + 5) / 2

    def test_average_lag_no_matches(self):
        """Zero lag when no matches."""
        detected = [300]
        actual = [100]

        avg_lag = compute_average_lag(detected, actual, lag=10)

        assert avg_lag == 0.0

    def test_average_lag_empty_lists(self):
        """Handle empty lists."""
        assert compute_average_lag([], [100], lag=10) == 0.0
        assert compute_average_lag([100], [], lag=10) == 0.0


class TestCSVWriter:
    """Test CSV writer functionality."""

    def test_csv_writer_format(self):
        """Verify CSV format and column names."""
        result = BenchmarkResult(
            algorithm="ARM",
            log_source="Bose",
            log_name="test_log",
            detected_changepoints=[100, 200],
            actual_changepoints=[100, 200],
            f1_score=1.0,
            average_lag=0.0,
            duration_seconds=10.5,
            num_traces=300,
            window_size=50,
            threshold=0.1,
            step_size=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            write_results_csv([result], output_path)

            # Read back
            df = pd.read_csv(output_path)

            # Verify column names
            assert list(df.columns) == REQUIRED_COLUMNS

            # Verify values
            assert df["Algorithm"].iloc[0] == "ARM"
            assert df["Log Source"].iloc[0] == "Bose"
            assert df["Log"].iloc[0] == "test_log"
            assert df["Detected Changepoints"].iloc[0] == "[100, 200]"
            assert df["Actual Changepoints for Log"].iloc[0] == "[100, 200]"
            assert df["F1-Score"].iloc[0] == 1.0
            assert df["Average Lag"].iloc[0] == 0.0
            assert df["Duration (Seconds)"].iloc[0] == 10.5
            assert df["Window Size"].iloc[0] == 50
            assert df["Threshold"].iloc[0] == 0.1
            assert df["Step Size"].iloc[0] == 1

        finally:
            output_path.unlink()

    def test_csv_duration_format(self):
        """Verify duration formatting."""
        result = BenchmarkResult(
            algorithm="ARM",
            log_source="Bose",
            log_name="test",
            detected_changepoints=[],
            actual_changepoints=[],
            duration_seconds=3723,  # 1h 2m 3s
            num_traces=100,
            window_size=50,
            threshold=0.1,
            step_size=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            write_results_csv([result], output_path)
            df = pd.read_csv(output_path)

            assert df["Duration"].iloc[0] == "01:02:03"

        finally:
            output_path.unlink()

    def test_csv_append_result(self):
        """Test append_result functionality."""
        result1 = BenchmarkResult(
            algorithm="ARM",
            log_source="Bose",
            log_name="log1",
            detected_changepoints=[100],
            actual_changepoints=[100],
            window_size=50,
            threshold=0.1,
            step_size=1,
        )

        result2 = BenchmarkResult(
            algorithm="ARM",
            log_source="Bose",
            log_name="log2",
            detected_changepoints=[200],
            actual_changepoints=[200],
            window_size=50,
            threshold=0.1,
            step_size=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Remove file so we start fresh
            output_path.unlink()

            # Append first result (creates file)
            append_result(result1, output_path)
            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert df["Log"].iloc[0] == "log1"

            # Append second result
            append_result(result2, output_path)
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert df["Log"].iloc[0] == "log1"
            assert df["Log"].iloc[1] == "log2"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_csv_seconds_per_case(self):
        """Verify seconds per case calculation."""
        result = BenchmarkResult(
            algorithm="ARM",
            log_source="Bose",
            log_name="test",
            detected_changepoints=[],
            actual_changepoints=[],
            duration_seconds=100.0,
            num_traces=50,
            window_size=50,
            threshold=0.1,
            step_size=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            write_results_csv([result], output_path)
            df = pd.read_csv(output_path)

            assert df["Seconds per Case"].iloc[0] == 2.0  # 100 / 50

        finally:
            output_path.unlink()
