"""Tests for benchmark runner."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from itertools import product

from armature.drift.benchmark import BenchmarkRunner, run_arm_benchmark
from armature.drift.csv_writer import BenchmarkResult


@pytest.fixture
def mock_cdrift_path(tmp_path):
    """Create mock cdrift directory structure."""
    cdrift_path = tmp_path / "cdrift-evaluation"
    eval_logs = cdrift_path / "EvaluationLogs"
    eval_logs.mkdir(parents=True)
    
    # Create mock source directories
    (eval_logs / "Bose").mkdir()
    (eval_logs / "Ceravolo").mkdir()
    
    return cdrift_path


def test_benchmark_runner_init(mock_cdrift_path, tmp_path):
    """BenchmarkRunner initializes with correct defaults."""
    output_path = tmp_path / "results.csv"
    
    runner = BenchmarkRunner(
        cdrift_path=mock_cdrift_path,
        output_path=output_path,
    )
    
    assert runner.cdrift_path == mock_cdrift_path
    assert runner.output_path == output_path
    assert runner.window_sizes == [50, 100, 200, 500]
    assert runner.thresholds == [0.05, 0.1, 0.15, 0.2]
    assert runner.step_sizes == [1]
    assert runner.lag_window == 200


def test_benchmark_runner_custom_params(mock_cdrift_path, tmp_path):
    """BenchmarkRunner accepts custom parameters."""
    output_path = tmp_path / "results.csv"
    
    runner = BenchmarkRunner(
        cdrift_path=mock_cdrift_path,
        output_path=output_path,
        window_sizes=[100, 200],
        thresholds=[0.1, 0.2],
        step_sizes=[1, 5],
        lag_window=300,
    )
    
    assert runner.window_sizes == [100, 200]
    assert runner.thresholds == [0.1, 0.2]
    assert runner.step_sizes == [1, 5]
    assert runner.lag_window == 300


def test_parameter_combinations():
    """Parameter combinations generated correctly."""
    window_sizes = [50, 100]
    thresholds = [0.1, 0.2]
    step_sizes = [1]
    
    # Generate using itertools.product (same as BenchmarkRunner)
    combinations = list(product(window_sizes, thresholds, step_sizes))
    
    # Should have 2 * 2 * 1 = 4 combinations
    assert len(combinations) == 4
    assert (50, 0.1, 1) in combinations
    assert (50, 0.2, 1) in combinations
    assert (100, 0.1, 1) in combinations
    assert (100, 0.2, 1) in combinations


@patch("armature.drift.benchmark.parse_xes")
@patch("armature.drift.benchmark.ARMDriftDetector")
def test_run_single_log_mock(mock_detector_class, mock_parse_xes, mock_cdrift_path, tmp_path):
    """run_single_log returns BenchmarkResult with correct structure."""
    output_path = tmp_path / "results.csv"
    
    # Mock XES parsing
    mock_event_log = Mock()
    mock_event_log.traces = [["A", "B", "C"]] * 100
    mock_parse_xes.return_value = mock_event_log
    
    # Mock detector
    mock_detector = Mock()
    mock_result = Mock()
    mock_result.drift_indices = [50, 100]
    mock_detector.detect.return_value = mock_result
    mock_detector_class.return_value = mock_detector
    
    # Mock dataset.get_log_info
    runner = BenchmarkRunner(mock_cdrift_path, output_path)
    runner.dataset.get_log_info = Mock(return_value={
        "log_source": "Bose",
        "log_name": "test_log",
        "ground_truth": [55, 105],
    })
    
    # Run on mock log
    log_path = Path("/fake/log.xes")
    result = runner.run_single_log(log_path, window_size=100, threshold=0.1, step_size=1)
    
    # Verify BenchmarkResult structure
    assert isinstance(result, BenchmarkResult)
    assert result.algorithm == "ARM"
    assert result.log_source == "Bose"
    assert result.log_name == "test_log"
    assert result.detected_changepoints == [50, 100]
    assert result.actual_changepoints == [55, 105]
    assert result.num_traces == 100
    assert result.window_size == 100
    assert result.threshold == 0.1
    assert result.step_size == 1
    assert result.duration_seconds > 0  # Should have timing
    assert 0 <= result.f1_score <= 1  # F1 should be in valid range
    
    # Verify detector was called correctly
    mock_detector_class.assert_called_once_with(window_size=100, threshold=0.1, step_size=1)
    mock_detector.detect.assert_called_once_with(mock_event_log.traces)


@patch("armature.drift.benchmark.parse_xes")
@patch("armature.drift.benchmark.ARMDriftDetector")
@patch("armature.drift.benchmark.append_result")
def test_error_handling_continues(mock_append, mock_detector_class, mock_parse_xes, mock_cdrift_path, tmp_path):
    """Runner continues processing after single-log error."""
    output_path = tmp_path / "results.csv"
    
    # Create runner
    runner = BenchmarkRunner(
        mock_cdrift_path,
        output_path,
        window_sizes=[100],
        thresholds=[0.1],
        step_sizes=[1],
    )
    
    # Mock dataset to return 3 logs
    log1 = Path("/fake/log1.xes")
    log2 = Path("/fake/log2.xes")
    log3 = Path("/fake/log3.xes")
    runner.dataset.list_logs = Mock(return_value=[log1, log2, log3])
    
    # Mock log info
    def mock_get_log_info(path):
        return {
            "log_source": "Bose",
            "log_name": path.stem,
            "ground_truth": [50],
        }
    runner.dataset.get_log_info = mock_get_log_info
    
    # Mock parse_xes to work for log1 and log3, fail for log2
    def mock_parse_side_effect(path):
        if path == log2:
            raise ValueError("Mock parsing error")
        mock_log = Mock()
        mock_log.traces = [["A", "B"]] * 10
        return mock_log
    mock_parse_xes.side_effect = mock_parse_side_effect
    
    # Mock detector
    mock_detector = Mock()
    mock_result = Mock()
    mock_result.drift_indices = [25]
    mock_detector.detect.return_value = mock_result
    mock_detector_class.return_value = mock_detector
    
    # Run benchmark (should continue despite log2 error)
    results = runner.run_all(progress=False)
    
    # Should have 2 results (log1 and log3), log2 skipped
    assert len(results) == 2
    assert results[0].log_name == "log1"
    assert results[1].log_name == "log3"
    
    # Verify append_result called twice (once for each successful log)
    assert mock_append.call_count == 2


@patch("armature.drift.benchmark.parse_xes")
@patch("armature.drift.benchmark.ARMDriftDetector")
@patch("armature.drift.benchmark.append_result")
def test_run_parameter_sweep(mock_append, mock_detector_class, mock_parse_xes, mock_cdrift_path, tmp_path):
    """run_parameter_sweep limits to first 5 logs by default."""
    output_path = tmp_path / "results.csv"
    
    runner = BenchmarkRunner(
        mock_cdrift_path,
        output_path,
        window_sizes=[100],
        thresholds=[0.1],
        step_sizes=[1],
    )
    
    # Mock dataset to return 10 logs
    logs = [Path(f"/fake/log{i}.xes") for i in range(10)]
    runner.dataset.list_logs = Mock(return_value=logs)
    
    # Mock everything to succeed
    mock_event_log = Mock()
    mock_event_log.traces = [["A", "B"]] * 10
    mock_parse_xes.return_value = mock_event_log
    
    mock_detector = Mock()
    mock_result = Mock()
    mock_result.drift_indices = []
    mock_detector.detect.return_value = mock_result
    mock_detector_class.return_value = mock_detector
    
    def mock_get_log_info(path):
        return {
            "log_source": "Bose",
            "log_name": path.stem,
            "ground_truth": [],
        }
    runner.dataset.get_log_info = mock_get_log_info
    
    # Run parameter sweep (should default to first 5 logs)
    results = runner.run_parameter_sweep(progress=False)
    
    # Should only process 5 logs
    assert len(results) == 5
    
    # Verify only first 5 logs processed
    log_names = [r.log_name for r in results]
    assert log_names == ["log0", "log1", "log2", "log3", "log4"]


def test_run_arm_benchmark_convenience_function(mock_cdrift_path, tmp_path):
    """run_arm_benchmark() convenience function works."""
    output_path = tmp_path / "results.csv"
    
    with patch("armature.drift.benchmark.BenchmarkRunner") as mock_runner_class:
        mock_runner = Mock()
        mock_runner.run_all.return_value = []
        mock_runner_class.return_value = mock_runner
        
        run_arm_benchmark(
            cdrift_path=mock_cdrift_path,
            output_path=output_path,
            window_sizes=[100],
            thresholds=[0.1],
            progress=False,
        )
        
        # Verify BenchmarkRunner created with correct params
        mock_runner_class.assert_called_once_with(
            cdrift_path=mock_cdrift_path,
            output_path=output_path,
            window_sizes=[100],
            thresholds=[0.1],
            step_sizes=None,
            lag_window=200,
        )
        
        # Verify run_all called
        mock_runner.run_all.assert_called_once_with(progress=False)
