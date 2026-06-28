"""Tests for cdrift-evaluation benchmark adapter."""
import pytest
from pathlib import Path
from armature.drift.cdrift_adapter import (
    CdriftDataset,
    extract_ground_truth,
    list_benchmark_logs,
    _parse_changepoint_attr,
)


def test_extract_ground_truth_from_filename(tmp_path):
    """Test ground truth extraction from filename patterns."""
    # Test multiple changepoints
    log_path = tmp_path / "log_cp_100_300.xes"
    log_path.touch()
    result = extract_ground_truth(log_path)
    assert result == [100, 300]
    
    # Test single changepoint
    log_path = tmp_path / "log_cp_500.xes"
    log_path.touch()
    result = extract_ground_truth(log_path)
    assert result == [500]
    
    # Test no changepoints
    log_path = tmp_path / "log.xes"
    log_path.touch()
    result = extract_ground_truth(log_path)
    assert result == []
    
    # Test three changepoints
    log_path = tmp_path / "log_cp_100_200_300.xes"
    log_path.touch()
    result = extract_ground_truth(log_path)
    assert result == [100, 200, 300]


def test_parse_changepoint_attr():
    """Test parsing of changepoint attribute values."""
    # Comma-separated
    assert _parse_changepoint_attr("100,300") == [100, 300]
    
    # Bracketed comma-separated
    assert _parse_changepoint_attr("[100, 300]") == [100, 300]
    
    # Space-separated
    assert _parse_changepoint_attr("100 300") == [100, 300]
    
    # Single value
    assert _parse_changepoint_attr("500") == [500]
    
    # Empty
    assert _parse_changepoint_attr("") == []
    assert _parse_changepoint_attr(None) == []


def test_cdrift_dataset_list_logs(tmp_path):
    """Test log enumeration from cdrift directory structure."""
    # Create mock cdrift directory structure
    eval_logs = tmp_path / "EvaluationLogs"
    bose_dir = eval_logs / "Bose"
    ceravolo_dir = eval_logs / "Ceravolo"
    ostovar_dir = eval_logs / "Ostovar"
    
    bose_dir.mkdir(parents=True)
    ceravolo_dir.mkdir(parents=True)
    ostovar_dir.mkdir(parents=True)
    
    # Create dummy XES files
    (bose_dir / "log1.xes").touch()
    (bose_dir / "log2.xes").touch()
    (ceravolo_dir / "log3.xes").touch()
    (ostovar_dir / "log4.xes").touch()
    
    # Create .gz file (should be excluded)
    (bose_dir / "log5.xes.gz").touch()
    
    # Initialize dataset
    dataset = CdriftDataset(tmp_path)
    
    # Test list_logs - should return all .xes files but not .gz
    logs = dataset.list_logs()
    assert len(logs) == 4
    log_names = [log.name for log in logs]
    assert "log1.xes" in log_names
    assert "log2.xes" in log_names
    assert "log3.xes" in log_names
    assert "log4.xes" in log_names
    assert "log5.xes.gz" not in log_names


def test_cdrift_dataset_list_logs_by_source(tmp_path):
    """Test filtering logs by source."""
    # Create mock cdrift directory structure
    eval_logs = tmp_path / "EvaluationLogs"
    bose_dir = eval_logs / "Bose"
    ceravolo_dir = eval_logs / "Ceravolo"
    
    bose_dir.mkdir(parents=True)
    ceravolo_dir.mkdir(parents=True)
    
    # Create dummy files
    (bose_dir / "log1.xes").touch()
    (bose_dir / "log2.xes").touch()
    (ceravolo_dir / "log3.xes").touch()
    
    dataset = CdriftDataset(tmp_path)
    
    # Test Bose filter
    bose_logs = dataset.list_logs_by_source("Bose")
    assert len(bose_logs) == 2
    assert all("Bose" in str(log.parent) for log in bose_logs)
    
    # Test Ceravolo filter
    ceravolo_logs = dataset.list_logs_by_source("Ceravolo")
    assert len(ceravolo_logs) == 1
    assert "Ceravolo" in str(ceravolo_logs[0].parent)
    
    # Test case insensitive
    bose_logs_lower = dataset.list_logs_by_source("bose")
    assert len(bose_logs_lower) == 2


def test_cdrift_dataset_get_log_info(tmp_path):
    """Test log info extraction."""
    # Create mock structure
    eval_logs = tmp_path / "EvaluationLogs"
    bose_dir = eval_logs / "Bose"
    bose_dir.mkdir(parents=True)
    
    log_path = bose_dir / "log_cp_100_200.xes"
    log_path.touch()
    
    dataset = CdriftDataset(tmp_path)
    info = dataset.get_log_info(log_path)
    
    assert info["path"] == log_path
    assert info["log_source"] == "Bose"
    assert info["log_name"] == "log_cp_100_200"
    assert info["ground_truth"] == [100, 200]


def test_list_benchmark_logs_structure(tmp_path):
    """Test benchmark log listing returns correct structure."""
    # Create mock structure
    eval_logs = tmp_path / "EvaluationLogs"
    bose_dir = eval_logs / "Bose"
    bose_dir.mkdir(parents=True)
    
    (bose_dir / "log_cp_100.xes").touch()
    (bose_dir / "log_cp_200.xes").touch()
    
    logs = list_benchmark_logs(tmp_path)
    
    assert len(logs) == 2
    
    # Check structure of first entry
    log_entry = logs[0]
    assert "path" in log_entry
    assert "source" in log_entry
    assert "name" in log_entry
    assert "ground_truth" in log_entry
    
    # Check values
    assert isinstance(log_entry["path"], Path)
    assert log_entry["source"] == "Bose"
    assert "log_cp" in log_entry["name"]
    assert isinstance(log_entry["ground_truth"], list)


def test_cdrift_dataset_missing_directory(tmp_path):
    """Test behavior when EvaluationLogs directory doesn't exist."""
    # Initialize with non-existent directory
    dataset = CdriftDataset(tmp_path / "nonexistent")
    
    # Should return empty lists, not crash
    logs = dataset.list_logs()
    assert logs == []
    
    bose_logs = dataset.list_logs_by_source("Bose")
    assert bose_logs == []


@pytest.mark.skipif(
    not Path("cdrift-evaluation").exists(),
    reason="cdrift-evaluation repository not cloned"
)
def test_extract_ground_truth_xes_attributes():
    """Test extraction from actual XES attributes (requires cdrift repo)."""
    # This test requires actual cdrift-evaluation repository
    # If attributes are present in XES logs, this will validate extraction
    pytest.xfail("XES attribute format unclear - requires manual inspection of cdrift logs")
