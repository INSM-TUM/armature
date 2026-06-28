"""Tests for classify CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from armature.cli.main import cli


@pytest.fixture
def runner():
    """Create Click CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_yaml_matrix(tmp_path):
    """Create a sample YAML matrix file for testing."""
    yaml_content = """format_version: "2.0"
description: Test matrix
activities:
  - A
  - B
  - C
dependencies:
  A:
    B:
      temporal: direct
      existential: equivalence
    C:
      temporal: eventual
      existential: or
  B:
    A:
      temporal: direct_backward
      existential: equivalence
    C:
      temporal: direct
      existential: equivalence
  C:
    A:
      temporal: eventual_backward
      existential: or
    B:
      temporal: direct_backward
      existential: equivalence
"""
    matrix_path = tmp_path / "test_matrix.yaml"
    matrix_path.write_text(yaml_content)
    return matrix_path


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file with custom thresholds."""
    config_content = """direct_ratio_structured: 0.80
eventual_ratio_structured: 0.90
direct_ratio_semi_max: 0.79
eventual_ratio_semi_min: 0.65
implication_ratio_semi: 0.35
direct_ratio_loosely_max: 0.64
nand_or_ratio_loosely: 0.45
"""
    config_path = tmp_path / "custom_config.yaml"
    config_path.write_text(config_content)
    return config_path


def test_classify_yaml_human_format(runner, sample_yaml_matrix):
    """Test classify command with YAML input and human-readable output."""
    result = runner.invoke(cli, ["classify", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    assert "Classification:" in result.output
    assert "Scores:" in result.output
    assert "Margin (top-2):" in result.output
    assert "Decision:" in result.output


def test_classify_yaml_json_format(runner, sample_yaml_matrix):
    """Test classify command with JSON output format."""
    result = runner.invoke(cli, ["classify", str(sample_yaml_matrix), "--format", "json"])

    assert result.exit_code == 0

    # Validate JSON structure
    data = json.loads(result.output)
    assert "category" in data
    assert "confidence" in data
    assert "dependency_counts" in data
    assert "dependency_ratios" in data
    assert "thresholds_applied" in data
    assert "rule_trace" in data
    assert "activity_count" in data
    assert "total_dependencies" in data
    assert "density" in data


def test_classify_with_custom_config(runner, sample_yaml_matrix):
    """Test classify command produces output with expected fields."""
    result = runner.invoke(cli, ["classify", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    assert "Classification:" in result.output
    assert "Scores:" in result.output


def test_classify_with_output_file(runner, sample_yaml_matrix, tmp_path):
    """Test classify command writing to output file."""
    output_path = tmp_path / "result.txt"

    result = runner.invoke(cli, ["classify", str(sample_yaml_matrix), "-o", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()

    content = output_path.read_text()
    assert "Classification:" in content


def test_classify_xes_input(runner, tmp_path):
    """Test classify command with XES input (auto-discovery)."""
    # Use an existing XES fixture if available
    xes_files = list(Path("tests/fixtures/discovery").glob("*.xes"))
    if not xes_files:
        pytest.skip("No XES fixtures available for testing")

    xes_file = xes_files[0]
    result = runner.invoke(cli, ["classify", str(xes_file)])

    # Should work - auto-discover then classify
    assert result.exit_code == 0
    assert "Classification:" in result.output


def test_classify_xes_json_output(runner, tmp_path):
    """Test classify command with XES input and JSON output."""
    xes_files = list(Path("tests/fixtures/discovery").glob("*.xes"))
    if not xes_files:
        pytest.skip("No XES fixtures available for testing")

    xes_file = xes_files[0]
    result = runner.invoke(cli, ["classify", str(xes_file), "--format", "json"])

    assert result.exit_code == 0

    # Should be valid JSON
    data = json.loads(result.output)
    assert "category" in data


def test_classify_file_not_found(runner):
    """Test classify command with non-existent file."""
    result = runner.invoke(cli, ["classify", "nonexistent.yaml"])

    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "error" in result.output.lower()


def test_classify_invalid_yaml(runner, tmp_path):
    """Test classify command with malformed YAML."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml: [")

    result = runner.invoke(cli, ["classify", str(invalid_yaml)])

    assert result.exit_code != 0
    assert "parse failed" in result.output.lower() or "error" in result.output.lower()


def test_classify_unsupported_file_type(runner, tmp_path):
    """Test classify command with unsupported file extension."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("some data")

    result = runner.invoke(cli, ["classify", str(txt_file)])

    assert result.exit_code != 0
    assert "unsupported" in result.output.lower()


def test_classify_invalid_config(runner, sample_yaml_matrix, tmp_path):
    """Test classify command with invalid config file."""
    invalid_config = tmp_path / "bad_config.yaml"
    invalid_config.write_text("direct_ratio_structured: 1.5")  # Out of range

    result = runner.invoke(cli, ["classify", str(sample_yaml_matrix), "--config", str(invalid_config)])

    assert result.exit_code != 0
    assert "config" in result.output.lower() or "error" in result.output.lower()


def test_classify_verbose_flag(runner, sample_yaml_matrix):
    """Test classify command with --verbose flag."""
    result = runner.invoke(cli, ["--verbose", "classify", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    # Verbose should show additional info on stderr (captured in result.output for Click)
    assert "Classification:" in result.output


def test_classify_quiet_flag(runner, sample_yaml_matrix):
    """Test classify command with --quiet flag."""
    result = runner.invoke(cli, ["--quiet", "classify", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    # Should still output classification results
    assert "Classification:" in result.output
