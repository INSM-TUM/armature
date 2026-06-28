"""Tests for inspect CLI command."""

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
source: test.xes
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


def test_inspect_full_matrix_human(runner, sample_yaml_matrix):
    """Test inspect command with full matrix summary."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    assert "Activity count:" in result.output
    assert "Dependency count:" in result.output
    assert "Density:" in result.output
    assert "Matrix Grid:" in result.output
    assert "Legend:" in result.output
    # Should show grid with activities
    assert "A" in result.output
    assert "B" in result.output
    assert "C" in result.output


def test_inspect_full_matrix_json(runner, sample_yaml_matrix):
    """Test inspect command with JSON output for full matrix."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "--format", "json"])

    assert result.exit_code == 0

    # Parse JSON
    data = json.loads(result.output)
    assert "activity_count" in data
    assert "activities" in data
    assert "dependency_count" in data
    assert "density" in data
    assert data["activity_count"] == 3
    assert set(data["activities"]) == {"A", "B", "C"}


def test_inspect_single_activity(runner, sample_yaml_matrix):
    """Test inspect command with single activity filter."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A"])

    assert result.exit_code == 0
    assert "Cells involving 'A':" in result.output
    assert "As source (A → ...):" in result.output
    assert "As target (... → A):" in result.output
    # Should show relationships
    assert "A → B:" in result.output or "B → A:" in result.output


def test_inspect_single_activity_json(runner, sample_yaml_matrix):
    """Test inspect command with single activity in JSON format."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A", "--format", "json"])

    assert result.exit_code == 0

    # Parse JSON
    data = json.loads(result.output)
    assert "activity" in data
    assert "cells" in data
    assert data["activity"] == "A"
    assert isinstance(data["cells"], dict)


def test_inspect_cell_details(runner, sample_yaml_matrix):
    """Test inspect command with two activities (cell details)."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A", "B"])

    assert result.exit_code == 0
    assert "Cell (A, B):" in result.output
    assert "Temporal:" in result.output
    assert "Existential:" in result.output


def test_inspect_cell_details_json(runner, sample_yaml_matrix):
    """Test inspect command with cell details in JSON format."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A", "B", "--format", "json"])

    assert result.exit_code == 0

    # Parse JSON
    data = json.loads(result.output)
    assert "source" in data
    assert "target" in data
    assert "temporal" in data
    assert "existential" in data
    assert "is_neutral" in data
    assert data["source"] == "A"
    assert data["target"] == "B"


def test_inspect_file_not_found(runner):
    """Test inspect command with non-existent file."""
    result = runner.invoke(cli, ["inspect", "nonexistent.yaml"])

    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "error" in result.output.lower()


def test_inspect_invalid_yaml(runner, tmp_path):
    """Test inspect command with malformed YAML."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml: [")

    result = runner.invoke(cli, ["inspect", str(invalid_yaml)])

    assert result.exit_code != 0
    assert "parse failed" in result.output.lower() or "error" in result.output.lower()


def test_inspect_activity_not_found(runner, sample_yaml_matrix):
    """Test inspect command with unknown activity."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "Z"])

    assert result.exit_code != 0
    assert "not in matrix" in result.output.lower()


def test_inspect_too_many_activities(runner, sample_yaml_matrix):
    """Test inspect command with more than 2 activities."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A", "B", "C"])

    assert result.exit_code != 0
    assert "too many" in result.output.lower()


def test_inspect_grid_rendering(runner, sample_yaml_matrix):
    """Test grid rendering shows dependency symbols."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    # Grid should use first letter of temporal type
    # D=DIRECT, E=EVENTUAL, -=neutral
    assert "D" in result.output or "E" in result.output
    assert "-" in result.output  # Neutral cells


def test_inspect_neutral_cell(runner, sample_yaml_matrix):
    """Test inspect command on neutral cell."""
    result = runner.invoke(cli, ["inspect", str(sample_yaml_matrix), "A", "A"])

    assert result.exit_code == 0
    # Diagonal cells should exist but may be neutral or have self-loop info
    assert "Cell (A, A):" in result.output


def test_inspect_verbose_flag(runner, sample_yaml_matrix):
    """Test inspect command with --verbose flag."""
    result = runner.invoke(cli, ["--verbose", "inspect", str(sample_yaml_matrix)])

    assert result.exit_code == 0
    assert "Activity count:" in result.output


def test_inspect_using_real_matrix(runner):
    """Test inspect with real discovered matrix if available."""
    # Use an existing matrix from fixtures
    matrix_files = list(Path("tests/fixtures/discovery/results").glob("*.yaml"))
    if not matrix_files:
        pytest.skip("No matrix fixtures available")

    matrix_file = matrix_files[0]
    result = runner.invoke(cli, ["inspect", str(matrix_file)])

    assert result.exit_code == 0
    assert "Activity count:" in result.output
    assert "Matrix Grid:" in result.output


def test_inspect_xes_input(runner):
    """Test inspect command with XES input (auto-discovery)."""
    xes_files = list(Path("tests/fixtures/discovery").glob("*.xes"))
    if not xes_files:
        pytest.skip("No XES fixtures available for testing")

    xes_file = xes_files[0]
    result = runner.invoke(cli, ["inspect", str(xes_file)])

    # Should work - auto-discover then inspect
    assert result.exit_code == 0
    assert "Activity count:" in result.output
    assert "Matrix Grid:" in result.output


def test_inspect_xes_json_output(runner):
    """Test inspect command with XES input and JSON output."""
    xes_files = list(Path("tests/fixtures/discovery").glob("*.xes"))
    if not xes_files:
        pytest.skip("No XES fixtures available for testing")

    xes_file = xes_files[0]
    result = runner.invoke(cli, ["inspect", str(xes_file), "--format", "json"])

    assert result.exit_code == 0

    # Should be valid JSON
    data = json.loads(result.output)
    assert "activity_count" in data
    assert "activities" in data
