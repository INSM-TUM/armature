"""Tests for discover CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from armature.cli.main import cli


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_xes(tmp_path: Path) -> Path:
    """Create a minimal valid XES file for testing."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="case1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2024-01-01T10:01:00.000+00:00"/>
    </event>
  </trace>
</log>
"""
    xes_file = tmp_path / "test.xes"
    xes_file.write_text(xes_content)
    return xes_file


@pytest.fixture
def threshold_config(tmp_path: Path) -> Path:
    """Create a threshold config YAML file."""
    config_content = """
direct_ratio_structured: 0.80
eventual_ratio_structured: 0.85
direct_ratio_semi_max: 0.74
eventual_ratio_semi_min: 0.60
"""
    config_file = tmp_path / "thresholds.yaml"
    config_file.write_text(config_content)
    return config_file


class TestDiscoverBasic:
    """Basic discover command tests."""

    def test_discover_yaml_to_stdout(self, runner: CliRunner, sample_xes: Path) -> None:
        """Discover outputs YAML to stdout by default."""
        result = runner.invoke(cli, ["discover", str(sample_xes)])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "format_version:" in result.output
        assert "activities:" in result.output

        # Validate it's valid YAML
        data = yaml.safe_load(result.output)
        assert data["metadata"]["format_version"] == "2.0"
        assert "A" in data["activities"]
        assert "B" in data["activities"]

    def test_discover_json_output(self, runner: CliRunner, sample_xes: Path) -> None:
        """Discover outputs JSON with --format json."""
        result = runner.invoke(cli, ["discover", str(sample_xes), "--format", "json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Validate it's valid JSON
        data = json.loads(result.output)
        assert data["format_version"] == "2.0"
        assert "A" in data["activities"]
        assert "B" in data["activities"]

    def test_discover_to_file(
        self, runner: CliRunner, sample_xes: Path, tmp_path: Path
    ) -> None:
        """Discover writes to file with -o flag."""
        output_file = tmp_path / "output.yaml"
        result = runner.invoke(cli, ["discover", str(sample_xes), "-o", str(output_file)])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate file content
        data = yaml.safe_load(output_file.read_text())
        assert data["metadata"]["format_version"] == "2.0"
        assert "A" in data["activities"]

    def test_discover_file_not_found(self, runner: CliRunner) -> None:
        """Discover fails with clear error for missing file."""
        result = runner.invoke(cli, ["discover", "/nonexistent/file.xes"])

        # Click validates path existence and exits with 2
        assert result.exit_code in (1, 2)
        assert "does not exist" in result.output or "File not found" in result.output


class TestDiscoverConfig:
    """Config and threshold parameter tests."""

    def test_discover_with_config_file(
        self, runner: CliRunner, sample_xes: Path, threshold_config: Path
    ) -> None:
        """Discover loads config file and shows warning."""
        result = runner.invoke(
            cli, ["discover", str(sample_xes), "--config", str(threshold_config)]
        )

        # Should succeed but warn
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Warning: Discovery thresholds not yet implemented" in result.output

    def test_discover_with_threshold_flags(
        self, runner: CliRunner, sample_xes: Path
    ) -> None:
        """Discover parses threshold override flags."""
        result = runner.invoke(
            cli,
            [
                "discover",
                str(sample_xes),
                "--threshold-eventual",
                "0.8",
                "--threshold-direct",
                "0.6",
            ],
        )

        # Should succeed but warn
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Warning: Discovery thresholds not yet implemented" in result.output

    def test_discover_config_with_flag_override(
        self, runner: CliRunner, sample_xes: Path, threshold_config: Path
    ) -> None:
        """Threshold flags override config file values."""
        result = runner.invoke(
            cli,
            [
                "discover",
                str(sample_xes),
                "--config",
                str(threshold_config),
                "--threshold-eventual",
                "0.9",
            ],
        )

        # Should succeed and warn
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Warning: Discovery thresholds not yet implemented" in result.output

    def test_discover_invalid_config_file(
        self, runner: CliRunner, sample_xes: Path
    ) -> None:
        """Discover fails gracefully with invalid config path."""
        result = runner.invoke(
            cli, ["discover", str(sample_xes), "--config", "/nonexistent/config.yaml"]
        )

        # Click validates path existence and exits with 2
        assert result.exit_code in (1, 2)
        assert "does not exist" in result.output or "Config validation failed" in result.output


class TestDiscoverMetadata:
    """Metadata and source parameter tests."""

    def test_discover_with_source_metadata(
        self, runner: CliRunner, sample_xes: Path
    ) -> None:
        """Discover accepts --source metadata parameter."""
        result = runner.invoke(
            cli,
            ["discover", str(sample_xes), "--source", "test-source", "--format", "json"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Check source in output
        data = json.loads(result.output)
        assert data["source"] == "test-source"

    def test_discover_source_defaults_to_path(
        self, runner: CliRunner, sample_xes: Path
    ) -> None:
        """Discover uses file path as source if not specified."""
        result = runner.invoke(cli, ["discover", str(sample_xes), "--format", "json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        data = json.loads(result.output)
        assert data["source"] == str(sample_xes)


class TestDiscoverErrors:
    """Error handling tests."""

    def test_discover_no_input_shows_error(self, runner: CliRunner) -> None:
        """Discover requires input file."""
        result = runner.invoke(cli, ["discover"])

        assert result.exit_code != 0
        # Click may show usage or our error message
        assert "required" in result.output.lower() or "error" in result.output.lower()

    def test_discover_invalid_xes_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Discover handles malformed XES gracefully."""
        bad_xes = tmp_path / "bad.xes"
        bad_xes.write_text("not valid XML")

        result = runner.invoke(cli, ["discover", str(bad_xes)])

        assert result.exit_code == 1
        assert "parse failed" in result.output.lower() or "error" in result.output.lower()
