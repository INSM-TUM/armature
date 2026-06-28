"""Tests for CLI utilities (TTY detection, output handling, progress)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from armature.cli.utils import format_matrix_as_json, is_tty, show_progress, write_output
from armature.core.matrix import Matrix


class TestTTYDetection:
    """Tests for is_tty() function."""

    def test_tty_with_real_file(self, tmp_path: Path) -> None:
        """Real files are not TTYs."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with open(test_file) as f:
            assert not is_tty(f)

    def test_tty_with_stringio(self) -> None:
        """StringIO objects are not TTYs."""
        string_io = io.StringIO()
        assert not is_tty(string_io)

    def test_tty_with_mock_tty(self, monkeypatch) -> None:
        """Mock TTY file descriptor."""

        class MockTTY:
            def isatty(self):
                return True

        assert is_tty(MockTTY())

    def test_tty_with_mock_pipe(self, monkeypatch) -> None:
        """Mock piped file descriptor."""

        class MockPipe:
            def isatty(self):
                return False

        assert not is_tty(MockPipe())

    def test_tty_without_isatty_method(self) -> None:
        """Objects without isatty() method return False."""

        class NoIsatty:
            pass

        assert not is_tty(NoIsatty())


class TestWriteOutput:
    """Tests for write_output() function."""

    def test_write_to_stdout(self, capsys) -> None:
        """Content written to stdout when no path specified."""
        content = "test output"
        write_output(content, None, "yaml")

        captured = capsys.readouterr()
        assert captured.out.strip() == content

    def test_write_to_file(self, tmp_path: Path) -> None:
        """Content written to file when path specified."""
        output_file = tmp_path / "output.yaml"
        content = "test content"

        write_output(content, output_file, "yaml")

        assert output_file.exists()
        assert output_file.read_text() == content

    def test_write_json_format(self, tmp_path: Path) -> None:
        """JSON format accepted."""
        output_file = tmp_path / "output.json"
        content = '{"test": true}'

        write_output(content, output_file, "json")

        assert output_file.exists()
        assert output_file.read_text() == content

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            write_output("content", None, "xml")


class TestShowProgress:
    """Tests for show_progress() wrapper."""

    def test_progress_with_iterable(self) -> None:
        """Progress wrapper works with iterables."""
        items = [1, 2, 3, 4, 5]
        result = list(show_progress("Test", items))
        assert result == items

    def test_progress_suppressed_when_not_tty(self, monkeypatch) -> None:
        """Progress bar disabled when stderr is not TTY."""

        # Mock stderr as non-TTY
        class MockPipe:
            def isatty(self):
                return False

        import sys

        monkeypatch.setattr(sys, "stderr", MockPipe())

        items = [1, 2, 3]
        pbar = show_progress("Test", items)

        # Should still iterate correctly
        result = list(pbar)
        assert result == items

    def test_progress_with_total(self) -> None:
        """Progress wrapper works with total count."""
        pbar = show_progress("Test", total=100)
        pbar.update(50)
        pbar.close()


class TestFormatMatrixAsJSON:
    """Tests for format_matrix_as_json() function."""

    def test_format_simple_matrix(self) -> None:
        """Matrix serialized to valid JSON."""
        matrix = Matrix(activities=["A", "B"])
        json_str = format_matrix_as_json(matrix)

        # Parse to verify valid JSON
        data = json.loads(json_str)
        assert data["format_version"] == "2.0"
        assert data["activities"] == ["A", "B"]

    def test_format_matrix_excludes_none(self) -> None:
        """None values excluded from JSON output."""
        matrix = Matrix(activities=["A"], source=None)
        json_str = format_matrix_as_json(matrix)

        data = json.loads(json_str)
        assert "source" not in data or data.get("source") is None

    def test_format_matrix_with_metadata(self) -> None:
        """Matrix with metadata serialized correctly."""
        matrix = Matrix(
            activities=["A", "B"],
            source="test.xes",
            description="Test matrix",
        )
        json_str = format_matrix_as_json(matrix)

        data = json.loads(json_str)
        assert data["source"] == "test.xes"
        assert data["description"] == "Test matrix"
