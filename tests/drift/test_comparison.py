"""Tests for result comparison utilities."""

import io
from pathlib import Path

import pandas as pd
import pytest

from armature.drift.comparison import (
    ComparisonReport,
    aggregate_by_params,
    compare_algorithms,
    generate_report_markdown,
    plot_comparison,
)


def create_mock_csv(content: str, path: Path):
    """Helper to create mock CSV file."""
    path.write_text(content)


def test_aggregate_by_params(tmp_path):
    """Test aggregation by algorithm and parameters."""
    csv_path = tmp_path / "results.csv"
    csv_content = """Algorithm,Log,Window Size,Threshold,F1-Score,Precision,Recall
ARM,log1,50,0.5,0.8,0.8,0.8
ARM,log2,50,0.5,0.9,0.9,0.9
ARM,log3,100,0.3,0.7,0.7,0.7
"""
    create_mock_csv(csv_content, csv_path)

    result = aggregate_by_params(csv_path)

    # Should have 2 rows (2 param combinations)
    assert len(result) == 2

    # Check first group (Window Size=50, Threshold=0.5)
    row1 = result[
        (result["Window Size"] == 50) & (result["Threshold"] == 0.5)
    ].iloc[0]
    assert row1["mean_f1"] == pytest.approx(0.85)  # (0.8 + 0.9) / 2
    assert row1["count"] == 2

    # Check second group (Window Size=100, Threshold=0.3)
    row2 = result[
        (result["Window Size"] == 100) & (result["Threshold"] == 0.3)
    ].iloc[0]
    assert row2["mean_f1"] == pytest.approx(0.7)
    assert row2["count"] == 1


def test_compare_algorithms_improvement(tmp_path):
    """Test improvement percentage calculation."""
    arm_csv = tmp_path / "arm.csv"
    bose_csv = tmp_path / "bose.csv"

    arm_content = """Algorithm,Log,Window Size,Threshold,F1-Score
ARM,log1,50,0.5,0.8
ARM,log2,50,0.5,0.9
"""
    bose_content = """Algorithm,Log,Window Size,Threshold,F1-Score
Bose,log1,50,0.5,0.6
Bose,log2,50,0.5,0.6
"""

    create_mock_csv(arm_content, arm_csv)
    create_mock_csv(bose_content, bose_csv)

    report = compare_algorithms(arm_csv, bose_csv)

    # ARM mean: 0.85, Bose mean: 0.6
    # Improvement: (0.85 - 0.6) / 0.6 * 100 = 41.67%
    assert report.arm_best_f1 == pytest.approx(0.85)
    assert report.bose_best_f1 == pytest.approx(0.6)
    assert report.improvement_pct == pytest.approx(41.67, rel=0.01)


def test_compare_algorithms_wins(tmp_path):
    """Test per-log win/lose/tie counting."""
    arm_csv = tmp_path / "arm.csv"
    bose_csv = tmp_path / "bose.csv"

    arm_content = """Algorithm,Log,Window Size,Threshold,F1-Score
ARM,log1,50,0.5,0.9
ARM,log2,50,0.5,0.6
ARM,log3,50,0.5,0.8
"""
    bose_content = """Algorithm,Log,Window Size,Threshold,F1-Score
Bose,log1,50,0.5,0.7
Bose,log2,50,0.5,0.6
Bose,log3,50,0.5,0.9
"""

    create_mock_csv(arm_content, arm_csv)
    create_mock_csv(bose_content, bose_csv)

    report = compare_algorithms(arm_csv, bose_csv)

    # log1: ARM=0.9 > Bose=0.7 → ARM wins
    # log2: ARM=0.6 = Bose=0.6 → Tie
    # log3: ARM=0.8 < Bose=0.9 → Bose wins
    assert report.arm_wins_count == 1
    assert report.bose_wins_count == 1
    assert report.ties_count == 1


def test_plot_comparison_creates_file(tmp_path):
    """Test plot generation creates PNG file."""
    arm_csv = tmp_path / "arm.csv"
    bose_csv = tmp_path / "bose.csv"
    output_png = tmp_path / "comparison.png"

    csv_content = """Algorithm,Log,F1-Score
ARM,log1,0.8
ARM,log2,0.9
"""
    create_mock_csv(csv_content, arm_csv)
    create_mock_csv(csv_content, bose_csv)

    plot_comparison(arm_csv, bose_csv, output_png)

    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_generate_report_markdown():
    """Test markdown report generation."""
    report = ComparisonReport(
        arm_best_f1=0.85,
        arm_best_params={"Window Size": 50, "Threshold": 0.5},
        bose_best_f1=0.6,
        bose_best_params={"Window Size": 100, "Threshold": 0.3},
        improvement_pct=41.67,
        arm_wins_count=3,
        bose_wins_count=1,
        ties_count=0,
    )

    markdown = generate_report_markdown(report)

    # Check key sections present
    assert "# ARM vs Bose Benchmark Comparison" in markdown
    assert "**ARM Best F1:** 0.850" in markdown
    assert "**Bose Best F1:** 0.600" in markdown
    assert "**Improvement:** +41.7%" in markdown
    assert "- ARM Wins: 3" in markdown
    assert "- Bose Wins: 1" in markdown
    assert "- Ties: 0" in markdown
    assert "Window Size: 50" in markdown
    assert "Threshold: 0.5" in markdown
