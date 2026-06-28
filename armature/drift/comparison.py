"""Result aggregation and comparison utilities for ARM vs Bose benchmarks.

Provides tools to analyze benchmark results, compare algorithms, and visualize
performance differences.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def aggregate_by_params(csv_path: Path) -> pd.DataFrame:
    """Aggregate results by algorithm and parameter combination.

    Groups results by Algorithm + parameter columns (Window Size, Threshold, Step Size)
    and computes mean metrics across all logs for each parameter combination.

    Args:
        csv_path: Path to results CSV file

    Returns:
        DataFrame with one row per parameter combination, containing:
        - Algorithm, Window Size, Threshold, Step Size (index)
        - mean_f1, mean_precision, mean_recall, count
    """
    df = pd.read_csv(csv_path)

    # Identify parameter columns (vary by algorithm)
    param_cols = ["Algorithm"]
    if "Window Size" in df.columns:
        param_cols.append("Window Size")
    if "Threshold" in df.columns:
        param_cols.append("Threshold")
    if "Step Size" in df.columns:
        param_cols.append("Step Size")

    # Compute precision and recall from F1 if needed
    if "F1-Score" in df.columns and "Precision" not in df.columns:
        # Estimate from F1 (assumes balanced precision/recall)
        df["Precision"] = df["F1-Score"]
        df["Recall"] = df["F1-Score"]

    # Group and aggregate
    grouped = df.groupby(param_cols, dropna=False).agg(
        mean_f1=("F1-Score", "mean"),
        mean_precision=("Precision", "mean") if "Precision" in df.columns else ("F1-Score", "mean"),
        mean_recall=("Recall", "mean") if "Recall" in df.columns else ("F1-Score", "mean"),
        count=("F1-Score", "count"),
    )

    return grouped.reset_index()


@dataclass
class ComparisonReport:
    """Comparison report between ARM and Bose algorithms.

    Attributes:
        arm_best_f1: Best F1 score achieved by ARM
        arm_best_params: Parameter combination for ARM's best F1
        bose_best_f1: Best F1 score achieved by Bose
        bose_best_params: Parameter combination for Bose's best F1
        improvement_pct: Percentage improvement (arm - bose) / bose * 100
        arm_wins_count: Number of logs where ARM F1 > Bose F1
        bose_wins_count: Number of logs where Bose F1 > ARM F1
        ties_count: Number of logs where F1 scores are equal
    """

    arm_best_f1: float
    arm_best_params: dict
    bose_best_f1: float
    bose_best_params: dict
    improvement_pct: float
    arm_wins_count: int
    bose_wins_count: int
    ties_count: int


def compare_algorithms(arm_csv: Path, bose_csv: Path) -> ComparisonReport:
    """Compare ARM and Bose benchmark results.

    Identifies best parameters for each algorithm and counts per-log wins.

    Args:
        arm_csv: Path to ARM results CSV
        bose_csv: Path to Bose results CSV

    Returns:
        ComparisonReport with performance comparison
    """
    # Aggregate by params
    arm_agg = aggregate_by_params(arm_csv)
    bose_agg = aggregate_by_params(bose_csv)

    # Find best F1 for each algorithm
    arm_best_idx = arm_agg["mean_f1"].idxmax()
    arm_best = arm_agg.iloc[arm_best_idx]
    arm_best_f1 = arm_best["mean_f1"]

    bose_best_idx = bose_agg["mean_f1"].idxmax()
    bose_best = bose_agg.iloc[bose_best_idx]
    bose_best_f1 = bose_best["mean_f1"]

    # Extract best params (excluding Algorithm and aggregated metrics)
    metric_cols = {"mean_f1", "mean_precision", "mean_recall", "count"}
    arm_best_params = {
        k: v for k, v in arm_best.items() if k not in metric_cols and k != "Algorithm"
    }
    bose_best_params = {
        k: v for k, v in bose_best.items() if k not in metric_cols and k != "Algorithm"
    }

    # Compute improvement
    if bose_best_f1 > 0:
        improvement_pct = (arm_best_f1 - bose_best_f1) / bose_best_f1 * 100
    else:
        improvement_pct = float("inf") if arm_best_f1 > 0 else 0.0

    # Count per-log wins
    arm_df = pd.read_csv(arm_csv)
    bose_df = pd.read_csv(bose_csv)

    # Match on log name
    arm_logs = arm_df.groupby("Log")["F1-Score"].max()
    bose_logs = bose_df.groupby("Log")["F1-Score"].max()

    # Find common logs
    common_logs = set(arm_logs.index) & set(bose_logs.index)

    arm_wins = 0
    bose_wins = 0
    ties = 0

    for log in common_logs:
        arm_f1 = arm_logs[log]
        bose_f1 = bose_logs[log]

        if arm_f1 > bose_f1:
            arm_wins += 1
        elif bose_f1 > arm_f1:
            bose_wins += 1
        else:
            ties += 1

    return ComparisonReport(
        arm_best_f1=arm_best_f1,
        arm_best_params=arm_best_params,
        bose_best_f1=bose_best_f1,
        bose_best_params=bose_best_params,
        improvement_pct=improvement_pct,
        arm_wins_count=arm_wins,
        bose_wins_count=bose_wins,
        ties_count=ties,
    )


def plot_comparison(arm_csv: Path, bose_csv: Path, output_path: Path):
    """Create visualization comparing ARM and Bose F1 distributions.

    Generates figure with:
    - Left subplot: Overlaid histogram of F1 distributions
    - Right subplot: Box plot comparison

    Args:
        arm_csv: Path to ARM results CSV
        bose_csv: Path to Bose results CSV
        output_path: Path to save PNG output

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    arm_df = pd.read_csv(arm_csv)
    bose_df = pd.read_csv(bose_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram
    ax1.hist(
        arm_df["F1-Score"],
        bins=20,
        alpha=0.6,
        label="ARM",
        color="blue",
        edgecolor="black",
    )
    ax1.hist(
        bose_df["F1-Score"],
        bins=20,
        alpha=0.6,
        label="Bose",
        color="orange",
        edgecolor="black",
    )
    ax1.set_xlabel("F1-Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("F1-Score Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Box plot
    data_to_plot = [arm_df["F1-Score"], bose_df["F1-Score"]]
    ax2.boxplot(data_to_plot, tick_labels=["ARM", "Bose"], patch_artist=True)
    ax2.set_ylabel("F1-Score")
    ax2.set_title("F1-Score Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return fig


def generate_report_markdown(report: ComparisonReport) -> str:
    """Generate markdown report from comparison results.

    Args:
        report: ComparisonReport to format

    Returns:
        Markdown-formatted string suitable for README or SUMMARY
    """
    lines = [
        "# ARM vs Bose Benchmark Comparison",
        "",
        "## Key Findings",
        "",
        f"- **ARM Best F1:** {report.arm_best_f1:.3f}",
        f"- **Bose Best F1:** {report.bose_best_f1:.3f}",
        f"- **Improvement:** {report.improvement_pct:+.1f}%",
        "",
        "## Best Parameters",
        "",
        "### ARM",
        "",
    ]

    for k, v in report.arm_best_params.items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "### Bose", ""])

    for k, v in report.bose_best_params.items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## Per-Log Results",
            "",
            f"- ARM Wins: {report.arm_wins_count}",
            f"- Bose Wins: {report.bose_wins_count}",
            f"- Ties: {report.ties_count}",
            "",
        ]
    )

    return "\n".join(lines)
