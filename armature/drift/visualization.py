"""Drift detection visualization for ARM vs Bose comparison.

Generates timeline plots showing when each detector identifies drifts,
and detailed comparison reports explaining detection outcomes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class DetectionResult:
    """Result from a single detector."""
    detector_name: str
    drift_indices: List[int]
    scores: Optional[List[float]] = None
    threshold: Optional[float] = None


@dataclass
class ComparisonResult:
    """Result from comparing ARM and Bose on a scenario."""
    scenario_name: str
    ground_truth_drift: int
    total_traces: int
    arm_result: DetectionResult
    bose_result: DetectionResult
    arm_advantage: str  # "coverage", "timing", "both", "none"
    explanation: str


def plot_drift_comparison(
    comparison: ComparisonResult,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot drift detection timeline comparing ARM and Bose.

    Creates a timeline showing:
    - Ground truth drift point (red dashed line)
    - ARM detection point(s) (green solid line)
    - Bose detection point(s) (blue solid line)
    - Drift scores over time (if available)

    Args:
        comparison: ComparisonResult with detection data
        output_path: Path to save PNG (optional)
        show: Whether to display plot interactively
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    trace_indices = list(range(comparison.total_traces))

    # Plot ARM drift scores if available
    if comparison.arm_result.scores:
        ax.plot(
            trace_indices[:len(comparison.arm_result.scores)],
            comparison.arm_result.scores,
            color='green',
            alpha=0.5,
            linewidth=1,
            label='ARM Drift Score'
        )
        if comparison.arm_result.threshold:
            ax.axhline(
                comparison.arm_result.threshold,
                color='green',
                linestyle=':',
                alpha=0.3,
                label=f'ARM Threshold ({comparison.arm_result.threshold})'
            )

    # Ground truth drift point
    ax.axvline(
        comparison.ground_truth_drift,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Ground Truth (trace {comparison.ground_truth_drift})'
    )

    # ARM detections
    for idx in comparison.arm_result.drift_indices:
        ax.axvline(
            idx,
            color='green',
            linestyle='-',
            linewidth=2,
            alpha=0.8,
        )
    if comparison.arm_result.drift_indices:
        ax.axvline(
            comparison.arm_result.drift_indices[0],
            color='green',
            linestyle='-',
            linewidth=2,
            label=f'ARM Detection (trace {comparison.arm_result.drift_indices[0]})'
        )

    # Bose detections
    for idx in comparison.bose_result.drift_indices:
        ax.axvline(
            idx,
            color='blue',
            linestyle='-',
            linewidth=2,
            alpha=0.8,
        )
    if comparison.bose_result.drift_indices:
        ax.axvline(
            comparison.bose_result.drift_indices[0],
            color='blue',
            linestyle='-',
            linewidth=2,
            label=f'Bose Detection (trace {comparison.bose_result.drift_indices[0]})'
        )
    else:
        # Add legend entry for no detection (use empty plot with label)
        ax.plot([], [], color='blue', alpha=0.3, linewidth=0, label='Bose: No Detection')

    # Annotations for timing
    if comparison.arm_result.drift_indices:
        arm_first = comparison.arm_result.drift_indices[0]
        delay = arm_first - comparison.ground_truth_drift
        ax.annotate(
            f'ARM: {delay:+d} traces',
            xy=(arm_first, 0.9),
            xytext=(arm_first + 5, 0.95),
            fontsize=9,
            color='green',
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.5)
        )

    if comparison.bose_result.drift_indices:
        bose_first = comparison.bose_result.drift_indices[0]
        delay = bose_first - comparison.ground_truth_drift
        ax.annotate(
            f'Bose: {delay:+d} traces',
            xy=(bose_first, 0.8),
            xytext=(bose_first + 5, 0.85),
            fontsize=9,
            color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5)
        )

    ax.set_xlabel('Trace Index')
    ax.set_ylabel('Drift Score')
    ax.set_title(f'Concept Drift Detection: {comparison.scenario_name}\n'
                 f'ARM Advantage: {comparison.arm_advantage}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, comparison.total_traces)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150)

    if show:
        plt.show()

    plt.close()


def generate_comparison_report(
    comparisons: List[ComparisonResult],
    output_dir: Path,
) -> Path:
    """Generate comprehensive comparison report.

    Creates:
    - Individual timeline plots for each scenario
    - Summary markdown report with findings

    Args:
        comparisons: List of comparison results
        output_dir: Directory for output files

    Returns:
        Path to generated README.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots for each comparison
    for comp in comparisons:
        plot_path = output_dir / f"{comp.scenario_name}.png"
        plot_drift_comparison(comp, output_path=plot_path)

    # Generate summary report
    report_path = output_dir / "README.md"

    lines = [
        "# ARM vs Bose Drift Detection Comparison Report",
        "",
        "This report summarizes the comparative evaluation of ARM-based drift detection",
        "against Bose's S/N/A approach across synthetic counter-example logs.",
        "",
        "## Summary",
        "",
        "| Scenario | Ground Truth | ARM Detected | Bose Detected | ARM Advantage |",
        "|----------|--------------|--------------|---------------|---------------|",
    ]

    arm_wins = 0
    bose_wins = 0
    ties = 0

    for comp in comparisons:
        arm_det = comp.arm_result.drift_indices[0] if comp.arm_result.drift_indices else "No"
        bose_det = comp.bose_result.drift_indices[0] if comp.bose_result.drift_indices else "No"
        lines.append(
            f"| {comp.scenario_name} | {comp.ground_truth_drift} | "
            f"{arm_det} | {bose_det} | {comp.arm_advantage} |"
        )

        if comp.arm_advantage in ["coverage", "timing", "both"]:
            arm_wins += 1
        elif comp.arm_advantage == "none":
            if comp.arm_result.drift_indices and comp.bose_result.drift_indices:
                ties += 1
            elif comp.bose_result.drift_indices and not comp.arm_result.drift_indices:
                bose_wins += 1

    lines.extend([
        "",
        "## Overall Results",
        "",
        f"- **ARM Wins:** {arm_wins}/{len(comparisons)} scenarios",
        f"- **Ties:** {ties}/{len(comparisons)} scenarios",
        f"- **Bose Wins:** {bose_wins}/{len(comparisons)} scenarios",
        "",
        "## Detailed Analysis",
        "",
    ])

    for comp in comparisons:
        lines.extend([
            f"### {comp.scenario_name}",
            "",
            f"![{comp.scenario_name}]({comp.scenario_name}.png)",
            "",
            f"**Ground Truth Drift:** Trace {comp.ground_truth_drift}",
            "",
            f"**ARM Detection:** {comp.arm_result.drift_indices or 'Not detected'}",
            "",
            f"**Bose Detection:** {comp.bose_result.drift_indices or 'Not detected'}",
            "",
            f"**Advantage:** {comp.arm_advantage}",
            "",
            f"**Explanation:** {comp.explanation}",
            "",
        ])

    lines.extend([
        "## Conclusion",
        "",
        "ARM's richer dependency model (temporal: DIRECT/EVENTUAL/TRUE_EVENTUAL, "
        "existential: IMPLICATION/EQUIVALENCE/XOR/OR/NAND/INDEPENDENCE) enables:",
        "",
        "1. **Better Coverage:** Detection of drifts invisible to succession-only analysis",
        "2. **Earlier Timing:** Stronger signals from multiple dependency dimensions",
        "",
        "The counter-example logs specifically target Bose's limitations:",
        "- Existential changes (IMPLICATION->INDEPENDENCE) are invisible to S/N/A",
        "- Directness changes (DIRECT->EVENTUAL) preserve succession but alter structure",
        "- Combined changes produce stronger ARM signals for earlier detection",
        "",
        "---",
        "*Generated by armature drift comparison suite*",
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return report_path
