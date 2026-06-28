"""Comparative test suite for ARM vs Bose drift detection.

These tests programmatically verify that ARM's richer dependency model
enables superior drift detection compared to Bose's S/N/A approach.

Success criteria from Phase 4 requirements:
1. ARM detects all drifts Bose finds PLUS additional ones (coverage)
2. ARM detects drifts with fewer traces than Bose (timing)
"""
import pytest
from pathlib import Path
from typing import List, Tuple

from armature.drift import ARMDriftDetector, BoseDriftDetector
from armature.drift.visualization import (
    ComparisonResult,
    DetectionResult,
    generate_comparison_report,
    plot_drift_comparison,
)
from armature.discovery.xes_parser import parse_xes
import pm4py


# Counter-example scenarios with expected outcomes
COUNTER_EXAMPLES = [
    pytest.param(
        "drift_01_existential.xes",
        50,  # ground truth drift point
        "existential_implication_to_independence",
        "ARM detects IMPLICATION->INDEPENDENCE change; Bose sees 'S' (sometimes) in both",
        True,   # ARM should detect
        False,  # Bose should NOT detect
        id="existential_drift"
    ),
    pytest.param(
        "drift_02_temporal_directness.xes",
        50,
        "temporal_direct_to_eventual",
        "ARM detects DIRECT->EVENTUAL change; Bose sees succession in both",
        True,
        False,  # Bose unlikely to detect directness change
        id="temporal_directness_drift"
    ),
    pytest.param(
        "drift_03_combined.xes",
        50,
        "combined_multiple_changes",
        "Multiple ARM signals (temporal + existential + structure) enable earlier detection",
        True,
        True,  # Bose may detect but later
        id="combined_drift"
    ),
    pytest.param(
        "drift_04_subtle_implication.xes",
        50,
        "subtle_implication_to_xor",
        "ARM detects shift from both B,C to exactly one (XOR); invisible to succession analysis",
        True,
        False,
        id="subtle_existential_drift"
    ),
]


class TestARMDetectsMore:
    """Verify ARM detects drifts that Bose misses (coverage advantage)."""

    @pytest.mark.parametrize(
        "log_file,drift_point,scenario_name,explanation,arm_should_detect,bose_should_detect",
        COUNTER_EXAMPLES
    )
    def test_arm_coverage_advantage(
        self,
        drift_logs_dir,
        log_file,
        drift_point,
        scenario_name,
        explanation,
        arm_should_detect,
        bose_should_detect,
    ):
        """Test that ARM detects drifts that Bose misses."""
        log_path = drift_logs_dir / log_file
        if not log_path.exists():
            pytest.skip(f"Counter-example log not found: {log_path}")

        # Run ARM detector
        arm_detector = ARMDriftDetector(window_size=20, threshold=0.05, step_size=5)
        traces = parse_xes(log_path)
        arm_result = arm_detector.detect(traces)

        # Run Bose detector
        bose_detector = BoseDriftDetector(window_size=20, measure="j", stat_test="mu")
        log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
        bose_result = bose_detector.detect(log)

        arm_detected = len(arm_result.drift_indices) > 0
        bose_detected = len(bose_result.drift_indices) > 0

        # Verify ARM detection matches expectation
        if arm_should_detect:
            assert arm_detected, (
                f"ARM should detect drift in {scenario_name} but didn't. "
                f"Scores: {arm_result.drift_scores[:5]}..."
            )

        # Verify coverage advantage: ARM detects what Bose misses
        if arm_should_detect and not bose_should_detect:
            assert arm_detected and not bose_detected, (
                f"Expected ARM advantage in {scenario_name}: "
                f"ARM detected={arm_detected}, Bose detected={bose_detected}"
            )


class TestARMDetectsEarlier:
    """Verify ARM detects drifts earlier than Bose (timing advantage)."""

    @pytest.mark.parametrize(
        "log_file,drift_point,scenario_name,explanation,arm_should_detect,bose_should_detect",
        [ce for ce in COUNTER_EXAMPLES if ce.values[4] and ce.values[5]]  # Both should detect
    )
    def test_arm_timing_advantage(
        self,
        drift_logs_dir,
        log_file,
        drift_point,
        scenario_name,
        explanation,
        arm_should_detect,
        bose_should_detect,
    ):
        """Test that ARM detects drifts earlier than Bose when both detect."""
        log_path = drift_logs_dir / log_file
        if not log_path.exists():
            pytest.skip(f"Counter-example log not found: {log_path}")

        # Run ARM detector
        arm_detector = ARMDriftDetector(window_size=20, threshold=0.05, step_size=5)
        traces = parse_xes(log_path)
        arm_result = arm_detector.detect(traces)

        # Run Bose detector
        bose_detector = BoseDriftDetector(window_size=20, measure="j", stat_test="mu")
        log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
        bose_result = bose_detector.detect(log)

        if arm_result.drift_indices and bose_result.drift_indices:
            arm_first = min(arm_result.drift_indices)
            bose_first = min(bose_result.drift_indices)

            # ARM should detect at same time or earlier
            assert arm_first <= bose_first, (
                f"ARM should detect earlier in {scenario_name}: "
                f"ARM={arm_first}, Bose={bose_first}"
            )


class TestComparisonReportGeneration:
    """Test report generation functionality."""

    def test_generate_full_comparison_report(self, drift_logs_dir, tmp_path):
        """Generate comparison report for all counter-examples."""
        reports_dir = tmp_path / "drift_comparison"
        comparisons: List[ComparisonResult] = []

        for param in COUNTER_EXAMPLES:
            log_file, drift_point, scenario_name, explanation, arm_should, bose_should = param.values

            log_path = drift_logs_dir / log_file
            if not log_path.exists():
                continue

            # Run detectors
            arm_detector = ARMDriftDetector(window_size=20, threshold=0.05, step_size=5)
            traces = parse_xes(log_path)
            arm_result = arm_detector.detect(traces)

            bose_detector = BoseDriftDetector(window_size=20, measure="j", stat_test="mu")
            log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
            bose_result = bose_detector.detect(log)

            # Determine advantage
            arm_detected = len(arm_result.drift_indices) > 0
            bose_detected = len(bose_result.drift_indices) > 0

            if arm_detected and not bose_detected:
                advantage = "coverage"
            elif arm_detected and bose_detected:
                arm_first = min(arm_result.drift_indices)
                bose_first = min(bose_result.drift_indices)
                if arm_first < bose_first:
                    advantage = "timing"
                elif arm_first == bose_first:
                    advantage = "none"
                else:
                    advantage = "none"  # Bose was faster (unexpected)
            elif not arm_detected and bose_detected:
                advantage = "none"  # Bose advantage (unexpected)
            else:
                advantage = "none"  # Neither detected

            comparisons.append(ComparisonResult(
                scenario_name=scenario_name,
                ground_truth_drift=drift_point,
                total_traces=len(traces),
                arm_result=DetectionResult(
                    detector_name="ARM",
                    drift_indices=arm_result.drift_indices,
                    scores=arm_result.drift_scores,
                    threshold=arm_result.threshold,
                ),
                bose_result=DetectionResult(
                    detector_name="Bose",
                    drift_indices=bose_result.drift_indices,
                ),
                arm_advantage=advantage,
                explanation=explanation,
            ))

        # Generate report
        report_path = generate_comparison_report(comparisons, reports_dir)

        assert report_path.exists()
        assert (reports_dir / "README.md").exists()

        # Verify at least one plot was generated
        plot_files = list(reports_dir.glob("*.png"))
        assert len(plot_files) > 0, "No comparison plots generated"


class TestOverallSuperiorityRequirement:
    """Verify phase success criteria are met."""

    def test_arm_wins_majority(self, drift_logs_dir):
        """ARM must demonstrate advantage on majority of counter-examples."""
        arm_wins = 0
        total_valid = 0

        for param in COUNTER_EXAMPLES:
            log_file, drift_point, scenario_name, explanation, arm_should, bose_should = param.values

            log_path = drift_logs_dir / log_file
            if not log_path.exists():
                continue

            total_valid += 1

            # Run detectors
            arm_detector = ARMDriftDetector(window_size=20, threshold=0.05, step_size=5)
            traces = parse_xes(log_path)
            arm_result = arm_detector.detect(traces)

            bose_detector = BoseDriftDetector(window_size=20, measure="j", stat_test="mu")
            log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
            bose_result = bose_detector.detect(log)

            arm_detected = len(arm_result.drift_indices) > 0
            bose_detected = len(bose_result.drift_indices) > 0

            # Count ARM wins
            if arm_detected and not bose_detected:
                arm_wins += 1  # Coverage advantage
            elif arm_detected and bose_detected:
                arm_first = min(arm_result.drift_indices)
                bose_first = min(bose_result.drift_indices)
                if arm_first < bose_first:
                    arm_wins += 1  # Timing advantage

        # ARM must win on majority of test cases
        assert total_valid > 0, "No valid counter-example logs found"
        win_rate = arm_wins / total_valid
        assert win_rate >= 0.5, (
            f"ARM must demonstrate superiority on majority of cases. "
            f"Won {arm_wins}/{total_valid} ({win_rate:.1%})"
        )
