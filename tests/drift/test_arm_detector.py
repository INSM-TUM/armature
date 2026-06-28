"""Tests for ARM-based drift detection."""
import pytest
from pathlib import Path

from armature.drift import ARMDriftDetector, detect_drift_arm, compute_drift_score
from armature.core.matrix import Matrix
from armature.core.dependencies import (
    DependencyCell,
    TemporalDependency,
    ExistentialDependency,
)


class TestDriftMetrics:
    """Test drift score computation."""

    def test_identical_matrices_zero_score(self):
        """Identical matrices should have zero drift score."""
        matrix = Matrix(source="test")
        matrix.add_activity("A")
        matrix.add_activity("B")
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT, existential=ExistentialDependency.IMPLICATION
            ),
        )

        score = compute_drift_score(matrix, matrix)

        assert score.cell_change_count == 0
        assert score.total_distance == 0.0
        assert score.activity_coverage == 0.0

    def test_temporal_change_detected(self):
        """Temporal dependency change should increase score."""
        matrix1 = Matrix(source="before")
        matrix1.add_activity("A")
        matrix1.add_activity("B")
        matrix1.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT, existential=ExistentialDependency.IMPLICATION
            ),
        )

        matrix2 = Matrix(source="after")
        matrix2.add_activity("A")
        matrix2.add_activity("B")
        matrix2.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.EVENTUAL,  # Changed!
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        score = compute_drift_score(matrix1, matrix2)

        assert score.cell_change_count >= 1
        assert score.temporal_distance > 0
        assert "A" in score.affected_activities
        assert "B" in score.affected_activities

    def test_existential_change_detected(self):
        """Existential dependency change should increase score."""
        matrix1 = Matrix(source="before")
        matrix1.add_activity("A")
        matrix1.add_activity("B")
        matrix1.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT, existential=ExistentialDependency.IMPLICATION
            ),
        )

        matrix2 = Matrix(source="after")
        matrix2.add_activity("A")
        matrix2.add_activity("B")
        matrix2.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.INDEPENDENCE,  # Changed!
            ),
        )

        score = compute_drift_score(matrix1, matrix2)

        assert score.cell_change_count >= 1
        assert score.existential_distance > 0


class TestARMDriftDetector:
    """Test ARMDriftDetector class."""

    def test_detector_initialization(self):
        """Test detector can be instantiated."""
        detector = ARMDriftDetector()
        assert detector.window_size == 100
        assert detector.threshold == 0.1

        detector = ARMDriftDetector(window_size=50, threshold=0.2)
        assert detector.window_size == 50
        assert detector.threshold == 0.2

    def test_empty_traces_handled(self):
        """Empty trace list should return empty result."""
        detector = ARMDriftDetector(window_size=10)
        result = detector.detect([])

        assert result.drift_indices == []
        assert result.drift_scores == []

    def test_insufficient_traces_handled(self):
        """Fewer traces than 2x window should return empty result."""
        from armature.discovery.models import Trace, Event

        # Create 15 traces (less than 2x10 window)
        traces = []
        for i in range(15):
            trace = Trace(
                case_id=f"trace_{i}",
                events=[
                    Event(activity="A", timestamp=None),
                    Event(activity="B", timestamp=None),
                ],
            )
            traces.append(trace)

        detector = ARMDriftDetector(window_size=10)
        result = detector.detect(traces)

        assert result.drift_indices == []


class TestARMDriftIntegration:
    """Integration tests with real discovery."""

    def test_stable_log_no_drift(self):
        """Log with consistent pattern should have low drift scores."""
        from armature.discovery.models import Trace, Event

        # Create 50 identical traces: A -> B -> C
        traces = []
        for i in range(50):
            trace = Trace(
                case_id=f"trace_{i}",
                events=[
                    Event(activity="A", timestamp=None),
                    Event(activity="B", timestamp=None),
                    Event(activity="C", timestamp=None),
                ],
            )
            traces.append(trace)

        detector = ARMDriftDetector(window_size=10, threshold=0.5)
        result = detector.detect(traces)

        # Stable process should have low/no drift
        # Scores should be low for identical patterns
        for score in result.drift_scores:
            assert score < 0.5, f"Unexpected high drift score {score} in stable log"

    def test_detects_order_swap_drift(self):
        """Detector should identify drift when activity order changes."""
        from armature.discovery.models import Trace, Event

        traces = []

        # Window 1: 100 traces with A -> B -> C order
        for i in range(100):
            trace = Trace(
                case_id=f"trace_before_{i}",
                events=[
                    Event(activity="A", timestamp=None),
                    Event(activity="B", timestamp=None),
                    Event(activity="C", timestamp=None),
                ],
            )
            traces.append(trace)

        # Window 2: 100 traces with A -> C -> B order (swap B and C)
        for i in range(100):
            trace = Trace(
                case_id=f"trace_after_{i}",
                events=[
                    Event(activity="A", timestamp=None),
                    Event(activity="C", timestamp=None),
                    Event(activity="B", timestamp=None),
                ],
            )
            traces.append(trace)

        # Discover matrices and check drift score
        from armature.drift.arm_detector import _discover_from_traces

        window1 = traces[:100]
        window2 = traces[100:200]

        matrix1 = _discover_from_traces(window1, source="window1")
        matrix2 = _discover_from_traces(window2, source="window2")

        score = compute_drift_score(matrix1, matrix2)

        # Order swap should change temporal dependencies
        assert score.cell_change_count > 0, "Order swap should change cells"
        assert score.total_distance > 0, "Order swap should have distance > 0"
        assert (
            score.normalized_distance > 0.1
        ), f"Expected detectable drift, got {score.normalized_distance}"

        # Run full detector
        detector = ARMDriftDetector(window_size=100, threshold=0.05)
        result = detector.detect(traces)

        # Should detect drift at boundary (position 100)
        assert len(result.drift_indices) > 0, "Detector should find drift at boundary"
        assert 100 in result.drift_indices, f"Expected drift at 100, found {result.drift_indices}"

    def test_result_structure(self):
        """Verify ARMDriftResult has expected structure."""
        from armature.discovery.models import Trace, Event

        traces = []
        for i in range(30):
            trace = Trace(
                case_id=f"trace_{i}",
                events=[
                    Event(activity="A", timestamp=None),
                    Event(activity="B", timestamp=None),
                ],
            )
            traces.append(trace)

        detector = ARMDriftDetector(window_size=10)
        result = detector.detect(traces)

        assert hasattr(result, "drift_indices")
        assert hasattr(result, "drift_scores")
        assert hasattr(result, "detailed_scores")
        assert hasattr(result, "threshold")
        assert hasattr(result, "window_size")
        assert isinstance(result.drift_indices, list)
        assert isinstance(result.drift_scores, list)


# Test data paths for integration tests with actual files
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "Test Data"
DISCOVERY_LOGS = TEST_DATA_DIR / "Discovery"


class TestARMDriftWithRealData:
    """Tests using real XES files."""

    @pytest.mark.skipif(not DISCOVERY_LOGS.exists(), reason="Discovery test logs not found")
    def test_detect_from_discovery_log(self):
        """Test detection on discovery validation log."""
        # Use a small log from discovery test data
        log_files = list(DISCOVERY_LOGS.glob("*.xes"))
        if not log_files:
            pytest.skip("No XES files in Discovery test data")

        log_path = log_files[0]
        result = detect_drift_arm(str(log_path), window_size=5, threshold=0.1)

        # Just verify it runs without error and returns valid structure
        assert result is not None
        assert isinstance(result.drift_indices, list)
