"""Tests for Bose S/N/A drift detection wrapper."""

import pytest
from pathlib import Path

from armature.drift import BoseDriftDetector, detect_drift_bose


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "Test Data" / "Concept-drift-characterization"
BOSE_TEST_LOGS = TEST_DATA_DIR / "Bose Test Log"
SYNTHETIC_LOGS = TEST_DATA_DIR / "Synthetic Data"


class TestBoseDriftDetector:
    """Test BoseDriftDetector class."""

    def test_detector_initialization(self):
        """Test detector can be instantiated with various parameters."""
        # Default parameters
        detector = BoseDriftDetector()
        assert detector.window_size == 100
        assert detector.measure == "j"
        assert detector.stat_test == "mu"

        # Custom parameters
        detector = BoseDriftDetector(window_size=50, measure="wc", stat_test="ks")
        assert detector.window_size == 50
        assert detector.measure == "wc"
        assert detector.stat_test == "ks"

    @pytest.mark.skipif(not BOSE_TEST_LOGS.exists(), reason="Bose test logs not found")
    def test_detect_on_bose_test_log_segment1(self):
        """Test detection on Bose test log segment 1."""
        log_path = BOSE_TEST_LOGS / "bose_sublog_segment_1_traces_0-1199.xes"
        if not log_path.exists():
            pytest.skip(f"Test log not found: {log_path}")

        result = detect_drift_bose(str(log_path), window_size=100)

        # Verify result structure
        assert hasattr(result, "drift_indices")
        assert hasattr(result, "p_values")
        assert hasattr(result, "method")
        assert isinstance(result.drift_indices, list)
        assert len(result.p_values) > 0

    @pytest.mark.skipif(not SYNTHETIC_LOGS.exists(), reason="Synthetic test logs not found")
    def test_detect_on_synthetic_structured(self):
        """Test detection on synthetic structured log."""
        log_path = SYNTHETIC_LOGS / "p01_structured_a.xes"
        if not log_path.exists():
            pytest.skip(f"Test log not found: {log_path}")

        result = detect_drift_bose(str(log_path), window_size=20)

        # Small log may not have detectable drifts, just verify it runs
        assert result.p_values is not None


class TestCausalFootprint:
    """Test causal footprint matrix construction."""

    def test_footprint_simple_log(self):
        """Test causal footprint on minimal log."""
        # Create minimal log with pm4py
        from pm4py.objects.log.obj import EventLog, Trace, Event

        log = EventLog()
        # Trace 1: A -> B -> C
        trace1 = Trace()
        trace1.append(Event({"concept:name": "A"}))
        trace1.append(Event({"concept:name": "B"}))
        trace1.append(Event({"concept:name": "C"}))
        log.append(trace1)
        # Trace 2: A -> B -> C (same pattern)
        trace2 = Trace()
        trace2.append(Event({"concept:name": "A"}))
        trace2.append(Event({"concept:name": "B"}))
        trace2.append(Event({"concept:name": "C"}))
        log.append(trace2)

        detector = BoseDriftDetector(window_size=1)
        result = detector.detect(log)

        # With only 2 identical traces, no drift should be detected
        # But the method should complete without error
        assert result.p_values is not None


class TestDriftDetectionMeasures:
    """Test different drift detection measures."""

    @pytest.mark.parametrize("measure", ["j", "wc"])
    @pytest.mark.parametrize("stat_test", ["mu", "ks"])
    def test_all_measure_combinations(self, measure, stat_test):
        """Test all combinations of measures and statistical tests."""
        from pm4py.objects.log.obj import EventLog, Trace, Event

        # Create log with 20 traces
        log = EventLog()
        for i in range(20):
            trace = Trace()
            trace.append(Event({"concept:name": "A"}))
            trace.append(Event({"concept:name": "B"}))
            log.append(trace)

        detector = BoseDriftDetector(window_size=5, measure=measure, stat_test=stat_test)
        result = detector.detect(log)

        assert result.method == f"{measure.upper()}_{stat_test.upper()}"
        assert result.p_values is not None
