"""Concept drift detection for process mining.

Provides both Bose S/N/A and ARM-based drift detection methods.
"""
from armature.drift.arm_detector import ARMDriftDetector, detect_drift_arm
from armature.drift.metrics import compute_drift_score, DriftScore
from armature.drift.evaluation import (
    compute_metrics,
    bipartite_match_changepoints,
    compute_average_lag,
)
from armature.drift.cdrift_adapter import (
    CdriftDataset,
    extract_ground_truth,
    list_benchmark_logs,
)

__all__ = [
    "ARMDriftDetector",
    "detect_drift_arm",
    "compute_drift_score",
    "DriftScore",
    "compute_metrics",
    "bipartite_match_changepoints",
    "compute_average_lag",
    "CdriftDataset",
    "extract_ground_truth",
    "list_benchmark_logs",
]

try:
    from armature.drift.visualization import (
        DetectionResult,
        ComparisonResult,
        plot_drift_comparison,
        generate_comparison_report,
    )
    __all__.extend([
        "DetectionResult",
        "ComparisonResult",
        "plot_drift_comparison",
        "generate_comparison_report",
    ])
except ImportError:
    # visualization requires matplotlib - optional dependency
    pass

try:
    from armature.drift.bose_wrapper import BoseDriftDetector, detect_drift_bose
    __all__.extend(["BoseDriftDetector", "detect_drift_bose"])
except ImportError:
    # bose_wrapper not yet implemented - will be added in later plan
    pass
