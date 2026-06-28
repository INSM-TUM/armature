#!/usr/bin/env python3.12
"""
Compare all cdrift-evaluation detectors with ARM-Hybrid on a sample log.

Shows exact outputs from each detector to understand their differences:
- What they detect (changepoint indices)
- What they report (their output format)
- How they differ from ARM's semantic explanations

Usage:
    python3.12 scripts/compare_all_detectors.py /path/to/cdrift-evaluation/ \
        --log tests/fixtures/drift_logs/drift_01_existential.xes

Output:
    - Console: Side-by-side comparison of all detector outputs
    - comparison_results.csv: Detailed results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def format_output(result: Any, detector_name: str) -> dict[str, Any]:
    """Extract and format key information from detector result."""
    if not result or (isinstance(result, list) and len(result) == 0):
        return {
            "detector": detector_name,
            "status": "ERROR/NO_RESULT",
            "detected": [],
            "count": 0,
            "f1_score": 0.0,
            "details": "No result returned",
        }

    # Handle list wrapper
    if isinstance(result, list):
        result = result[0]

    detected = result.get("Detected Changepoints", [])

    output = {
        "detector": detector_name,
        "status": "SUCCESS",
        "detected": detected,
        "count": len(detected) if detected else 0,
        "f1_score": result.get("F1-Score", 0.0),
    }

    # Add detector-specific details
    details = {}

    # ARM-Hybrid specific
    if "explanations" in result:
        explanations = result["explanations"]
        if explanations and len(explanations) > 0:
            exp = explanations[0]
            details["temporal_changes"] = exp.get("temporal_changes", [])
            details["existential_changes"] = exp.get("existential_changes", [])
            details["affected_activities"] = list(exp.get("affected_activities", []))
            details["chi2_score"] = exp.get("chi2_score")

    # Bose specific
    if "J-measure score" in result:
        details["j_measure"] = result["J-measure score"]

    # Add parameters
    params = {}
    if "window_size" in result:
        params["window_size"] = result["window_size"]
    if "prominence" in result:
        params["prominence"] = result["prominence"]
    if "step_size" in result:
        params["step_size"] = result["step_size"]

    if params:
        details["parameters"] = params

    output["details"] = details

    return output


def print_detector_output(output: dict[str, Any], ground_truth: list[int]):
    """Pretty print a detector's output."""
    print(f"\n{'='*80}")
    print(f"DETECTOR: {output['detector']}")
    print(f"{'='*80}")

    print(f"Status: {output['status']}")
    print(f"Detected: {output['detected']}")
    print(f"Count: {output['count']}")
    print(f"F1-Score: {output['f1_score']:.3f}")

    # Check if detected near ground truth
    if output["detected"] and ground_truth:
        gt_str = f"GT={ground_truth}"
        tolerance = 5
        near_gt = any(any(abs(det - gt) <= tolerance for gt in ground_truth) for det in output["detected"])
        status = "✓ NEAR GT" if near_gt else "✗ MISSED GT"
        print(f"{status} (tolerance ±{tolerance} traces) | {gt_str}")

    # Print detector-specific details
    details = output.get("details", {})

    if isinstance(details, dict):
        # Parameters
        if "parameters" in details:
            print("\nParameters:")
            for k, v in details["parameters"].items():
                print(f"  {k}: {v}")

        # ARM explanations
        if "temporal_changes" in details or "existential_changes" in details:
            print("\n🔍 ARM SEMANTIC EXPLANATION:")

            if details.get("temporal_changes"):
                print("  Temporal changes:")
                for act1, act2, before, after in details["temporal_changes"]:
                    print(f"    ({act1}, {act2}): {before} → {after}")

            if details.get("existential_changes"):
                print("  Existential changes:")
                for act1, act2, before, after in details["existential_changes"]:
                    print(f"    ({act1}, {act2}): {before} → {after}")

            if details.get("affected_activities"):
                print(f"  Affected activities: {', '.join(details['affected_activities'])}")

            if details.get("chi2_score"):
                print(f"  Chi-squared score: {details['chi2_score']:.2f}")

        # Bose J-measure
        if "j_measure" in details:
            print(f"\nJ-measure score: {details['j_measure']}")


def main():
    """Run comparison of all detectors."""
    parser = argparse.ArgumentParser(description="Compare all drift detectors")
    parser.add_argument("cdrift_path", type=Path, help="Path to cdrift-evaluation directory")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("tests/fixtures/drift_logs/drift_01_existential.xes"),
        help="Log file to analyze",
    )
    parser.add_argument(
        "--ground-truth", type=int, nargs="+", default=[50], help="Ground truth changepoints (default: [50])"
    )
    parser.add_argument("-o", "--output", type=Path, default="comparison_results.csv", help="Output CSV file")
    args = parser.parse_args()

    # Validate paths
    cdrift_path = args.cdrift_path.resolve()
    if not cdrift_path.exists():
        print(f"Error: cdrift-evaluation not found at {cdrift_path}")
        sys.exit(1)

    log_path = args.log.resolve()
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    # Add to path
    sys.path.insert(0, str(cdrift_path))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import detector functions
    try:
        from testAll_reproducibility import (
            testArmHybrid,
            testBose,
            testEarthMover,
            testLCDD,
            testMartjushev,
        )
    except ImportError as e:
        print(f"Error: Could not import cdrift functions: {e}")
        sys.exit(1)

    # Configuration
    F1_LAG = 25
    WINDOW_SIZE = 25
    STEP_SIZE = 5
    PROMINENCE = 1.0

    print("\n" + "=" * 80)
    print("DRIFT DETECTOR COMPARISON")
    print("=" * 80)
    print(f"\nLog: {log_path.name}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"F1 Lag Window: ±{F1_LAG} traces")
    print("\n" + "=" * 80)

    # Define detectors
    detectors = {
        "ARM-Hybrid": lambda: testArmHybrid(
            str(log_path), WINDOW_SIZE, PROMINENCE, STEP_SIZE, F1_LAG, args.ground_truth, None, False
        ),
        "Bose J-measure": lambda: testBose(
            str(log_path),
            WINDOW_SIZE,
            STEP_SIZE,
            F1_LAG,
            args.ground_truth,
            do_j=True,
            do_wc=False,
            position=None,
            show_progress_bar=False,
        ),
        "Bose Window Count": lambda: testBose(
            str(log_path),
            WINDOW_SIZE,
            STEP_SIZE,
            F1_LAG,
            args.ground_truth,
            do_j=False,
            do_wc=True,
            position=None,
            show_progress_bar=False,
        ),
        "Martjushev ADWIN J": lambda: testMartjushev(
            str(log_path),
            WINDOW_SIZE,
            F1_LAG,
            args.ground_truth,
            do_j=True,
            do_wc=False,
            position=None,
            show_progress_bar=False,
        ),
        "Martjushev ADWIN WC": lambda: testMartjushev(
            str(log_path),
            WINDOW_SIZE,
            F1_LAG,
            args.ground_truth,
            do_j=False,
            do_wc=True,
            position=None,
            show_progress_bar=False,
        ),
        "Earth Mover's Distance": lambda: testEarthMover(
            str(log_path), WINDOW_SIZE, STEP_SIZE, F1_LAG, args.ground_truth, None, False
        ),
        "LCDD": lambda: testLCDD(
            str(log_path), [(WINDOW_SIZE, WINDOW_SIZE)], 0.05, F1_LAG, args.ground_truth, None, False
        ),
    }

    all_outputs = []

    # Run each detector
    for detector_name, detector_func in detectors.items():
        print(f"\n{'─'*80}")
        print(f"Running: {detector_name}...")
        print(f"{'─'*80}")

        try:
            result = detector_func()
            output = format_output(result, detector_name)
            all_outputs.append(output)
            print_detector_output(output, args.ground_truth)

        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            all_outputs.append(
                {
                    "detector": detector_name,
                    "status": "ERROR",
                    "detected": [],
                    "count": 0,
                    "f1_score": 0.0,
                    "details": {"error": error_msg},
                }
            )

    # Summary comparison
    print("\n\n" + "=" * 80)
    print("SUMMARY: DETECTION COMPARISON")
    print("=" * 80)

    summary_data = []
    for output in all_outputs:
        summary_data.append(
            {
                "Detector": output["detector"],
                "Status": output["status"],
                "Detected Count": output["count"],
                "Detected CPs": str(output["detected"]),
                "F1-Score": f"{output['f1_score']:.3f}",
            }
        )

    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))

    # Find ARM-Hybrid output
    arm_output = next((o for o in all_outputs if o["detector"] == "ARM-Hybrid"), None)

    if arm_output and arm_output["status"] == "SUCCESS":
        print("\n\n" + "=" * 80)
        print("KEY INSIGHT: Why ARM is Different")
        print("=" * 80)

        details = arm_output.get("details", {})

        print("\nMost detectors only report WHERE drift occurred:")
        print("  → List of changepoint indices")
        print("  → Statistical scores (p-values, distances)")

        print("\nARM-Hybrid reports both WHERE and WHAT:")
        print("  → Changepoint indices (like others)")

        if details.get("temporal_changes") or details.get("existential_changes"):
            print("  → PLUS semantic explanation:")

            if details.get("temporal_changes"):
                print("\n    Temporal relationship changes:")
                for act1, act2, before, after in details["temporal_changes"][:3]:
                    print(f"      • ({act1}, {act2}): {before} → {after}")

                    # Add interpretation
                    if before == "DIRECT" and after == "EVENTUAL":
                        print("        Meaning: Intermediate steps added between activities")
                    elif before == "DIRECT" and after == "NO_ORDERING":
                        print("        Meaning: Sequential constraint removed")

            if details.get("existential_changes"):
                print("\n    Existential relationship changes:")
                for act1, act2, before, after in details["existential_changes"][:3]:
                    print(f"      • ({act1}, {act2}): {before} → {after}")

                    # Add interpretation
                    if before == "IMPLICATION" and after == "INDEPENDENCE":
                        print("        Meaning: Dependency lost (A no longer guarantees B)")
                    elif before == "EQUIVALENCE" and after == "XOR":
                        print("        Meaning: Always together → mutually exclusive")

    # Save results
    results_list = []
    for output in all_outputs:
        row = {
            "Detector": output["detector"],
            "Status": output["status"],
            "Detected": json.dumps(output["detected"]),
            "Count": output["count"],
            "F1-Score": output["f1_score"],
            "Details": json.dumps(output.get("details", {})),
        }
        results_list.append(row)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(args.output, index=False)

    print(f"\n\nDetailed results saved to: {args.output}")
    print("\nTo view ARM explanations in detail, check the 'Details' column (JSON format)")


if __name__ == "__main__":
    main()
