#!/usr/bin/env python3.12
"""
Run ARM-Hybrid detector with full explanations enabled.

Shows detailed ARM matrix differences for each detected drift point.

Usage:
    python3.12 scripts/run_arm_with_explanations.py \
        tests/fixtures/drift_logs/drift_01_existential.xes
"""

import argparse
import sys
from pathlib import Path

from armature.discovery.xes_parser import parse_xes
from armature.drift.hybrid_detector import HybridDriftDetector


def main():
    """Run ARM-Hybrid with full explanations."""
    parser = argparse.ArgumentParser(description="Run ARM-Hybrid with explanations")
    parser.add_argument("log_file", type=Path, help="XES log file to analyze")
    parser.add_argument("--window-size", type=int, default=25, help="Sliding window size (default: 25)")
    parser.add_argument("--prominence", type=float, default=1.0, help="Peak prominence threshold (default: 1.0)")
    parser.add_argument("--step-size", type=int, default=5, help="Window step size (default: 5)")
    parser.add_argument("--ground-truth", type=int, nargs="+", help="Ground truth changepoints for comparison")
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print("=" * 80)
    print("ARM-Hybrid Drift Detection with Explanations")
    print("=" * 80)
    print(f"\nLog: {log_path.name}")
    print(f"Window size: {args.window_size}")
    print(f"Prominence: {args.prominence}")
    print(f"Step size: {args.step_size}")
    if args.ground_truth:
        print(f"Ground truth: {args.ground_truth}")
    print()

    # Parse log
    print("Parsing XES log...")
    traces = parse_xes(str(log_path))
    print(f"Loaded {len(traces)} traces")

    # Create detector with explanations ENABLED
    print("\nRunning ARM-Hybrid detector...")
    detector = HybridDriftDetector(
        window_size=args.window_size,
        step_size=args.step_size,
        prominence=args.prominence,
        min_gap=args.window_size,
        explain=True,  # ← Enable explanations
    )

    result = detector.detect(traces)

    # Display results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)

    print(f"\nDetected changepoints: {result.drift_indices}")
    print(f"Count: {len(result.drift_indices)}")

    if args.ground_truth:
        tolerance = 5
        for detected in result.drift_indices:
            near_gt = any(abs(detected - gt) <= tolerance for gt in args.ground_truth)
            status = "✓" if near_gt else "✗"
            print(f"  {status} Position {detected} (GT: {args.ground_truth})")

    # Display explanations
    if result.explanations:
        print("\n" + "=" * 80)
        print("SEMANTIC EXPLANATIONS")
        print("=" * 80)

        for i, exp in enumerate(result.explanations):
            print(f"\n{'─'*80}")
            print(f"DRIFT #{i+1} at position {exp.position}")
            print(f"{'─'*80}")

            print(f"\nChi-squared score: {exp.chi2_score:.2f}")

            if exp.arm_score:
                print(f"ARM cell changes: {exp.arm_score.cell_change_count}")
                print(f"Temporal distance: {exp.arm_score.temporal_distance:.2f}")
                print(f"Existential distance: {exp.arm_score.existential_distance:.2f}")
                print(f"Affected activities: {len(exp.affected_activities)}")

            # Temporal changes
            if exp.temporal_changes:
                print("\n🔄 Temporal Relationship Changes:")
                for act1, act2, before, after in exp.temporal_changes:
                    print(f"  • ({act1}, {act2}): {before} → {after}")

                    # Add semantic interpretation
                    if before == "DIRECT" and after == "EVENTUAL":
                        print(f"    → Intermediate steps added between {act1} and {act2}")
                    elif before == "DIRECT" and after == "NO_ORDERING":
                        print("    → Sequential constraint removed")
                    elif before == "DIRECT" and after == "TRUE_EVENTUAL":
                        print("    → Activities no longer consecutive (always separated)")
                    elif before == "EVENTUAL" and after == "NO_ORDERING":
                        print("    → Ordering relationship lost completely")

            # Existential changes
            if exp.existential_changes:
                print("\n🔗 Existential Relationship Changes:")
                for act1, act2, before, after in exp.existential_changes:
                    print(f"  • ({act1}, {act2}): {before} → {after}")

                    # Add semantic interpretation
                    if before == "IMPLICATION" and after == "INDEPENDENCE":
                        print(f"    → {act1} no longer guarantees {act2} will occur")
                    elif before == "EQUIVALENCE" and after == "INDEPENDENCE":
                        print("    → Activities no longer occur together")
                    elif before == "EQUIVALENCE" and after == "XOR":
                        print("    → Changed from always together to mutually exclusive")
                    elif before == "XOR" and after == "OR":
                        print("    → Mutual exclusivity relaxed (both can occur now)")
                    elif before == "IMPLICATION" and after == "IMPLICATION_BACKWARD":
                        print("    → Dependency direction reversed")

            # Show affected activities
            if exp.affected_activities:
                print(f"\n📋 Affected activities ({len(exp.affected_activities)} total):")
                print(f"  {', '.join(sorted(exp.affected_activities))}")

            # Summary
            print("\n📊 Summary:")
            summary_lines = []
            if exp.temporal_changes:
                summary_lines.append(f"{len(exp.temporal_changes)} temporal changes")
            if exp.existential_changes:
                summary_lines.append(f"{len(exp.existential_changes)} existential changes")
            if exp.affected_activities:
                summary_lines.append(f"{len(exp.affected_activities)} activities affected")

            print(f"  {', '.join(summary_lines)}")

    else:
        print("\nNo explanations generated (no drifts detected or explain=False)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
