#!/usr/bin/env python3.12
"""Compare ARM and Bose benchmark results.

Usage:
    python3.12 scripts/compare_arm_bose.py arm_results.csv bose_results.csv -o comparison.png
    python3.12 scripts/compare_arm_bose.py arm_results.csv bose_results.csv --report
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare ARM vs Bose results")
    parser.add_argument("arm_csv", type=Path, help="ARM results CSV")
    parser.add_argument("bose_csv", type=Path, help="Bose results CSV")
    parser.add_argument("-o", "--output", type=Path, default=Path("comparison.png"))
    parser.add_argument("--report", action="store_true", help="Print markdown report")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    from armature.drift.comparison import (
        compare_algorithms,
        generate_report_markdown,
        plot_comparison,
    )

    report = compare_algorithms(args.arm_csv, args.bose_csv)

    if args.report:
        md = generate_report_markdown(report)
        print(md)
    else:
        print(f"ARM Best F1: {report.arm_best_f1:.3f} ({report.arm_best_params})")
        print(f"Bose Best F1: {report.bose_best_f1:.3f} ({report.bose_best_params})")
        print(f"Improvement: {report.improvement_pct:+.1f}%")
        print(f"Wins: ARM={report.arm_wins_count}, Bose={report.bose_wins_count}, Ties={report.ties_count}")

    if not args.no_plot:
        plot_comparison(args.arm_csv, args.bose_csv, args.output)
        print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
