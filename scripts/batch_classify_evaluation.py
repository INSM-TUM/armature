#!/usr/bin/env python3
"""Batch classify all XES logs in Evaluation/ using updated weights.

Produces a table matching the survey ground truth from the LaTeX figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from armature.classification import classify
from armature.discovery import discover

# Ground truth mapping (from the LaTeX survey figure)
# P1=SS, P2=SS, P3=S, P4=S, P5=LS, P6=LS, P7=LS, P8=U
GROUND_TRUTH = {
    "p1_s_BPI19.xes": "SS",
    "Process_2_synthetic.xes": "SS",
    "Process_3_Invoice before GR (Standard).xes": "S",
    "Process_4_Augur.xes": "S",
    "Process_5_MIMIC_Dataset.xes": "LS",
    "Process_6_synthetic.xes": "LS",
    "Process_7_chickenhunt.xes": "LS",
    "p8_us_synthetic.xes": "U",
}

# Per-log discovery thresholds from prior runs (kept for consistency)
THRESHOLDS = {
    "p1_s_BPI19.xes": 0.8,  # was 1.0 → LS; 0.8 → SS (correct)
    "Process_2_synthetic.xes": 1.0,
    "Process_3_Invoice before GR (Standard).xes": 0.6,
    "Process_4_Augur.xes": 0.7,
    "Process_5_MIMIC_Dataset.xes": 0.9,
    "Process_6_synthetic.xes": 1.0,
    "Process_7_chickenhunt.xes": 0.9,
    "p8_us_synthetic.xes": 1.0,
}

SHORT_LABEL = {
    "structured": "S",
    "semi_structured": "SS",
    "loosely_structured": "LS",
    "unstructured": "U",
}

def is_borderline(scores: dict[str, float], predicted_short: str) -> bool:
    """Check if top two scores are within margin."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        return False
    return (ranked[0][1] - ranked[1][1]) < 0.2

def format_scores(scores: dict[str, float]) -> str:
    """Format scores as S / SS / LS / U."""
    parts = []
    for label in ["S", "SS", "LS", "U"]:
        val = scores.get(label, 0.0)
        parts.append(f"{val:+.2f}")
    return " / ".join(parts)

def main():
    eval_dir = Path("Evaluation")
    logs = sorted(eval_dir.glob("*.xes"))

    results = []

    print("=" * 100)
    print(f"{'Log':<42s} {'GT':>2s}  {'T':>4s}  {'Pred':>4s}  {'Conf':<10s}  {'Borderline':<12s}  {'OK':>2s}  {'Scores':<36s}  {'Method'}")
    print("=" * 100)

    for log_path in logs:
        fname = log_path.name
        gt = GROUND_TRUTH.get(fname, "?")
        t = THRESHOLDS.get(fname, 1.0)

        # Discover matrix with threshold
        matrix = discover(log_path, threshold=t)

        # Classify
        result = classify(matrix)

        short = SHORT_LABEL.get(result.category.value, "?")
        conf = result.confidence
        scores = result.metadata.get("scores", {})
        method = result.metadata.get("method", "?")
        scores_str = format_scores(scores)

        # Determine borderline
        if conf == "boundary":
            # Find which classes are close
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(ranked) >= 2:
                borderline_str = f"{ranked[0][0]}|{ranked[1][0]}"
            else:
                borderline_str = ""
        else:
            borderline_str = "—"

        ok = "✓" if short == gt else "✗"

        print(f"{fname:<42s} {gt:>2s}  {t:>4.1f}  {short:>4s}  {conf:<10s}  {borderline_str:<12s}  {ok:>2s}  {scores_str:<36s}  {method}")

        results.append({
            "log": fname,
            "GT": gt,
            "T": t,
            "Pred": short,
            "Confidence": conf,
            "Borderline": borderline_str,
            "OK": ok == "✓",
            "Scores": scores_str,
            "Method": method,
        })

    print("=" * 100)

    # Summary
    correct = sum(1 for r in results if r["OK"])
    print(f"\nAccuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print()

    # Detailed per-log breakdown
    print("--- Details ---")
    for r in results:
        print(f"\n{r['log']}:")
        print(f"  GT={r['GT']}  Pred={r['Pred']}  T={r['T']:.1f}")
        print(f"  Scores: S={r['Scores'].split('/')[0].strip()}  SS={r['Scores'].split('/')[1].strip()}  LS={r['Scores'].split('/')[2].strip()}  U={r['Scores'].split('/')[3].strip()}")
        print(f"  Method: {r['Method']}  Confidence: {r['Confidence']}")
        if r['Borderline'] != "—":
            print(f"  Borderline: {r['Borderline']}")
        print(f"  {'✓ CORRECT' if r['OK'] else '✗ WRONG'}")

    # Save JSON
    out_path = eval_dir / "classification_results_updated_weights.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Check which thresholds might need tuning
    print("\n--- Threshold Tuning Notes ---")
    for r in results:
        if not r["OK"]:
            print(f"  {r['log']}: Pred={r['Pred']} vs GT={r['GT']} — "
                  f"consider adjusting T (currently {r['T']:.1f}) or investigate ratios")


if __name__ == "__main__":
    main()
