"""Verify greedy vs LP matching on actual benchmark results.

Takes sample logs from benchmark and compares F1 scores.
"""

import sys
from pathlib import Path
import pandas as pd
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.drift.evaluation import compute_metrics as greedy_metrics
from scripts.compare_matching import lp_f1

results_file = Path("/home/choky/kerstin/armature/results_hybrid_full.csv")
df = pd.read_csv(results_file)

# Filter for window_size=200 (standard comparison)
df = df[df["Window Size"] == 200].copy()

# Parse lists
df["detected"] = df["Detected Changepoints"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)
df["actual"] = df["Actual Changepoints for Log"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

print("Verifying Greedy vs LP Matching on Benchmark Logs")
print("=" * 60)

# Sample 10 logs with non-empty detections
sample = df[(df["detected"].str.len() > 0) & (df["actual"].str.len() > 0)].head(10)

differences = []
for idx, row in sample.iterrows():
    detected = row["detected"]
    actual = row["actual"]
    lag = 200

    greedy = greedy_metrics(detected, actual, lag)
    lp = lp_f1(detected, actual, lag)

    diff = abs(greedy["f1"] - lp)

    if diff > 0.001:
        differences.append(
            {
                "log": row["Log"],
                "detected": detected,
                "actual": actual,
                "greedy_f1": greedy["f1"],
                "lp_f1": lp,
                "diff": diff,
            }
        )
        print(f"\n⚠️ DIFFERENCE FOUND:")
        print(f"Log: {row['Log']}")
        print(f"Detected: {detected}")
        print(f"Actual: {actual}")
        print(f"Greedy F1: {greedy['f1']:.4f}")
        print(f"LP F1: {lp:.4f}")
        print(f"Difference: {diff:.4f}")

if not differences:
    print("\n✓ ALL SAMPLED LOGS MATCH")
    print("Greedy and LP produce identical F1 scores on benchmark data")
    print(f"Tested {len(sample)} logs with detections")
else:
    print(f"\n⚠️ Found {len(differences)} logs with differences")
    print("LP matching may improve benchmark F1 score")
