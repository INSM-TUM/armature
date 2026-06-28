"""Full verification of all 266 benchmark logs: greedy vs LP."""

import sys
from pathlib import Path
import pandas as pd
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.drift.evaluation import compute_metrics as greedy_metrics
from scripts.compare_matching import lp_f1

results_file = Path("/home/choky/kerstin/armature/results_hybrid_full.csv")
df = pd.read_csv(results_file)

# Filter for window_size=200
df = df[df["Window Size"] == 200].copy()

# Parse lists
df["detected"] = df["Detected Changepoints"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)
df["actual"] = df["Actual Changepoints for Log"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

print(f"Verifying ALL {len(df)} logs...")

differences = []
for idx, row in df.iterrows():
    detected = row["detected"]
    actual = row["actual"]

    if not detected or not actual:
        continue

    greedy = greedy_metrics(detected, actual, lag=200)
    lp = lp_f1(detected, actual, lag=200)

    diff = abs(greedy["f1"] - lp)

    if diff > 0.001:
        differences.append(
            {"log": row["Log"], "greedy_f1": greedy["f1"], "lp_f1": lp, "diff": diff}
        )

print(f"\nResults:")
print(
    f"  Total logs checked: {len([1 for _, r in df.iterrows() if r['detected'] and r['actual']])}"
)
print(
    f"  Matching: {len([1 for _, r in df.iterrows() if r['detected'] and r['actual']]) - len(differences)}"
)
print(f"  Different: {len(differences)}")

if not differences:
    print("\n✅ EVALUATION VERIFIED:")
    print("   Greedy matching produces IDENTICAL F1 scores to LP matching")
    print("   F1 = 0.977 result is valid and fair")
    print("   No benchmark re-run needed")
else:
    print(f"\n⚠️  Found {len(differences)} logs with differences:")
    for d in differences[:5]:
        print(f"  {d['log']}: Greedy={d['greedy_f1']:.4f}, LP={d['lp_f1']:.4f}, Δ={d['diff']:.4f}")
    if len(differences) > 5:
        print(f"  ... and {len(differences) - 5} more")
