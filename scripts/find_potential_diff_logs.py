"""Find logs where greedy vs LP matching might differ.

Looks for cases with multiple detections that could match the same actual drift.
"""

import pandas as pd
import ast
from pathlib import Path

results_file = Path("/home/choky/kerstin/armature/results_hybrid_full.csv")
df = pd.read_csv(results_file)

# Parse detected/actual columns
df["detected_parsed"] = df["Detected Changepoints"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else []
)
df["actual_parsed"] = df["Actual Changepoints for Log"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else []
)


def has_potential_greedy_issue(detected, actual, lag=200):
    """Check if greedy might produce suboptimal matching."""
    if len(detected) < 2 or len(actual) < 2:
        return False

    # Look for cases where multiple detected could match the same actual
    for a in actual:
        matches = [d for d in detected if abs(d - a) <= lag]
        if len(matches) >= 2:
            return True

    # Look for cases where detected could match multiple actuals
    for d in detected:
        matches = [a for a in actual if abs(d - a) <= lag]
        if len(matches) >= 2:
            return True

    return False


# Filter for potential issues
df["potential_issue"] = df.apply(
    lambda row: has_potential_greedy_issue(row["detected_parsed"], row["actual_parsed"]), axis=1
)

potential_issues = df[df["potential_issue"] & (df["Window Size"] == 200)]

print(f"Total logs analyzed: {len(df[df['Window Size'] == 200])}")
print(f"Logs with potential greedy issues: {len(potential_issues)}")
print(f"Percentage: {len(potential_issues) / len(df[df['Window Size'] == 200]) * 100:.1f}%")

if len(potential_issues) > 0:
    print("\nSample cases (first 5):")
    for idx, row in potential_issues.head(5).iterrows():
        print(f"\nLog: {row['Log']}")
        print(f"  Detected: {row['detected_parsed']}")
        print(f"  Actual:   {row['actual_parsed']}")
        print(f"  Greedy F1: {row['F1-Score']:.4f}")
