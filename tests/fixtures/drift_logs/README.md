# Counter-Example Logs for ARM vs Bose Comparison

These synthetic XES logs demonstrate cases where ARM's richer dependency model
detects concept drifts that Bose's S/N/A approach misses or detects later.

## Log Summary

| Log | Drift Type | Drift Point | ARM Detects | Bose Detects |
|-----|------------|-------------|-------------|--------------|
| drift_01_existential.xes | IMPLICATION->INDEPENDENCE | 50 | Yes (50) | No |
| drift_02_temporal_directness.xes | DIRECT->EVENTUAL | 50 | Yes (50) | Unlikely |
| drift_03_combined.xes | Multiple changes | 50 | Yes (50) | Late (~60) |
| drift_04_subtle_implication.xes | EQUIVALENCE->XOR | 50 | Yes (50) | No |

## Detailed Descriptions

### drift_01_existential.xes - Existential Dependency Change

**Pattern:**
- Before (traces 0-49): A -> B -> C (every trace has all three activities)
- After (traces 50-99): A -> C (50%) or A -> B -> C (50%)

**Why ARM wins:**
- ARM detects IMPLICATION(A=>B) becoming INDEPENDENCE(A,B)
- Bose sees "A sometimes followed by B" both before and after
- S/N/A cannot distinguish co-occurrence patterns

### drift_02_temporal_directness.xes - Temporal Directness Change

**Pattern:**
- Before (traces 0-49): A -> B -> C (A directly followed by B)
- After (traces 50-99): A -> C -> B (C inserted between A and B)

**Why ARM wins:**
- ARM detects DIRECT(A,B) becoming EVENTUAL(A,B)
- Bose sees "A followed by B" in both cases (succession exists)
- S/N/A tracks succession existence, not directness

### drift_03_combined.xes - Multiple Dependency Changes

**Pattern:**
- Before (traces 0-49): A -> B -> C (linear sequence)
- After (traces 50-99): A -> B -> D (50%) or A -> C -> D (50%)

**Why ARM wins:**
- Multiple ARM dimensions change:
  - Temporal: DIRECT(A,B) and DIRECT(A,C) patterns shift
  - Existential: IMPLICATION becomes NAND (B,C mutually exclusive)
  - Structure: D added, C sometimes absent
- More signals = earlier/stronger detection
- Bose may detect late due to statistical window averaging

### drift_04_subtle_implication.xes - Subtle Implication Pattern

**Pattern:**
- Before (traces 0-49): A -> B -> C (A always with both B and C)
- After (traces 50-99): A -> B (50%) or A -> C (50%)

**Why ARM wins:**
- ARM detects shift from both B,C to exactly one (XOR)
- Bose sees "A sometimes followed by B, sometimes by C" in both
- Existential pattern change is invisible to succession analysis

## Usage

These logs are used in `tests/drift/test_comparative_suite.py` to programmatically
verify ARM's detection superiority. Each test:

1. Runs both ARM and Bose detectors on the log
2. Verifies ARM detects the drift
3. Verifies Bose either misses or detects later
4. Generates detailed comparison report

## Generation

Logs were generated using `armature.drift.log_generator`:

```python
from armature.drift.log_generator import generate_existential_drift_log
scenario = generate_existential_drift_log(Path("drift_01_existential.xes"))
```

## Updating Logs

If drift detection thresholds or logic change, regenerate logs:

```bash
python -c "
from pathlib import Path
from armature.drift.log_generator import *

output_dir = Path('tests/fixtures/drift_logs')
generate_existential_drift_log(output_dir / 'drift_01_existential.xes')
generate_temporal_directness_drift_log(output_dir / 'drift_02_temporal_directness.xes')
generate_combined_drift_log(output_dir / 'drift_03_combined.xes')
generate_subtle_implication_drift_log(output_dir / 'drift_04_subtle_implication.xes')
"
```
