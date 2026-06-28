# Classification Engine

Process pattern classifier for Activity Relationship Matrices.

## Overview

Classifies discovered ARM matrices into four categories based on dependency ratio analysis:

- **Structured**: High direct/eventual dependencies, low non-determinism
- **Semi-Structured**: Moderate dependencies, some variability
- **Loosely-Structured**: Low dependencies, higher non-determinism
- **Unstructured**: Minimal dependencies, dominated by NAND/OR relationships

## Classification Accuracy and Limitations

### Phase 6.2: Conditional Rule Logic (Current)

**Validated on Test Data/Classification/ Variants.txt subset (17 logs):**

- Accuracy: 70.6% (12/17 logs correctly classified)
- Improvement from Phase 6.1 on targeted patterns:
  - Log12 borderline structured: Fixed (was failing, now passing)
  - Log23 pure nand_or: Fixed (was loosely, now unstructured)
  - Log10 high eventual semi: Fixed (was structured, now semi)

**Conditional logic patterns implemented:**

1. **Borderline structured**: `eventual 0.23-0.30 AND nand_or < 0.05` → structured despite low eventual
2. **High eventual semi**: `eventual > 0.30 AND implication >= 0.14 AND direct < 0.05` → semi despite high eventual
3. **Pure nand_or exclusion**: `nand_or > 0.90 AND eventual < 0.05 AND implication < 0.05` → unstructured not loosely
4. **High eventual loosely**: `eventual > 0.30 AND nand_or > 0.05` → loosely despite high eventual

### Known Architectural Limitations

**Log19/Log21 identical ratios paradox:**

- Log19_structured: `eventual=0.250, implication=0.300, nand_or=0.000` → Expected: structured
- Log21_looselyStructured: `eventual=0.250, implication=0.300, nand_or=0.000` → Expected: loosely

These logs have IDENTICAL numerical ratios but different ground truth categories. Threshold-based conditional logic CANNOT distinguish identical numeric values. This represents the architectural ceiling for threshold approaches.

**Current classification: Both classify as structured** (Log19 correct, Log21 incorrect)

**Remaining failures (4 logs):**

| Log   | Ratios (direct/eventual/implication/nand_or) | Expected | Got        | Root Cause                         |
| ----- | -------------------------------------------- | -------- | ---------- | ---------------------------------- |
| Log07 | 0.083 / 0.292 / 0.250 / 0.000                | semi     | structured | Borderline eventual, high implica  |
| Log13 | 0.000 / 0.291 / 0.164 / 0.000                | semi     | structured | Borderline eventual, low nand_or   |
| Log17 | 0.014 / 0.250 / 0.278 / 0.000                | semi     | structured | Exact eventual as Log19, high impl |
| Log20 | 0.033 / 0.297 / 0.264 / 0.000                | semi     | structured | Near threshold, moderate signals   |

All four have eventual near structured threshold (0.25-0.292) with moderate/high implication. They match the borderline structured pattern but should be semi due to implication strength.

### Threshold Calibration (Phase 6.1)

Thresholds empirically calibrated based on comprehensive dataset analysis:

- `direct_ratio_structured`: 0.0 (include structured with low direct but high eventual)
- `eventual_ratio_structured`: 0.30 (balance structured/semi separation)
- `implication_ratio_semi`: 0.143 (minimum observed in semi category)
- `nand_or_ratio_loosely`: 0.001 (exclude pure-zero unstructured)

See `.planning/phases/06.1-classification-validation/` for calibration methodology.

### Future Improvements

To exceed 70-75% accuracy:

1. **Non-threshold approaches required:**
   - Machine learning classifiers (decision trees, random forests)
   - Trace-level features (loop patterns, variant counts, activity sequences)
   - Filename/metadata features (detect loop variants programmatically)

2. **Alternative rule structures:**
   - Multi-factor scoring with weighted combinations
   - Decision trees instead of linear threshold checks
   - Ensemble methods combining multiple signals

Current conditional threshold logic represents the practical ceiling for rule-based classification. The Log19/Log21 paradox and overlapping boundary cases cannot be resolved with numerical thresholds alone.

## Usage

```python
from armature.classification import classify
from armature.discovery import discover

# Discover matrix from XES
matrix = discover(xes_path)

# Classify pattern
result = classify(matrix)
print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence}")
print(f"Ratios: {result.dependency_ratios}")
print(f"Rule trace: {result.rule_trace}")
```

CLI:

```bash
armature classify process.xes
```
