# Classification Test Limitations

## Current Status: 76.5% Accuracy (26/34 logs)

After ground truth relabeling and pure nand_or exclusion fix.

### Remaining Failing Logs (8/34)

**Semi-structured (2 failures):**

- `Log06_semiStructured.xes` - direct=0.048, eventual=0.333, implication=0.143, nand_or=0.000
  - Caught by structured (low direct but high eventual)
- `Log10_semiStructured.xes` - direct=0.024, eventual=0.357, implication=0.238, nand_or=0.000
  - Caught by structured (low direct but high eventual)

**Structured (1 failure):**

- `Log12_structured.xes` - direct=0.083, eventual=0.292, implication=0.194, nand_or=0.278
  - Falls to semi (eventual=0.292 < 0.30 threshold)

**Loosely-structured (5 failures):**

- `Log08_looselyStructured.xes` - direct=0.048, eventual=0.238, implication=0.238, nand_or=0.143
  - Caught by semi (high implication)
- `Log21_looselyStructured.xes` - direct=0.000, eventual=0.250, implication=0.300, nand_or=0.000
  - Caught by semi (very high implication)
- `Log24_looselyStructured.xes` - direct=0.000, eventual=0.000, implication=0.000, nand_or=0.667
  - Falls to unstructured (all ratios=0 except nand_or, fails direct_max check)
- `p03_looselyStructured.xes` - direct=0.013, eventual=0.390, implication=0.234, nand_or=0.052
  - Caught by structured (very high eventual)

### Root Causes

**1. Rule Precedence Issues** (5 logs)
Categories have overlapping metric ranges. Earlier categories catch later ones:

- Log06, Log10, p03: Semi/loosely with high eventual caught by structured
- Log08, Log21: Loosely with high implication caught by semi

**2. Borderline Threshold** (Log12)

- eventual=0.292 vs threshold=0.30 (0.008 difference)
- Lowering threshold breaks 3+ other structured tests

**3. Metric Pattern Mismatch** (Log24)

- Has nand_or=0.667 but all other ratios=0.000
- Fails loosely direct_max check, falls to unstructured

### Attempted Fixes (All Failed)

**Threshold Adjustments:**

- `nand_or_loosely`: 0.001 → 0.000: Broke Log09, created 9 failures
- `eventual_structured`: 0.30 → 0.29: Broke Log07/13/20, created 10 failures

**Exclusion Patterns:**

- Structured: `if direct < 0.10 and eventual > 0.30`: Broke 5 structured tests

### Conclusion

**76.5% is the practical ceiling for threshold-based classification.**

Remaining failures have fundamentally incompatible metric patterns:

- Categories overlap in metric space (no clean threshold separation)
- Fixing one log breaks 2-5 passing logs
- Rule precedence conflicts cannot be resolved without complex conditional logic

**Options:**

1. Accept 76.5% as threshold-rule limit
2. Relabel suspicious ground truth (some failures may be mislabeled)
3. Add ML classifier (user rejected)
4. Add complex conditional helpers (defeats simplicity goal)
