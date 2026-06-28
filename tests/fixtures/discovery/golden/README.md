# Golden Baseline Documentation

Generated: 2026-01-30
Discovery version: post-Phase-3.5-fixes

## Purpose

These YAML files are golden baselines for regression testing. They represent correct discovery algorithm output after Phases 3.2-3.5 bug fixes:

- Phase 3.2: Implication directionality, matrix symmetry, self-loop detection, direct/eventual classification, determinism checks
- Phase 3.3: Parallel path detection via BFS, backward dependency detection
- Phase 3.4: Self-loop temporal classification (EVENTUAL not DIRECT), self-loop existential IMPLICATION
- Phase 3.5: TRUE_EVENTUAL vs EVENTUAL logic (DFG edge conditional)

## Expected Patterns by Log

### Log 01: Negated Equivalence
- **Pattern:** Sequential process A→B→C→D or A→B→D (exclusive choice between C and D paths)
- **Key relationships:**
  - DIRECT temporal for consecutive transitions (a→b, b→c, c→d, b→d)
  - TRUE_EVENTUAL for reachable but never consecutive (a→c, a→d, b→d)
  - EQUIVALENCE existential showing activities occur together
  - Self-loops have NO_ORDERING + IMPLICATION_BACKWARD (never repeat)

### Log 02: Parallelism + Negated Equivalence
- **Pattern:** Activities can occur in parallel (any order) but some pairs never co-occur
- **Key relationships:**
  - INDEPENDENCE temporal showing concurrent execution
  - EQUIVALENCE existential for activities that co-occur
  - NAND existential for mutually exclusive activities

### Log 03: Negated Equivalence
- **Pattern:** Similar to Log 01, exclusive choice structure
- **Key relationships:**
  - DIRECT for consecutive transitions
  - NAND existential for mutually exclusive branches

### Log 04: Nested Structure + Parallelism + Negated Equivalence
- **Pattern:** Hierarchical process with parallel sub-processes and exclusive choices
- **Key relationships:**
  - Mix of DIRECT, EVENTUAL, and INDEPENDENCE temporal dependencies
  - EQUIVALENCE for co-occurring activities
  - NAND for mutually exclusive paths

### Log 05: Nested Structured + Parallelism + Negated Equivalence
- **Pattern:** Complex nested structure with parallel execution and exclusions
- **Key relationships:**
  - INDEPENDENCE for parallel activities
  - NAND for exclusive choices
  - Multiple levels of nesting reflected in TRUE_EVENTUAL relationships

### Log 06: Inclusive OR + Parallelism
- **Pattern:** Activities can occur in any order or combination (no exclusions)
- **Key relationships:**
  - OR temporal showing flexible ordering
  - EQUIVALENCE existential showing all activities can co-occur
  - No NAND dependencies (inclusive, not exclusive)

### Log 07: Non-Block Structure
- **Pattern:** Sequential process without blocking structure
- **Key relationships:**
  - DIRECT temporal for consecutive transitions
  - TRUE_EVENTUAL for reachable but non-consecutive
  - EQUIVALENCE existential throughout

### Log 08: Loop + Negated Equivalence + Parallelism + Skipping
- **Pattern:** Cyclic structure with optional activities and parallel execution
- **Key relationships:**
  - EVENTUAL_BACKWARD on diagonal (b→b, c→c) showing self-loops with repetition
  - IMPLICATION_BACKWARD existential for self-loops (activity can repeat)
  - INDEPENDENCE temporal for parallel activities (b↔c)
  - IMPLICATION existential for skipped activities (e only if others occur)

### Log 09: Negated Equivalence + Parallelism
- **Pattern:** Parallel activities with some mutual exclusions
- **Key relationships:**
  - INDEPENDENCE temporal for concurrent execution
  - NAND existential for mutually exclusive activities
  - EQUIVALENCE for activities that co-occur

### Log 10: Nested Structure + Negated Equivalence
- **Pattern:** Hierarchical process with exclusive choices at multiple levels
- **Key relationships:**
  - DIRECT for immediate transitions
  - TRUE_EVENTUAL for nested reachability
  - NAND for exclusive branches

### Log 11: NAND
- **Pattern:** Pure exclusive choice pattern (activities never co-occur)
- **Key relationships:**
  - NAND existential for all activity pairs
  - Temporal dependencies show execution order
  - No EQUIVALENCE (all activities mutually exclusive)

## Validation Workflow

1. Re-run discovery: `python3.12 -m pytest tests/validation/test_discovery_validation.py -v`
2. Golden files automatically loaded and compared in test_clean_log_discovery
3. Assertion failure indicates regression - discovery output changed
4. Review HTML report to diagnose which relationships changed

## Updating Golden Files

If discovery algorithm intentionally improves:

1. Document reason for change in commit message
2. Human verify new output correctness via HTML report
3. Run: `cp tests/fixtures/discovery/results/*.yaml tests/fixtures/discovery/golden/`
4. Update discovery version in YAML headers
5. Commit with message explaining improvement

## File Naming Convention

Files match the source XES log names from Test Data/Discovery/:
- `event_log_NN_patternName.yaml`
- Pattern names describe the primary relationships tested
- Underscores separate pattern components

## Version History

- 2026-01-30: Initial golden baselines created after Phase 3.5 completion
  - All temporal and existential classification bugs fixed
  - TRUE_EVENTUAL vs EVENTUAL distinction correct
  - Self-loop detection working for both temporal and existential
