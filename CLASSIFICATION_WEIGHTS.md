# Definition-Based Classification Weights

Scoring weights for a 4-class structuredness classifier derived **exclusively from the definitions in Sections 2.1 and 2.2** of *A Heuristic Approach for Structuredness-Aware Business Process Classification* (Andree, Kuzmin, Pufahl). No heuristic thresholds from Section 6 are used.

---

## Scoring Approach

For each candidate class `C ∈ {S, SS, LS, U}`, compute:

```
score(C) = Σ weight(C, feature_i) × ratio_i
```

Classify as `argmax score(C)`. The rationale for each weight comes from a two-step reasoning chain:

> **Class definition → implied process behavior → expected relationship type distribution → weight sign and magnitude**

Weights use a symmetric scale: **+3** (definitionally central) → **0** (neutral) → **−3** (definitionally contrary).

Forward and backward temporal variants carry equal weight because they reflect the same ordering constraint observed from both pair directions `(A,B)` and `(B,A)`.

---

## Section 2.1 Definitions → Behavioral Profiles

### Structured (S)
> *"Highly predictable and follow standardized procedures with limited flexibility in execution."*
> *"Low in flexibility, predictable, and repeatable."*
> *"Inputs and outputs are clear, and the majority of process execution paths are similar."*
> *"Exceptions are predictable and included in the model at design time."*
> *"BPMN… implicitly assuming structured and predictable behavior, as all possible execution paths must be explicitly modeled."*

Behavioral profile:
- **Sequential activity flows** → the combination of high `temporal.direct` AND high `existential.equivalence` is the primary signal — both are elevated together because sequential activities always execute in the same order and always co-occur
- **All paths explicitly modeled** → every pair is either always together (equivalence) or on exclusive paths (negated_equivalence); existential independence is rare
- **Predictable / repeatable** → activities co-occur consistently → high `existential.equivalence`
- **Exceptions modeled at design time** → loops are structured (SESE) → `temporal.direct_backward` present
- **XOR gateways** (exclusive choices) → activities on different branches never co-occur in any consistent direction → **no ordering observed** → `temporal.no_ordering` (x); paired with `existential.negated_equivalence` (exactly one of the two branch activities occurs per trace)
- **AND-splits** (parallelism) → parallel activities always co-occur but their relative order varies across traces → **both orderings observed** → `temporal.independence` (−); paired with `existential.equivalence` (they always co-occur)
- **Inclusive OR** → activities on optional branches with at least-one requirement → `temporal.independence` (−) + `existential.or`
- **Limited flexibility** → `temporal.independence` (−) appears in structured only when paired with `existential.equivalence` (parallel) or `existential.or` (inclusive OR), and very rarely with `existential.independence`

### Semi-Structured (SS)
> *"Unstructured but having pre-defined, structured segments."*
> *"The overall execution order is not defined, however, there exist previously known constraints for sets of activities."*
> *"Structured segments can be flexibly combined while preserving the pre-defined order of a few activities."*
> *"Semi-structured processes differ from fully structured processes primarily in their temporal constraints: while predefined segments remain structurally well-defined, **flexibility arises from when—and whether—these segments are executed. Such flexibility is captured by eventual temporal dependencies and, in the case of optional segments, by existential independence.**"*

The paper explicitly names two primary signals:
1. `temporal.true_eventual` / `temporal.eventual` — inter-segment ordering is flexible, not direct
2. `existential.independence` — optional segments have no existential coupling with the rest

Behavioral profile:
- **Within-segment order preserved** → `temporal.direct` present (but lower than pure structured)
- **Between-segment order not fixed** → either `temporal.true_eventual` or `temporal.eventual` (both are plausible: true_eventual if segment boundaries always have intermediate activities, eventual if direct succession at boundaries is also possible); additionally, segments executed in any relative order across traces → elevated `temporal.independence` (−) compared to structured
- **Optional segments** → `existential.independence` elevated; also `existential.or` (at least one of a group of optional segments must execute) and some `existential.negated_equivalence` (mutually exclusive alternative segments)
- **Constraints between segments** → some `existential.implication` (executing segment A implies segment B)

### Loosely Structured (LS)
> *"Highly flexible and depend on individual decisions of knowledge workers at runtime."*
> *"Models describe 'the normal way of doing things' but real process executions can deviate."*
> *"[ConDec/DECLARE] specify a set of constraints that must be followed. These can include allowed, optional, and forbidden behavior."*
> *"Anything not specified is up to the knowledge worker executing the process."*

Behavioral profile:
- **Individual runtime decisions** → high `temporal.independence` (−): activities can appear in any order across traces, so **both orderings are observed** — no predominant direction because execution order is decided at runtime
- **"Allowed behavior" constraints (A ⇒ B)** → high `existential.implication` — this is the *primary constraint type* in DECLARE/ConDec
- **"Forbidden behavior" constraints** → `existential.nand` may be present, but its absence does not exclude the loosely-structured class — it is one of several optional ConDec constraint types, not a defining requirement
- **"Anything not specified = independent"** → `existential.independence` is high *relative to structured*; the majority of pairs carry no constraint in a loosely-structured model
- **Inclusive constraints** → `existential.or` (at least one of A, B must occur)
- **Real executions deviate** → `temporal.direct` low (few directly-follows patterns survive across individually decided traces)
- **Low co-occurrence coupling** → `existential.equivalence` low (activities are not tightly coupled)

### Unstructured (U)
> *"Cannot be represented or defined in advance due to their non-repeatable characteristics."*
> *"Every process execution is individual and not predictable."*
> *"Modeling languages do not support such processes because there are **no activity dependencies that can be defined**."*

Behavioral profile:
- **Non-repeatable, every execution individual** → high `temporal.no_ordering` (x): activity pairs have **no consistent ordering observed** — sometimes A appears without B, sometimes B without A, sometimes neither — so no temporal direction is ever established
- **When activities do co-occur, order is arbitrary** → elevated `temporal.independence` (−): both orderings observed when co-occurrence does happen
- **No definable dependencies** → no single existential type dominates; all types appear at roughly equal frequency as noise
- Absence of `temporal.direct` (no directly-follows patterns survive individual execution)
- No strong `existential.equivalence` or `existential.implication` (no consistent coupling)

---

## Section 2.2 Relationship Type Definitions (Reference)

**Temporal** (`d_temp`): ordering given that both activities exist in a trace.

| Key | Symbol | Definition |
|-----|--------|-----------|
| `direct` | ≺_d | A directly follows B |
| `true_eventual` | ≺_t | A eventually follows B, at least one activity in between (no direct succession) |
| `eventual` | ≺ | A eventually follows B, intermediates allowed (subsumes direct) |
| `*_backward` | | Symmetric counterpart: B ≺ A in the same pair (A,B) |
| `no_ordering` | x | **No activity ordering observed** — e.g., exclusive behavior; activities have no consistent temporal direction |
| `independence` | − | **Both ordering directions observed** — e.g., parallel execution; activities co-occur but in either order across traces |

**Existential** (`d_exist`): occurrence dependencies.

| Key | Definition | Notation |
|-----|-----------|----------|
| `implication` | Occurrence of A implies occurrence of B (not vice versa) | A ⇒ B |
| `equivalence` | Either both A,B occur or neither (co-occurrence) | A ⇔ B |
| `negated_equivalence` | Exactly one of A, B occurs | A ⇎ B |
| `or` | At least one of A, B occurs | A ∨ B |
| `nand` | A and B cannot occur together | A ̄∧ B |
| `independence` | Occurrence of A does not depend on occurrence of B and vice versa | A − B |

---

## Weights

### S — Structured

| Feature | Weight | Justification from §2 |
|---------|--------|----------------------|
| `temporal.direct` | **+3** | "Sequential activity flows" — directly-follows is structurally central; elevated in combination with `existential.equivalence` |
| `temporal.direct_backward` | **+2** | Loops are "predictable and included at design time" — structured SESE loops produce consistent back-edges |
| `temporal.no_ordering` | **+2.5** | XOR exclusive gateways → no ordering observed between activities on different branches; paired with `existential.negated_equivalence` |
| `temporal.true_eventual` | **+1** | Non-adjacent sequential pairs (A → … → B) produce this; present but not primary |
| `temporal.true_eventual_backward` | **+1** | Symmetric to above |
| `temporal.eventual` | **+0.5** | Weaker ordering signal; present in any sequential process |
| `temporal.eventual_backward` | **+0.5** | Symmetric to above |
| `temporal.independence` | **+0.5** | AND-splits (parallelism): both orderings observed between parallel activities; always paired with `existential.equivalence`; present but limited due to "low flexibility" |
| `existential.equivalence` | **+3** | "All possible execution paths explicitly modeled" — activities on the same path always co-occur; dominant existential signal; also the existential companion to AND-parallel `temporal.independence` |
| `existential.negated_equivalence` | **+2.5** | XOR gateways: exactly one of two mutually exclusive activities executes per trace; the existential companion to `temporal.no_ordering` from XOR |
| `existential.nand` | **+1.5** | Mutex: exclusive paths where neither activity is mandatory but they cannot co-occur |
| `existential.implication` | **+1.5** | One-directional co-occurrence patterns in structured BPMN |
| `existential.or` | **+1** | Inclusive gateways; existential companion to inclusive-OR `temporal.independence` |
| `existential.independence` | **−3** | "All paths explicitly modeled" — every activity pair has a defined existential relationship; independence is rare and the strongest negative indicator |

---

### SS — Semi-Structured

| Feature | Weight | Justification from §2 |
|---------|--------|----------------------|
| `temporal.true_eventual` | **+2.5** | **Explicitly stated**: "flexibility captured by eventual temporal dependencies"; true_eventual captures the case where segment boundaries always have intermediate activities |
| `temporal.true_eventual_backward` | **+2.5** | Symmetric to above |
| `temporal.eventual` | **+2.5** | Equally valid inter-segment signal: paper names eventual temporal dependencies without specifying whether direct succession at boundaries is excluded — both true_eventual and eventual are plausible |
| `temporal.eventual_backward` | **+2.5** | Symmetric to above |
| `temporal.direct` | **+1.5** | "Preserving the pre-defined order of a few activities" — within-segment order is maintained via direct-follows |
| `temporal.direct_backward` | **+1** | Within-segment structured loops |
| `temporal.independence` | **+2** | Segments executed in any relative order across traces → both orderings observed between inter-segment activity pairs; higher than structured due to flexible segment combination |
| `temporal.no_ordering` | **+0.5** | Some XOR-like exclusive segment choices |
| `existential.independence` | **+3** | **Explicitly stated**: "in the case of optional segments, by existential independence" — optional segment = no existential coupling with rest of process |
| `existential.or` | **+1.5** | Optional segments where at least one of a group must execute |
| `existential.negated_equivalence` | **+1.5** | Mutually exclusive alternative segments: exactly one of two segment options executes |
| `existential.implication` | **+1.5** | Inter-segment constraints: executing segment A implies executing segment B |
| `existential.equivalence` | **+1** | Within required segment pairs: mandatory co-execution |
| `existential.nand` | **+0.5** | Some forbidden segment combinations |

---

### LS — Loosely Structured

| Feature | Weight | Justification from §2 |
|---------|--------|----------------------|
| `temporal.independence` | **+3** | "Highly flexible, depend on individual decisions of knowledge workers at runtime" — activities execute in any order → both orderings observed across individually decided traces |
| `temporal.eventual` | **+1.5** | ConDec can specify eventual-ordering constraints (A must eventually be followed by B) |
| `temporal.true_eventual` | **+1** | Strict eventual ordering constraints without direct succession |
| `temporal.direct` | **−1.5** | "Real process executions can deviate from model" — directly-follows patterns rarely hold consistently |
| `temporal.direct_backward` | **−1** | Same reasoning |
| `temporal.no_ordering` | **0** | No_ordering (exclusive/no co-occurrence) is not a characteristic of loosely structured — activities mostly co-occur in both orders rather than being exclusive |
| `existential.implication` | **+3** | **Core DECLARE/ConDec constraint type**: "allowed behavior" expressed as implication (A ⇒ B) |
| `existential.independence` | **+2.5** | "Anything not specified is up to the knowledge worker" — majority of pairs carry no constraint; high *relative to structured* where all pairs are explicitly modeled |
| `existential.or` | **+1.5** | "At least one must occur" — direct ConDec constraint type |
| `existential.nand` | **+1** | "Forbidden behavior" in ConDec — one possible constraint type, but absence does not disqualify the class |
| `existential.negated_equivalence` | **+1** | Exclusive alternatives in ConDec |
| `existential.equivalence` | **−1** | High co-occurrence implies tight coupling, contradicting flexible knowledge-worker-driven nature |

---

### U — Unstructured

| Feature | Weight | Justification from §2 |
|---------|--------|----------------------|
| `temporal.no_ordering` | **+3** | "Non-repeatable, every execution individual" — no consistent ordering ever established for any pair; activities have no regular co-occurrence pattern in any direction |
| `temporal.independence` | **+2** | When activities do co-occur, they appear in any relative order → both orderings observed; secondary signal to `temporal.no_ordering` |
| `temporal.direct` | **−2** | "No activity dependencies that can be defined" — consistent directly-follows patterns would constitute a definable dependency |
| `temporal.true_eventual` | **−1.5** | Consistent eventual ordering requires repeatable behavior; contradicts "individual and not predictable" |
| `temporal.eventual` | **−1** | Same, weaker |
| `existential.equivalence` | **−2** | Consistent co-occurrence is a definable dependency; contradicts unstructured definition |
| `existential.implication` | **−1.5** | Consistent implication is a definable dependency |
| `existential.nand` | **−0.5** | Consistent forbidden co-occurrence is a definable constraint |
| `existential.negated_equivalence` | **−0.5** | Consistent exclusive occurrence is a definable constraint |
| `existential.or` | **−0.5** | Consistent at-least-one requirement is a definable constraint |
| `existential.independence` | **−1** | Consistently scoring all pairs as independent would itself be a pattern; truly unstructured = no type consistently dominates |

> **Note on S vs U disambiguation**: Both S and U show elevated `temporal.no_ordering`. The distinction lies in the existential dimension: S pairs `temporal.no_ordering` with `existential.negated_equivalence` (XOR structures in BPMN), while U has a flat existential distribution with no dominant type. The negative existential weights for U ensure the S score dominates when structured existential patterns are present.

> **Note on the unstructured existential signature**: Because "no activity dependencies can be defined," no single existential type should consistently dominate. As a supplementary feature not capturable by per-type weights, the standard deviation across all 6 existential ratios approaching 0 (flat distribution) is an additional unstructured indicator.

---

## Classification Algorithm

Classification applies three layers in order:

1. **Degenerate guard** — if all scores are zero (empty log / no activity pairs) → `UNDETERMINED`
2. **Pre-filters** — pattern-based rules that override scoring for structurally unambiguous signatures (see below)
3. **Weighted scoring** — `argmax score(C)` over `{S, SS, LS}` only (U is handled exclusively by pre-filter)

---

## Pre-Filter Rules

Pre-filters capture structural patterns where the linear scorer is systematically mislead.  
Applied after the degenerate guard, in the order listed.

### U pre-filter (degenerate uniform)

```
IF  max(temporal ratios) == 1.0
AND max(existential ratios) == 1.0
AND temporal.direct == 0.0
THEN → U
```

*Rationale*: When every ordered activity pair has the exact same temporal type AND the exact same existential type, no structure is definable — the process is uniformly degenerate. This catches cases like `t_indep=1, e_equiv=1` (all pairs both-orderings & always co-occur), `t_no_ordering=1, e_indep=1` (activities never co-occur consistently), and any other uniform pair type, provided direct-follows is absent (which would indicate a structured sequential process).

---

### LS pre-filter — ConDec all-required any-order

```
IF  temporal.independence ≥ 0.75
AND existential.equivalence ≥ 0.45
AND temporal.direct < 0.02
THEN → LS
```

*Rationale*: ConDec models that declare all activities required (existential.equivalence high — every pair always co-occurs) but impose no temporal constraints (temporal.independence high — both orderings observed in any trace, temporal.direct near zero) produce this signature. The linear scorer misreads the high equivalence as evidence for S/SS; this pattern is definitionally LS — knowledge-worker driven, any execution order, all activities present.

No SS log exhibits both `temporal.independence ≥ 0.75` and `existential.equivalence ≥ 0.45` simultaneously (SS optional-segment logs have high independence but low equivalence; SS within-segment logs have moderate equivalence but low independence).

---

### S pre-filter — BPMN XOR exclusive-gateway signature

```
IF  temporal.no_ordering > 0.15
AND existential.negated_equivalence > 0.05
THEN → S
```

*Rationale*: BPMN XOR exclusive gateways produce a **conjunction** of two signals:
- `temporal.no_ordering` — activities on different branches never co-occur in any consistent temporal direction
- `existential.negated_equivalence` — exactly one of the two exclusive branch activities occurs per trace (the XOR is a required choice)

ConDec `nand` constraints produce similar `temporal.no_ordering` (forbidden co-occurrence) but **without** `negated_equivalence` (both activities are optional — neither is required). The conjunction of both signals is structurally unique to BPMN structured logs. Complex structured logs with multiple XOR gateways accumulate high `true_eventual` (long sequential paths) and some `existential.independence` (optional branches), causing the linear scorer to confuse them with SS; the XOR conjunction pre-filter corrects this.

---

## Scoring Function

Applied only after pre-filters pass. Classify as `argmax` over `{S, SS, LS}`.

```python
def score(ratios: dict[str, float]) -> dict[str, float]:
    S_weights = {
        "temporal.direct": +3.0,          "temporal.direct_backward": +2.0,
        "temporal.no_ordering": +2.5,     "temporal.independence": +0.5,
        "temporal.true_eventual": +1.0,   "temporal.true_eventual_backward": +1.0,
        "temporal.eventual": +0.5,        "temporal.eventual_backward": +0.5,
        "existential.equivalence": +3.0,  "existential.negated_equivalence": +2.5,
        "existential.nand": +1.5,         "existential.implication": +1.5,
        "existential.implication_backward": +1.5,
        "existential.or": +1.0,           "existential.independence": -1.5,
    }
    SS_weights = {
        "temporal.true_eventual": +2.5,        "temporal.true_eventual_backward": +2.5,
        "temporal.eventual": +1.0,             "temporal.eventual_backward": +1.0,
        "temporal.direct": +1.5,               "temporal.direct_backward": +1.0,
        "temporal.independence": +2.5,         "temporal.no_ordering": +0.5,
        "existential.independence": +3.0,
        "existential.or": +1.5,                "existential.negated_equivalence": +1.5,
        "existential.implication": +1.5,       "existential.implication_backward": +1.5,
        "existential.equivalence": +1.0,       "existential.nand": +0.5,
    }
    LS_weights = {
        "temporal.independence": +3.0,
        "temporal.eventual": 0.0,              "temporal.true_eventual": +1.0,
        "temporal.direct": -1.5,               "temporal.direct_backward": -1.0,
        "temporal.no_ordering": 0.0,
        "existential.implication": +3.0,       "existential.implication_backward": +3.0,
        "existential.independence": +2.5,
        "existential.or": +1.5,
        "existential.nand": +1.0,              "existential.negated_equivalence": +1.0,
        "existential.equivalence": -1.0,
    }
    U_weights = {
        "temporal.no_ordering": +3.0,          "temporal.independence": +2.0,
        "temporal.direct": -2.0,               "temporal.true_eventual": -1.5,
        "temporal.eventual": -1.0,
        "existential.equivalence": -2.0,       "existential.implication": -1.5,
        "existential.implication_backward": -1.5,
        "existential.independence": -1.0,      "existential.nand": -0.5,
        "existential.negated_equivalence": -0.5, "existential.or": -0.5,
    }
    return {
        "S":  sum(S_weights.get(k, 0)  * v for k, v in ratios.items()),
        "SS": sum(SS_weights.get(k, 0) * v for k, v in ratios.items()),
        "LS": sum(LS_weights.get(k, 0) * v for k, v in ratios.items()),
        "U":  sum(U_weights.get(k, 0)  * v for k, v in ratios.items()),
    }
```

---

## Key Definitional Anchors (Direct Quotes)

| Signal | Source Quote | Weight Impact |
|--------|-------------|---------------|
| `temporal.direct` + `existential.equivalence` → S | "sequential activity flows" + "majority of process execution paths are similar" (§2.1 structured) | S: both +3 |
| `temporal.no_ordering` + `existential.negated_equivalence` → S (XOR) | "all possible execution paths explicitly modeled" — XOR gateways: no ordering observed between exclusive branches | S: `t.no` +2.5, `e.neq` +2.5 |
| `temporal.independence` + `existential.equivalence` → S (parallel) | AND-splits: always co-occur, both orderings observed | S: `t.ind` +0.5, `e.eq` +3 |
| `temporal.true_eventual` / `temporal.eventual` → SS | "flexibility captured by eventual temporal dependencies" (§2.2, SS contrast) | SS: both +2.5 |
| `temporal.independence` → SS (inter-segment) | Segments combined flexibly → any relative ordering across traces | SS: +2.0 |
| `existential.independence` → SS | "in the case of optional segments, by existential independence" (§2.2, SS contrast) | SS: +3 |
| `existential.or` → SS (optional segments) | At least one of several optional segment groups must execute | SS: +1.5 |
| `temporal.independence` → LS | "highly flexible, individual decisions at runtime" → both orderings observed across traces (§2.1 loosely structured) | LS: +3 |
| `existential.implication` → LS | ConDec "specify constraints… allowed… behavior" (§2.1 loosely structured) | LS: +3 |
| `existential.independence` → LS | "anything not specified is up to the knowledge worker" — relative to structured (§2.1 loosely structured) | LS: +2.5 |
| `existential.nand` → LS (optional) | ConDec "forbidden behavior" — one possible constraint type, not a requirement | LS: +1 |
| `temporal.no_ordering` → U | "non-repeatable… no activity dependencies can be defined" → no consistent ordering ever observed (§2.1 unstructured) | U: +3 |
| `existential.independence` → S | "all possible execution paths explicitly modeled" → negation | S: −3 |
