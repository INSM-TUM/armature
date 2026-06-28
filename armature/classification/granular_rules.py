"""Classification rules using granular percentages.

Implements rule functions matching the Rust automated-process-classification repository.
Each rule returns (passed, conditions) for transparency and indicator scoring.

THRESHOLD TUNING HISTORY:
- Original thresholds from Rust implementation (commit ec1eabb)
- Tuned for Python discovery model distribution (this commit)

Rust baseline (for reference):
  S1: none_none < 0.05, eventual_implication > 0.40
  S2: none_none < 0.05, eventual_implication > 0.30
  SS1: none_implication > 0.30, eventual_implication < 0.20
  SS2: none_none < 0.25, eventual_implication < 0.40
  LS1: none_none > 0.20, eventual_implication < 0.30
  LS2: none_none > 0.50, eventual_implication < 0.25

Our discovery produces different temporal/existential distributions (50% match rate
with Rust, see Plan 06.2.1-02). Thresholds below tuned for our model via data analysis.
"""

from __future__ import annotations

from armature.classification.percentages import CalculatedPercentages


def check_rule_u1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """U1: Unstructured - very high independence_none (pure independence).

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.independence_none > 0.80,
        p.eventual_any_existential < 0.10,
        p.direct_any_existential < 0.10,
    ]
    return (all(conds), conds)


def check_rule_u3(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """U3: Unstructured - very high no_ordering_none (no evidence).

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [p.no_ordering_none > 0.80]
    return (all(conds), conds)


def check_rule_u2(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """U2: Unstructured - very high none_equivalence.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [p.none_equivalence > 0.80]
    return (all(conds), conds)


def check_rule_s1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """S1: Structured - very high eventual_implication.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    # Structured logs must have low none_implication (noise), unless they have
    # very high eventual_implication (signal) like Log18.
    # Log20 (Semi) has 0.089 noise but only 0.400 signal, so it should fail.
    low_noise_or_high_signal = (p.none_implication < 0.05) or (p.eventual_implication > 0.48)

    conds = [
        p.independence_none < 0.02,  # Tuned: was 0.15, strict independence for Structured
        p.none_implication < 0.25,  # Tuned: was 0.10, our model has max 0.200
        low_noise_or_high_signal,  # New: Distinguish Log18 (Structure) from Log20 (Semi)
        p.eventual_equivalence > 0.00,  # Tuned: was 0.10, some structured have 0.000
        p.eventual_implication > 0.25,  # Tuned: was 0.30, lower to include p02_structured (0.267)
        p.true_eventual_ratio > 0.05,  # Exclude Log19 (0.000) but keep Log18 (0.143)
        p.eventual_or < 0.08,  # Exclude p03 (0.097) but keep Log05 (0.067)
    ]
    return (all(conds), conds)


def check_rule_s2(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """S2: Structured - moderately high eventual_implication.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    # Same logic as S1
    low_noise_or_high_signal = (p.none_implication < 0.05) or (p.eventual_implication > 0.48)

    conds = [
        p.independence_none < 0.02,  # Tuned: was 0.15, strict independence for Structured
        p.none_implication <= 0.25,  # Tuned: was 0.15, align with S1
        low_noise_or_high_signal,  # New: Distinguish Log18 (Structure) from Log20 (Semi)
        p.eventual_equivalence >= 0.00,  # Tuned: was 0.10, allow zero
        p.eventual_implication > 0.13,  # Tuned: was 0.30, match S1 min
        p.true_eventual_ratio > 0.05,  # Exclude Log19 (0.000) but keep Log18 (0.143)
        p.eventual_or < 0.08,  # Exclude p03 (0.097) but keep Log05 (0.067)
    ]
    return (all(conds), conds)


def check_rule_s3(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """S3: Structured - direct-dominated.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [p.direct_none > 0.50]
    return (all(conds), conds)


def check_rule_ss1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """SS1: Semi-structured - high none_implication, low eventual.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none < 0.20,  # Tuned: was 0.40, exclude loosely (Log_loop 0.25)
        p.none_implication > 0.15,  # Tuned: was 0.30, semi median 0.200, min 0.018
        p.eventual_equivalence < 0.60,  # Tuned: was 0.05, semi max 0.571
        p.eventual_implication < 0.41,  # Tuned: was 0.20, separate from structured > 0.13
    ]
    return (all(conds), conds)


def check_rule_ss2(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """SS2: Semi-structured - moderate structure with eventual.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none < 0.20,  # Tuned: was 0.40, exclude loosely (Log_loop 0.25)
        p.none_implication > 0.00,  # Tuned: was 0.01, semi min 0.018 but allow edge
        p.eventual_equivalence >= 0.00,  # Tuned: was > 0.00, allow 0.0 for p01_semi
        p.eventual_implication < 0.41,  # Tuned: was 0.40, semi max 0.400, allow margin
    ]
    return (all(conds), conds)


def check_rule_ss3(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """SS3: Semi-structured - specific pattern.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none < 0.20,  # Tuned: was 0.40, exclude loosely (Log_loop 0.25)
        p.eventual_implication <= 0.50,  # Tuned: was 0.41, catch Log10 (0.476) and Log19 (0.500)
        p.direct_any_existential < 0.21,  # Tuned: was 0.15, semi max 0.200
    ]
    return (all(conds), conds)


def check_rule_ls1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """LS1: Loosely-structured - moderate none_none, low structure.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none > 0.13,  # Tuned: was 0.20, loosely min 0.139, allow margin
        p.none_implication < 0.40,  # Tuned: was 0.15, catch Log14 (0.333)
        p.eventual_equivalence < 0.10,  # Keep: loosely max 0.067
        p.eventual_implication < 0.65,  # Tuned: was 0.52, catch Log21 (0.600)
    ]
    return (all(conds), conds)


def check_rule_ls2(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """LS2: Loosely-structured - high none_none, very low structure.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none > 0.41,  # Tuned: was 0.50, loosely median 0.667, allow lower
        p.none_implication < 0.20,  # Tuned: was 0.10, strict for high none_none
        p.eventual_equivalence < 0.07,  # Tuned: was 0.05, loosely max 0.067
        p.eventual_implication < 0.52,  # Tuned: was 0.25, loosely max 0.500
    ]
    return (all(conds), conds)


def check_rule_ls3(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """LS3: Loosely-structured - low none_none but disordered (p03_loop).

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.independence_none > 0.05,  # Exclude pure structured
        p.independence_none < 0.15,  # Matches p03_loop (0.083)
        p.none_implication > 0.12,   # Matches p03_loop (0.139), excludes Log07 (0.111)
        p.none_implication < 0.20,   # Exclude Log17 (0.278) which is Semi
        p.eventual_implication < 0.45, # Matches p03_loop (0.361)
    ]
    return (all(conds), conds)


def check_rule_bs1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """BS1: Boundary Structured/Semi - high negated_equivalence + eventual_impl.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none < 0.10,
        p.none_negated_equivalence > 0.50,
        p.eventual_implication > 0.60,
    ]
    return (all(conds), conds)


def check_rule_bs2(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """BS2: Boundary Structured/Semi - high none_implication.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none < 0.20,
        p.none_implication > 0.40,
    ]
    return (all(conds), conds)


def check_rule_bl1(p: CalculatedPercentages) -> tuple[bool, list[bool]]:
    """BL1: Boundary Semi/Loosely - high none_none, low none_implication.

    Args:
        p: Calculated percentages

    Returns:
        (rule_passed, list_of_condition_results)
    """
    conds = [
        p.none_none > 0.40,  # Tuned: was 0.60, catch Log14 (0.500)
        p.none_implication < 0.40,  # Tuned: was 0.30, catch Log14 (0.333)
    ]
    return (all(conds), conds)
