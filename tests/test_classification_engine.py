"""Tests for classification engine and rule evaluation."""

from __future__ import annotations

import pytest

from armature.classification.config import EPSILON, ThresholdConfig
from armature.classification.engine import ClassificationEngine
from armature.classification.result import CategoryEnum, ClassificationResult
from armature.classification.rules import (
    compare_with_tolerance,
    evaluate_loosely_structured_rules,
    evaluate_semi_structured_rules,
    evaluate_structured_rules,
    evaluate_unstructured_rules,
)
from armature.classification.trace import RuleOutcome, RuleTrace
from armature.core.dependencies import (
    DependencyCell,
    ExistentialDependency,
    TemporalDependency,
)
from armature.core.matrix import Matrix


class TestCompareWithTolerance:
    """Test epsilon tolerance in floating-point comparisons."""

    def test_compare_exact_match(self):
        """Exact match should pass."""
        assert compare_with_tolerance(0.75, 0.75, ">=")
        assert compare_with_tolerance(0.75, 0.75, "<=")

    def test_compare_within_epsilon(self):
        """Values within epsilon should pass boundary cases."""
        # Just below threshold - should pass with epsilon
        assert compare_with_tolerance(0.75 - EPSILON / 2, 0.75, ">=")
        # Just above threshold - should pass with epsilon
        assert compare_with_tolerance(0.75 + EPSILON / 2, 0.75, "<=")

    def test_compare_outside_epsilon(self):
        """Values outside epsilon should fail."""
        # Well below threshold
        assert not compare_with_tolerance(0.74, 0.75, ">=")
        # Well above threshold
        assert not compare_with_tolerance(0.76, 0.75, "<=")

    def test_compare_rounding_artifacts(self):
        """Floating-point rounding artifacts should not fail."""
        # Common rounding artifact: 0.7500000001
        value = 0.7500000001
        threshold = 0.75
        assert compare_with_tolerance(value, threshold, ">=")

    def test_compare_all_operators(self):
        """All operators should work correctly."""
        assert compare_with_tolerance(0.8, 0.75, ">=")
        assert compare_with_tolerance(0.7, 0.75, "<=")
        assert compare_with_tolerance(0.8, 0.75, ">")
        assert compare_with_tolerance(0.7, 0.75, "<")

    def test_compare_invalid_operator(self):
        """Invalid operator should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operator"):
            compare_with_tolerance(0.75, 0.75, "==")


class TestStructuredRules:
    """Test structured classification rules."""

    def test_structured_all_pass(self):
        """High direct and eventual ratios should pass."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"direct_ratio": 0.80, "eventual_ratio": 0.90}

        result = evaluate_structured_rules(ratios, config, trace)

        assert result is True
        assert len(trace.steps) == 2
        assert all(step.outcome == RuleOutcome.PASSED for step in trace.steps)

    def test_structured_direct_fails(self):
        """Low direct ratio should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        # Use value below calibrated threshold (0.071)
        ratios = {"direct_ratio": 0.05, "eventual_ratio": 0.90}

        result = evaluate_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[0].outcome == RuleOutcome.FAILED
        assert trace.steps[0].metric_name == "direct_ratio"

    def test_structured_eventual_fails(self):
        """Low eventual ratio should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        # Direct passes (0.80 > 0.071), but eventual fails (0.30 < 0.417)
        ratios = {"direct_ratio": 0.80, "eventual_ratio": 0.30}

        result = evaluate_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[1].outcome == RuleOutcome.FAILED
        assert trace.steps[1].metric_name == "eventual_ratio"


class TestSemiStructuredRules:
    """Test semi-structured classification rules."""

    def test_semi_all_pass(self):
        """High implication, moderate eventual, lower direct should pass."""
        config = ThresholdConfig()
        trace = RuleTrace()
        # implication >= 0.264, eventual >= 0.100, direct <= 0.100
        ratios = {"implication_ratio": 0.30, "eventual_ratio": 0.15, "direct_ratio": 0.08}

        result = evaluate_semi_structured_rules(ratios, config, trace)

        assert result is True
        assert len(trace.steps) == 3
        assert all(step.outcome == RuleOutcome.PASSED for step in trace.steps)

    def test_semi_implication_fails(self):
        """Low implication ratio should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"implication_ratio": 0.20, "eventual_ratio": 0.70, "direct_ratio": 0.60}

        result = evaluate_semi_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[0].outcome == RuleOutcome.FAILED

    def test_semi_direct_too_high(self):
        """Direct ratio above semi max should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"implication_ratio": 0.40, "eventual_ratio": 0.70, "direct_ratio": 0.80}

        result = evaluate_semi_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[2].outcome == RuleOutcome.FAILED


class TestLooselyStructuredRules:
    """Test loosely-structured classification rules."""

    def test_loosely_all_pass(self):
        """High nand/or ratio and low direct should pass."""
        config = ThresholdConfig()
        trace = RuleTrace()
        # nand_or >= 0.172, direct <= 0.048
        ratios = {"nand_or_ratio": 0.20, "direct_ratio": 0.03}

        result = evaluate_loosely_structured_rules(ratios, config, trace)

        assert result is True
        assert len(trace.steps) == 2
        assert all(step.outcome == RuleOutcome.PASSED for step in trace.steps)

    def test_loosely_nand_or_fails(self):
        """Low nand/or ratio should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        # nand_or below 0.172 threshold
        ratios = {"nand_or_ratio": 0.10, "direct_ratio": 0.03}

        result = evaluate_loosely_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[0].outcome == RuleOutcome.FAILED

    def test_loosely_direct_too_high(self):
        """Direct ratio above loosely max should fail."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"nand_or_ratio": 0.50, "direct_ratio": 0.70}

        result = evaluate_loosely_structured_rules(ratios, config, trace)

        assert result is False
        assert trace.steps[1].outcome == RuleOutcome.FAILED


class TestUnstructuredRules:
    """Test unstructured classification rules."""

    def test_unstructured_always_passes(self):
        """Unstructured should always return True (default category)."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"direct_ratio": 0.10, "eventual_ratio": 0.20}

        result = evaluate_unstructured_rules(ratios, config, trace)

        assert result is True
        assert len(trace.steps) == 1
        assert trace.steps[0].outcome == RuleOutcome.PASSED
        assert trace.steps[0].rule_name == "unstructured_default"


class TestRuleTraceRecording:
    """Test that rule evaluations are properly recorded in trace."""

    def test_trace_records_all_steps(self):
        """Every rule evaluation should appear in trace."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"direct_ratio": 0.80, "eventual_ratio": 0.90}

        evaluate_structured_rules(ratios, config, trace)

        assert len(trace.steps) == 2
        assert trace.steps[0].rule_name == "structured_direct_ratio"
        assert trace.steps[1].rule_name == "structured_eventual_ratio"

    def test_trace_includes_all_fields(self):
        """Each trace step should include all required fields."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"direct_ratio": 0.80, "eventual_ratio": 0.90}

        evaluate_structured_rules(ratios, config, trace)

        step = trace.steps[0]
        assert step.rule_name == "structured_direct_ratio"
        assert step.metric_name == "direct_ratio"
        assert step.computed_value == 0.80
        assert step.threshold == config.direct_ratio_structured
        assert step.operator == ">="
        assert step.outcome == RuleOutcome.PASSED

    def test_trace_serialization(self):
        """Trace should serialize to list of dicts."""
        config = ThresholdConfig()
        trace = RuleTrace()
        ratios = {"direct_ratio": 0.80, "eventual_ratio": 0.90}

        evaluate_structured_rules(ratios, config, trace)

        serialized = trace.to_list()
        assert isinstance(serialized, list)
        assert len(serialized) == 2
        assert all(isinstance(step, dict) for step in serialized)
        assert serialized[0]["rule_name"] == "structured_direct_ratio"
        assert serialized[0]["outcome"] == "passed"


class TestClassificationEngine:
    """Test ClassificationEngine classify() method."""

    def test_engine_classify_structured(self):
        """Matrix with high direct/eventual ratios classifies as structured."""
        # Create matrix with high direct and eventual ratios
        # Need: direct_ratio >= 0.75, eventual_ratio >= 0.85
        matrix = Matrix(activities=["A", "B", "C", "D", "E"])

        # Add mostly DIRECT and TRUE_EVENTUAL dependencies
        # 4 DIRECT out of 5 = 0.8, 5 eventual out of 5 = 1.0
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "B",
            "C",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "C",
            "D",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "D",
            "E",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "A",
            "E",
            DependencyCell(
                temporal=TemporalDependency.TRUE_EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        engine = ClassificationEngine()
        result = engine.classify(matrix)

        assert result.category == CategoryEnum.STRUCTURED
        assert result.activity_count == 5
        assert result.total_dependencies == 5
        assert len(result.rule_trace) > 0

    def test_engine_classify_semi_structured(self):
        """Matrix with high implication classifies as semi-structured."""
        # Semi-structured needs:
        # - implication_ratio >= 0.30
        # - eventual_ratio >= 0.60
        # - direct_ratio <= 0.74
        matrix = Matrix(activities=["A", "B", "C", "D", "E"])

        # Create 2 EVENTUAL + 3 TRUE_EVENTUAL = 5 total
        # direct_ratio = 0/5 = 0.0 (under 0.74)
        # eventual_ratio = 5/5 = 1.0 (over 0.60)
        # implication_ratio = 5/5 = 1.0 (over 0.30)
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.TRUE_EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "B",
            "C",
            DependencyCell(
                temporal=TemporalDependency.TRUE_EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "C",
            "D",
            DependencyCell(
                temporal=TemporalDependency.TRUE_EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "A",
            "D",
            DependencyCell(
                temporal=TemporalDependency.EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "B",
            "E",
            DependencyCell(
                temporal=TemporalDependency.EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        engine = ClassificationEngine()
        result = engine.classify(matrix)

        assert result.category == CategoryEnum.SEMI_STRUCTURED
        assert result.activity_count == 5
        assert result.total_dependencies == 5

    def test_engine_classify_loosely_structured(self):
        """Matrix with high NAND/OR classifies as loosely-structured."""
        matrix = Matrix(activities=["A", "B", "C"])

        # Add dependencies with high NAND/OR ratio
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.INDEPENDENCE,
                existential=ExistentialDependency.NAND,
            ),
        )
        matrix.set_cell(
            "B",
            "C",
            DependencyCell(
                temporal=TemporalDependency.INDEPENDENCE,
                existential=ExistentialDependency.OR,
            ),
        )
        matrix.set_cell(
            "A",
            "C",
            DependencyCell(
                temporal=TemporalDependency.INDEPENDENCE,
                existential=ExistentialDependency.NAND,
            ),
        )

        engine = ClassificationEngine()
        result = engine.classify(matrix)

        assert result.category == CategoryEnum.LOOSELY_STRUCTURED
        assert result.total_dependencies == 3

    def test_engine_classify_unstructured(self):
        """Matrix with no strong patterns classifies as unstructured."""
        matrix = Matrix(activities=["A", "B"])

        # Add minimal dependencies
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.INDEPENDENCE,
                existential=ExistentialDependency.INDEPENDENCE,
            ),
        )

        engine = ClassificationEngine()
        result = engine.classify(matrix)

        assert result.category == CategoryEnum.UNSTRUCTURED
        assert result.total_dependencies == 1

    def test_empty_matrix_handling(self):
        """Empty matrix with no dependencies raises ValueError."""
        matrix = Matrix(activities=["A", "B"])
        # No dependencies added

        engine = ClassificationEngine()
        with pytest.raises(ValueError, match="Cannot classify empty matrix"):
            engine.classify(matrix)

    def test_boundary_detection(self):
        """Matrix at threshold boundary gets confidence='boundary'."""
        # Create matrix with 4 cells, 3 DIRECT
        # direct_ratio = 3/4 = 0.75
        # Use custom config with threshold at 0.75 to trigger boundary
        matrix = Matrix(activities=["A", "B", "C", "D"])

        # 3 DIRECT cells
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "B",
            "C",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        matrix.set_cell(
            "C",
            "D",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )
        # 1 EVENTUAL cell
        matrix.set_cell(
            "A",
            "D",
            DependencyCell(
                temporal=TemporalDependency.EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        # Use custom config with threshold exactly at 0.75
        custom_config = ThresholdConfig(
            direct_ratio_structured=0.75,
            eventual_ratio_structured=0.85,
        )
        engine = ClassificationEngine(config=custom_config)
        result = engine.classify(matrix)

        # Should be flagged as boundary due to exact threshold match
        # direct_ratio = 3/4 = 0.75 (exactly at threshold)
        assert result.confidence == "boundary"

    def test_result_completeness(self):
        """Result includes all required transparency fields."""
        matrix = Matrix(activities=["A", "B"])
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        engine = ClassificationEngine()
        result = engine.classify(matrix)

        # Check all fields present
        assert result.category is not None
        assert result.confidence in ["exact", "boundary"]
        assert isinstance(result.dependency_counts, dict)
        assert isinstance(result.dependency_ratios, dict)
        assert isinstance(result.thresholds_applied, dict)
        assert isinstance(result.rule_trace, list)
        assert result.activity_count >= 0
        assert result.total_dependencies >= 0
        assert 0.0 <= result.density <= 1.0
        assert result.loop_count >= 0


class TestPublicAPI:
    """Test public classify() API."""

    def test_public_api_without_config(self):
        """Public classify() works without explicit config."""
        from armature.classification import classify

        # Create simple matrix
        matrix = Matrix(activities=["A", "B"])
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        result = classify(matrix)

        assert isinstance(result, ClassificationResult)
        assert result.category is not None
        assert len(result.rule_trace) > 0

    def test_public_api_with_config(self):
        """Public classify() accepts custom config."""
        from armature.classification import classify

        # Custom config with different thresholds
        config = ThresholdConfig(direct_ratio_structured=0.5)

        matrix = Matrix(activities=["A", "B"])
        matrix.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.DIRECT,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        result = classify(matrix, config)

        assert isinstance(result, ClassificationResult)
        assert result.thresholds_applied["direct_ratio_structured"] == 0.5

    def test_all_categories_classifiable(self):
        """All 4 categories can be classified."""
        from armature.classification import classify

        # Structured: high direct + eventual
        m_structured = Matrix(activities=["A", "B", "C", "D"])
        for i, (src, tgt) in enumerate([("A", "B"), ("B", "C"), ("C", "D")]):
            m_structured.set_cell(
                src,
                tgt,
                DependencyCell(
                    temporal=TemporalDependency.DIRECT,
                    existential=ExistentialDependency.IMPLICATION,
                ),
            )
        m_structured.set_cell(
            "A",
            "D",
            DependencyCell(
                temporal=TemporalDependency.TRUE_EVENTUAL,
                existential=ExistentialDependency.IMPLICATION,
            ),
        )

        # Semi-structured: high implication, moderate eventual
        m_semi = Matrix(activities=["A", "B", "C"])
        for src, tgt in [("A", "B"), ("B", "C"), ("A", "C")]:
            m_semi.set_cell(
                src,
                tgt,
                DependencyCell(
                    temporal=TemporalDependency.TRUE_EVENTUAL,
                    existential=ExistentialDependency.IMPLICATION,
                ),
            )

        # Loosely-structured: high NAND/OR
        m_loosely = Matrix(activities=["A", "B", "C"])
        for src, tgt in [("A", "B"), ("B", "C"), ("A", "C")]:
            m_loosely.set_cell(
                src,
                tgt,
                DependencyCell(
                    temporal=TemporalDependency.INDEPENDENCE,
                    existential=ExistentialDependency.NAND,
                ),
            )

        # Unstructured: minimal dependencies
        m_unstructured = Matrix(activities=["A", "B"])
        m_unstructured.set_cell(
            "A",
            "B",
            DependencyCell(
                temporal=TemporalDependency.INDEPENDENCE,
                existential=ExistentialDependency.INDEPENDENCE,
            ),
        )

        # Classify all
        r_structured = classify(m_structured)
        r_semi = classify(m_semi)
        r_loosely = classify(m_loosely)
        r_unstructured = classify(m_unstructured)

        # Verify categories
        assert r_structured.category == CategoryEnum.STRUCTURED
        assert r_semi.category == CategoryEnum.SEMI_STRUCTURED
        assert r_loosely.category == CategoryEnum.LOOSELY_STRUCTURED
        assert r_unstructured.category == CategoryEnum.UNSTRUCTURED
