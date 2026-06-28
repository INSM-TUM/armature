"""Tests for classification configuration and trace infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from armature.classification.config import ConfigLoader, ThresholdConfig
from armature.classification.result import CategoryEnum, ClassificationResult
from armature.classification.trace import RuleOutcome, RuleTrace, TraceStep


class TestThresholdConfig:
    """Test ThresholdConfig validation."""

    def test_config_defaults_valid(self):
        """Default config should be valid."""
        config = ThresholdConfig()

        # Calibrated from ground truth test data (2026-02-06)
        assert config.direct_ratio_structured == 0.071
        assert config.eventual_ratio_structured == 0.417
        assert config.direct_ratio_semi_max == 0.100
        assert config.eventual_ratio_semi_min == 0.100
        assert config.implication_ratio_semi == 0.264
        assert config.direct_ratio_loosely_max == 0.048
        assert config.nand_or_ratio_loosely == 0.172

    def test_config_invalid_range_fails(self):
        """Config with ratio > 1.0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ThresholdConfig(direct_ratio_structured=1.5)

        assert "less than or equal to 1" in str(exc_info.value)

    def test_config_negative_fails(self):
        """Config with negative ratio should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ThresholdConfig(eventual_ratio_structured=-0.1)

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_config_eventual_less_than_direct_fails(self):
        """Eventual ratio < direct ratio should fail (DIRECT ⊂ EVENTUAL)."""
        with pytest.raises(ValidationError) as exc_info:
            ThresholdConfig(
                direct_ratio_structured=0.80,
                eventual_ratio_structured=0.70,  # Less than direct
            )

        assert "DIRECT ⊂ EVENTUAL" in str(exc_info.value)


class TestConfigLoader:
    """Test ConfigLoader YAML loading."""

    def test_load_none_returns_defaults(self):
        """Loading with None path should return default config."""
        config = ConfigLoader.load()

        # Calibrated defaults
        assert config.direct_ratio_structured == 0.071
        assert config.eventual_ratio_structured == 0.417

    def test_load_nonexistent_file_raises(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load(Path("/nonexistent/config.yaml"))

    def test_load_valid_yaml(self, tmp_path):
        """Loading valid YAML should return configured values."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
direct_ratio_structured: 0.80
eventual_ratio_structured: 0.90
direct_ratio_semi_max: 0.70
eventual_ratio_semi_min: 0.65
implication_ratio_semi: 0.35
direct_ratio_loosely_max: 0.60
nand_or_ratio_loosely: 0.45
"""
        )

        config = ConfigLoader.load(config_path)

        assert config.direct_ratio_structured == 0.80
        assert config.eventual_ratio_structured == 0.90
        assert config.direct_ratio_semi_max == 0.70
        assert config.eventual_ratio_semi_min == 0.65
        assert config.implication_ratio_semi == 0.35
        assert config.direct_ratio_loosely_max == 0.60
        assert config.nand_or_ratio_loosely == 0.45

    def test_load_invalid_yaml_raises(self, tmp_path):
        """Loading invalid YAML should raise ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: {")

        with pytest.raises(ValueError) as exc_info:
            ConfigLoader.load(config_path)

        assert "Invalid YAML" in str(exc_info.value)

    def test_load_empty_file_raises(self, tmp_path):
        """Loading empty file should raise ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        with pytest.raises(ValueError) as exc_info:
            ConfigLoader.load(config_path)

        assert "empty" in str(exc_info.value)

    def test_load_invalid_values_raises(self, tmp_path):
        """Loading file with out-of-range values should raise ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("direct_ratio_structured: 1.5")

        with pytest.raises(ValueError):
            ConfigLoader.load(config_path)


class TestRuleTrace:
    """Test RuleTrace and TraceStep."""

    def test_trace_immutability(self):
        """TraceStep should be immutable (frozen dataclass)."""
        step = TraceStep(
            rule_name="test_rule",
            metric_name="test_metric",
            computed_value=0.8,
            threshold=0.75,
            operator=">=",
            outcome=RuleOutcome.PASSED,
        )

        # Attempt to modify should fail
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.11+
            step.computed_value = 0.9

    def test_trace_serialization(self):
        """to_list() should serialize all steps to dicts."""
        trace = RuleTrace()

        trace.record_step(
            rule_name="rule1",
            metric_name="direct_ratio",
            computed_value=0.8,
            threshold=0.75,
            operator=">=",
            outcome=RuleOutcome.PASSED,
        )

        trace.record_step(
            rule_name="rule2",
            metric_name="eventual_ratio",
            computed_value=0.7,
            threshold=0.85,
            operator=">=",
            outcome=RuleOutcome.FAILED,
        )

        result = trace.to_list()

        assert len(result) == 2
        assert result[0]["rule_name"] == "rule1"
        assert result[0]["metric_name"] == "direct_ratio"
        assert result[0]["computed_value"] == 0.8
        assert result[0]["threshold"] == 0.75
        assert result[0]["operator"] == ">="
        assert result[0]["outcome"] == "passed"

        assert result[1]["rule_name"] == "rule2"
        assert result[1]["outcome"] == "failed"

    def test_trace_accumulation(self):
        """RuleTrace should accumulate steps."""
        trace = RuleTrace()
        assert len(trace.steps) == 0

        trace.record_step("r1", "m1", 0.5, 0.6, ">=", RuleOutcome.FAILED)
        assert len(trace.steps) == 1

        trace.record_step("r2", "m2", 0.8, 0.7, ">=", RuleOutcome.PASSED)
        assert len(trace.steps) == 2

        # Steps should be in order
        assert trace.steps[0].rule_name == "r1"
        assert trace.steps[1].rule_name == "r2"


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_result_json_roundtrip(self):
        """JSON serialization should preserve all fields."""
        result = ClassificationResult(
            category=CategoryEnum.STRUCTURED,
            confidence="exact",
            dependency_counts={
                "DIRECT": 10,
                "EVENTUAL": 12,
                "NAND": 2,
                "OR": 1,
                "IMPLICATION": 5,
            },
            dependency_ratios={
                "direct_ratio": 0.80,
                "eventual_ratio": 0.90,
            },
            thresholds_applied={
                "direct_ratio_structured": 0.75,
                "eventual_ratio_structured": 0.85,
            },
            rule_trace=[
                {
                    "rule_name": "test_rule",
                    "metric_name": "direct_ratio",
                    "computed_value": 0.80,
                    "threshold": 0.75,
                    "operator": ">=",
                    "outcome": "passed",
                }
            ],
            activity_count=5,
            total_dependencies=30,
            density=0.75,  # 30 / (5 * 4)
            loop_count=2,
        )

        # Serialize to JSON
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "structured" in json_str
        assert "exact" in json_str
        assert "0.8" in json_str or "0.80" in json_str

        # Deserialize back
        import json

        data = json.loads(json_str)
        result_restored = ClassificationResult.model_validate(data)

        assert result_restored.category == CategoryEnum.STRUCTURED
        assert result_restored.confidence == "exact"
        assert result_restored.dependency_counts["DIRECT"] == 10
        assert result_restored.dependency_ratios["direct_ratio"] == 0.80
        assert result_restored.activity_count == 5
        assert result_restored.total_dependencies == 30
        assert result_restored.loop_count == 2
        assert len(result_restored.rule_trace) == 1

    def test_result_density_validation(self):
        """Density should be in range [0.0, 1.0]."""
        # Valid density
        result = ClassificationResult(
            category=CategoryEnum.STRUCTURED,
            confidence="exact",
            dependency_counts={},
            dependency_ratios={},
            thresholds_applied={},
            rule_trace=[],
            activity_count=5,
            total_dependencies=20,
            density=0.5,
        )
        assert result.density == 0.5

        # Invalid density > 1.0 should fail
        with pytest.raises(ValidationError):
            ClassificationResult(
                category=CategoryEnum.STRUCTURED,
                confidence="exact",
                dependency_counts={},
                dependency_ratios={},
                thresholds_applied={},
                rule_trace=[],
                activity_count=5,
                total_dependencies=20,
                density=1.5,
            )

    def test_result_category_enum(self):
        """CategoryEnum should have all 4 categories."""
        assert CategoryEnum.UNSTRUCTURED == "unstructured"
        assert CategoryEnum.LOOSELY_STRUCTURED == "loosely_structured"
        assert CategoryEnum.SEMI_STRUCTURED == "semi_structured"
        assert CategoryEnum.STRUCTURED == "structured"

    def test_result_default_loop_count(self):
        """loop_count should default to 0."""
        result = ClassificationResult(
            category=CategoryEnum.STRUCTURED,
            confidence="exact",
            dependency_counts={},
            dependency_ratios={},
            thresholds_applied={},
            rule_trace=[],
            activity_count=5,
            total_dependencies=20,
            density=0.5,
        )
        assert result.loop_count == 0
