"""Configuration for ARM classification engine.

Loads threshold configuration with Pydantic validation for fail-fast error handling.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

# Epsilon for floating-point comparisons
EPSILON = 1e-9


class ThresholdConfig(BaseModel):
    """Threshold configuration for ARM classification.

    All thresholds are ratios in range [0.0, 1.0].
    Defaults calibrated from Test Data/Classification ground truth (2026-02-06).
    """

    direct_ratio_structured: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum DIRECT ratio for structured classification",
    )

    eventual_ratio_structured: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum EVENTUAL ratio for structured classification",
    )

    direct_ratio_semi_max: float = Field(
        default=0.100,
        ge=0.0,
        le=1.0,
        description="Maximum DIRECT ratio for semi-structured classification",
    )

    eventual_ratio_semi_min: float = Field(
        default=0.100,
        ge=0.0,
        le=1.0,
        description="Minimum EVENTUAL ratio for semi-structured classification",
    )

    implication_ratio_semi: float = Field(
        default=0.143,
        ge=0.0,
        le=1.0,
        description="Minimum IMPLICATION ratio for semi-structured classification",
    )

    direct_ratio_loosely_max: float = Field(
        default=0.048,
        ge=0.0,
        le=1.0,
        description="Maximum DIRECT ratio for loosely-structured classification",
    )

    nand_or_ratio_loosely: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Minimum NAND/OR ratio for loosely-structured classification",
    )

    @field_validator("eventual_ratio_structured", "eventual_ratio_semi_min")
    @classmethod
    def eventual_gte_direct(cls, v: float, info) -> float:
        """Validate logical ordering: direct_ratio <= eventual_ratio.

        DIRECT ⊂ EVENTUAL, so eventual ratio should be >= direct ratio.
        """
        # Get the corresponding direct field name
        field_name = info.field_name
        if field_name == "eventual_ratio_structured":
            direct_field = "direct_ratio_structured"
        elif field_name == "eventual_ratio_semi_min":
            # For semi-structured, we check against semi_max not structured
            return v
        else:
            return v

        # Get direct value from data
        if direct_field in info.data:
            direct_val = info.data[direct_field]
            if v < direct_val - EPSILON:
                raise ValueError(
                    f"{field_name}={v} must be >= {direct_field}={direct_val} "
                    "(DIRECT ⊂ EVENTUAL)"
                )

        return v


class ConfigLoader:
    """Loader for threshold configuration with fail-fast validation."""

    @staticmethod
    def load(config_path: Path | None = None) -> ThresholdConfig:
        """Load threshold configuration from YAML file.

        Args:
            config_path: Path to YAML config file. If None, returns defaults.

        Returns:
            Validated ThresholdConfig instance.

        Raises:
            ValueError: If config file is invalid or values out of range.
            FileNotFoundError: If config_path specified but doesn't exist.
        """
        if config_path is None:
            return ThresholdConfig()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)

            if data is None:
                raise ValueError("Config file is empty")

            # Validate with Pydantic
            return ThresholdConfig.model_validate(data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to load config: {e}")
