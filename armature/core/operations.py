"""Operation models for batch Matrix changes.

This module defines Pydantic models for batch operations on the Matrix.
Operations are immutable and validated, enabling type-safe batch execution.
"""

from pydantic import BaseModel
from typing import Literal
from armature.core.dependencies import TemporalDependency, ExistentialDependency


class AddActivity(BaseModel):
    """Add an activity to the matrix."""

    op: Literal["add_activity"] = "add_activity"
    activity: str

    model_config = {"frozen": True}


class RemoveActivity(BaseModel):
    """Remove an activity from the matrix."""

    op: Literal["remove_activity"] = "remove_activity"
    activity: str

    model_config = {"frozen": True}


class SetDependency(BaseModel):
    """Set dependency between two activities."""

    op: Literal["set_dependency"] = "set_dependency"
    source: str
    target: str
    temporal: TemporalDependency | None = None
    existential: ExistentialDependency | None = None

    model_config = {"frozen": True}


# Type alias for discriminated union
MatrixOperation = AddActivity | RemoveActivity | SetDependency
