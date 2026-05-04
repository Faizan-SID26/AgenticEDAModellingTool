"""PROJECT.json — project-level metadata.

This is the root file for a project; it records lifecycle status, budget
caps, framework version pin (replay), and the running confidence_tier.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel

ProjectStatus = Literal[
    "created",
    "inspected",
    "planned",
    "running",
    "completed",
    "no_signal",
    "abandoned",
    "archived",
]
"""Lifecycle states. `no_signal` is the honest-failure shippable end state."""

ConfidenceTier = Literal["high", "medium", "low", "no_signal", "unknown"]


class ProjectMeta(VersionedModel):
    """Top-level project metadata. Committed."""

    project_name: str = Field(...)
    domain: str = Field(...)
    recipe: Optional[str] = Field(default=None)
    branch: str = Field(
        ...,
        description="Git branch the project lives on.",
    )
    status: ProjectStatus = Field(default="created")
    confidence_tier: ConfidenceTier = Field(default="unknown")

    framework_version_pin: str = Field(
        ...,
        description="lib.__version__ at /init time. Used by replay.",
    )

    token_budget: int = Field(..., gt=0)
    iteration_budget: int = Field(default=100, gt=0)

    created_by: Optional[str] = Field(default=None)
    last_run_started_at: Optional[str] = Field(default=None)
    last_run_finished_at: Optional[str] = Field(default=None)

    notes: str = Field(default="")
