"""Shared base model with `schema_version` and `framework_version` fields.

All concrete schemas in `lib.schemas.*` extend `VersionedModel`. This is the
mechanism by which we can later detect and migrate older artifacts: every
write stamps the current `lib.SCHEMA_VERSION` and `lib.__version__`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from lib import SCHEMA_VERSION, __version__


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp used as the default for `created_at` fields."""
    return datetime.now(tz=timezone.utc).isoformat()


class VersionedModel(BaseModel):
    """Base for every framework artifact.

    Subclasses set ``model_config`` only if they need overrides. Two
    forward-compatibility fields are included on every artifact:

    - ``schema_version`` — bumps on breaking schema changes.
    - ``framework_version`` — pins the code that wrote the artifact, so
      replay knows whether it can rehydrate without a migration.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema epoch. Used by replay/migration tooling.",
    )
    framework_version: str = Field(
        default=__version__,
        description="`lib.__version__` at the moment this artifact was written.",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="ISO-8601 UTC timestamp when this record was first created.",
    )
