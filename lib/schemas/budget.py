"""Budget ledger entry schema (one row per ledger event)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel

LedgerEvent = Literal[
    "iter_start",
    "iter_end",
    "synthesis",
    "vision_checkpoint",
    "hypothesis_gen",
    "bootstrap",
    "finalize",
    "interrupt",
]


class BudgetLedgerEntry(VersionedModel):
    """One append-only row in `budget.jsonl`.

    Replay reconstructs the running budget by summing tokens column.
    """

    sequence: int = Field(
        ...,
        ge=0,
        description="Monotonic counter starting at 0 within the project.",
    )
    iteration: int = Field(..., ge=0)
    event: LedgerEvent = Field(...)
    role: str = Field(
        ...,
        description="Role that consumed tokens: planner|researcher|reviewer|analyst|runner|skeptic.",
    )

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    cumulative_total: int = Field(
        ...,
        ge=0,
        description="Running total after this entry (computed by lib.budget).",
    )
    cap: int = Field(..., gt=0, description="Token cap from MISSION.budget at write time.")
    fraction_consumed: float = Field(
        ...,
        ge=0.0,
        description="cumulative_total / cap. Used by termination check.",
    )

    notes: Optional[str] = Field(default=None)
