"""Budget ledger tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lib.budget import current_total, fraction_consumed, record_event
from lib.project import create_project


@pytest.fixture()
def proj(tmp_workspace: Path) -> Path:
    return create_project(name="bud", domain="manufacturing", token_budget=1000, workspace=tmp_workspace)


def test_record_and_running_total(proj: Path):
    e1 = record_event(proj, iteration=0, event="bootstrap", role="researcher", cap=1000, input_tokens=100, output_tokens=20)
    assert e1.cumulative_total == 120
    e2 = record_event(proj, iteration=1, event="iter_end", role="researcher", cap=1000, input_tokens=200, output_tokens=30)
    assert e2.sequence == 1
    assert e2.cumulative_total == 350
    assert current_total(proj) == 350
    assert abs(fraction_consumed(proj, cap=1000) - 0.35) < 1e-9
