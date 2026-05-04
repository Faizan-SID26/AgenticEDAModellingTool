"""Pytest fixtures shared across the suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root directory (containing pyproject.toml)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """Empty workspace directory under pytest tmp_path."""
    (tmp_path / "projects").mkdir()
    (tmp_path / "knowledge").mkdir()
    return tmp_path
