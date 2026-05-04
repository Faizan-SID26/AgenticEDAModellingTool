"""Thompson sampling over technique families.

Each family carries a Beta(α, β) posterior over "expected info gain" that
is updated after each experiment. `pick_arm(seed)` samples one arm; the
researcher uses the result as a strong prior, not a hard rule.

Persisted JSON: `<project>/memory/BANDIT.json`.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lib.schemas.plan import TechniqueFamily

_log = logging.getLogger("eda.bandit")


_BANDIT_PATH_REL = "memory/BANDIT.json"

_FAMILIES: tuple[str, ...] = (
    "linear",
    "tree",
    "boosted_tree",
    "neural",
    "ensemble",
    "rule_based",
    "survival",
    "anomaly",
    "forecasting_classical",
    "forecasting_neural",
)


@dataclass
class BanditState:
    """Beta posterior parameters per arm."""

    alpha: dict[str, float] = field(default_factory=lambda: {f: 1.0 for f in _FAMILIES})
    beta: dict[str, float] = field(default_factory=lambda: {f: 1.0 for f in _FAMILIES})

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> "BanditState":
        return cls(alpha=dict(d.get("alpha", {})), beta=dict(d.get("beta", {})))


def _path(project_dir: Path) -> Path:
    return Path(project_dir) / _BANDIT_PATH_REL


def load(project_dir: Path) -> BanditState:
    p = _path(project_dir)
    if not p.exists():
        return BanditState()
    return BanditState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save(project_dir: Path, state: BanditState) -> Path:
    p = _path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return p


def update(state: BanditState, family: str, info_gain_actual: float) -> BanditState:
    """Update one arm's posterior.

    `info_gain_actual` is mapped to a Bernoulli reward via clipping to [0,1].
    """
    r = max(0.0, min(1.0, float(info_gain_actual)))
    state.alpha[family] = float(state.alpha.get(family, 1.0) + r)
    state.beta[family] = float(state.beta.get(family, 1.0) + (1.0 - r))
    return state


def sample(state: BanditState, *, seed: int = 0) -> dict[str, float]:
    """Return a Thompson sample from each arm's posterior."""
    rng = np.random.default_rng(seed)
    return {
        f: float(rng.beta(state.alpha.get(f, 1.0), state.beta.get(f, 1.0)))
        for f in _FAMILIES
    }


def pick_arm(state: BanditState, *, seed: int = 0) -> str:
    """Pick the best-sampled arm."""
    s = sample(state, seed=seed)
    return max(s.items(), key=lambda kv: kv[1])[0]


def posterior_means(state: BanditState) -> dict[str, float]:
    return {
        f: float(state.alpha.get(f, 1.0) / (state.alpha.get(f, 1.0) + state.beta.get(f, 1.0)))
        for f in _FAMILIES
    }
