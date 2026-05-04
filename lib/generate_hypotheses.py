"""Hypothesis generation: every 5 iterations.

Inputs:
- Sketch (top_interactions, regimes, motifs, causal_neighbors).
- Cross-project knowledge (similar past projects' successful patterns).
- Bandit posteriors.
- Recent experiment outcomes.

Output: 3-5 candidate hypotheses to seed the next iterations, each with an
expected info-gain prior. Cold-start (iter < 5) uses the universal seeds
verbatim; warm-start (iter ≥ 5) generates new hypotheses.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from lib.bandit import load as bandit_load, posterior_means
from lib.schemas.experiment import ExperimentResult
from lib.schemas.mission import Mission
from lib.sketch.l2_joint import load_l2
from lib.sketch.l3_regimes import load_l3
from lib.sketch.l6_causal import load_l6
from lib.sketch.l7_failure_modes import load_l7
from lib.sketch.manifest import load_manifest

_log = logging.getLogger("eda.hyp")


_MAX_HYPOTHESES = 5


def _pick_top_arm(project_dir: Path) -> str:
    """Pick the technique family with the highest posterior mean."""
    state = bandit_load(project_dir)
    means = posterior_means(state)
    return max(means.items(), key=lambda kv: kv[1])[0]


def _hypothesis_from_interaction(it: dict, *, family: str, idx: int, iteration: int) -> dict[str, Any]:
    return {
        "hypothesis_id": f"H-iter{iteration}-int{idx}",
        "name": f"interaction_{it.get('col_a')}_{it.get('col_b')}",
        "summary": (
            f"Interaction between {it.get('col_a')} and {it.get('col_b')} "
            f"(MI={it.get('mutual_info', 0.0):.3f}) was promoted; test it explicitly."
        ),
        "technique_family": family,
        "area": "interactions",
        "model_hint": "lgbm_default",
        "features_dsl": [it.get("col_a"), it.get("col_b"), "engineered:interactions_top5"],
        "expected_info_gain": float(0.5 + 0.5 * float(it.get("mutual_info", 0.0))),
        "rationale": "Promoted by L2 / iteration outcomes.",
        "source": "generator",
    }


def _hypothesis_from_regime(l3, family: str, iteration: int) -> Optional[dict[str, Any]]:
    if l3.n_regimes < 2:
        return None
    return {
        "hypothesis_id": f"H-iter{iteration}-regime",
        "name": "regime_specific_submodel_iter",
        "summary": f"L3 reports {l3.n_regimes} regimes; train per-regime submodels.",
        "technique_family": family,
        "area": "regimes",
        "model_hint": "lgbm_per_regime",
        "features_dsl": ["+all_allowed"],
        "expected_info_gain": 0.55,
        "rationale": f"Regime structure (L3): n_regimes={l3.n_regimes}",
        "source": "generator",
    }


def _hypothesis_from_causal(l6, target: str, family: str, iteration: int) -> Optional[dict[str, Any]]:
    if not l6.edges:
        return None
    nbrs = [
        e["src"] if e["dst"] == target else e["dst"]
        for e in l6.edges
        if target in (e["src"], e["dst"])
    ]
    if not nbrs:
        return None
    return {
        "hypothesis_id": f"H-iter{iteration}-causal",
        "name": "causal_neighbors_only",
        "summary": (
            f"Restrict to L6 causal neighbors of {target}: {nbrs[:5]}; test attribution."
        ),
        "technique_family": family,
        "area": "causal",
        "model_hint": "logreg_or_ridge",
        "features_dsl": list(nbrs[:5]),
        "expected_info_gain": 0.5,
        "rationale": "L6 causal hints suggest these neighbors carry direct effect.",
        "source": "generator",
    }


def _hypothesis_from_failures(catalog, family: str, iteration: int) -> Optional[dict[str, Any]]:
    if not catalog:
        return None
    biggest = max(catalog, key=lambda c: c.n_observations)
    return {
        "hypothesis_id": f"H-iter{iteration}-failmode",
        "name": "robustness_against_failure_cluster",
        "summary": (
            f"Failure cluster {biggest.cluster_id} has fired {biggest.n_observations} times; "
            "test a robustness-promoting variant (regularization / ensembling)."
        ),
        "technique_family": "ensemble",
        "area": "robustness",
        "model_hint": "lgbm_default",
        "features_dsl": ["+all_allowed"],
        "expected_info_gain": 0.4,
        "rationale": f"L7 cluster {biggest.cluster_id} dominant.",
        "source": "generator",
    }


def generate(
    project_dir: Path,
    mission: Mission,
    iteration: int,
    *,
    cold_start_iterations: int = 5,
    seeds_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Return a fresh list of 3-5 hypotheses for upcoming iterations.

    For ``iteration < cold_start_iterations``, returns the universal seeds
    untouched. After that, mixes sketch-derived hypotheses + the
    best-arm prior from the bandit.
    """
    project_dir = Path(project_dir)
    if iteration < cold_start_iterations:
        seeds_path = seeds_path or (project_dir / "memory" / "HYPOTHESES.jsonl")
        if not seeds_path.exists():
            return []
        out = []
        for line in seeds_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= _MAX_HYPOTHESES:
                break
        return out

    manifest = load_manifest(project_dir)
    family = _pick_top_arm(project_dir)
    out: list[dict[str, Any]] = []

    l2 = load_l2(project_dir / manifest.l2_path)
    for idx, it in enumerate(l2.top_interactions[:2]):
        out.append(_hypothesis_from_interaction(it, family=family, idx=idx, iteration=iteration))

    l3 = load_l3(project_dir / manifest.l3_path)
    h = _hypothesis_from_regime(l3, family=family, iteration=iteration)
    if h:
        out.append(h)

    l6 = load_l6(project_dir / manifest.l6_path)
    h = _hypothesis_from_causal(l6, target=mission.target_column, family=family, iteration=iteration)
    if h:
        out.append(h)

    catalog = load_l7(project_dir / manifest.l7_path)
    h = _hypothesis_from_failures(catalog, family=family, iteration=iteration)
    if h:
        out.append(h)

    return out[:_MAX_HYPOTHESES]


def append_to_log(project_dir: Path, hypotheses: Iterable[dict[str, Any]]) -> Path:
    """Append generated hypotheses to memory/HYPOTHESES.jsonl."""
    p = Path(project_dir) / "memory" / "HYPOTHESES.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps(h) + "\n")
    return p
