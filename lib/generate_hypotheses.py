"""Hypothesis generation: every 5 iterations.

Inputs:
- Sketch (top_interactions, regimes, motifs, causal_neighbors, failures).
- Cross-project knowledge (similar past projects' successful patterns).
- Bandit posteriors.
- Recent experiment outcomes + RUN_STATE.iterations_since_improvement.
- MISSION.notes (process knowledge captured at /plan).

Output: 5-12 candidate hypotheses, *deliberately diversified* across
technique families and areas. The previous version emitted at most 5
hypotheses biased to the bandit's top arm — that converged too fast.
The new version:

- Always emits at least 8 hypotheses when warm.
- Spans the **top 3 bandit arms** instead of just the top 1.
- Always includes at least one **wildcard / SOTA** hypothesis pointing
  at a technique family that has *not* been tried yet.
- When `iterations_since_improvement` rises, biases toward
  area=`robustness` / `causal` / `ensembling` (different shape, not
  just different hyperparameters).
- Pulls **process-knowledge** hypotheses out of MISSION.notes
  ("expected_drivers" → first-class hypotheses).
- Pulls **cross-project knowledge** from knowledge/hypothesis_library
  via lib.retrieval (sketch-similarity-ranked).

Cold-start (iter < 5) still returns the seed list untouched.
"""
from __future__ import annotations

import json
import logging
import re
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


_MAX_HYPOTHESES = 12
_MIN_HYPOTHESES_WARM = 8
_STAGNATION_THRESHOLD = 3
"""When iterations_since_improvement >= this value, force the generator
into 'shake the tree' mode (more wildcards, more area diversity)."""


# --- Bandit helpers ----------------------------------------------------


def _bandit_top_arms(project_dir: Path, k: int = 3) -> list[str]:
    """Return the top-k technique families by posterior mean."""
    state = bandit_load(project_dir)
    means = posterior_means(state)
    return [f for f, _ in sorted(means.items(), key=lambda kv: -kv[1])[:k]]


def _bandit_untried_families(project_dir: Path, *, n: int = 3) -> list[str]:
    """Return up to `n` families with the *least* observation mass — i.e.,
    families the bandit hasn't yet sampled. These are the wildcard arms."""
    state = bandit_load(project_dir)
    obs = {f: state.alpha.get(f, 1.0) + state.beta.get(f, 1.0) for f in state.alpha}
    return [f for f, _ in sorted(obs.items(), key=lambda kv: kv[1])[:n]]


# --- Single-hypothesis builders ----------------------------------------


def _hypothesis_from_interaction(
    it: dict, *, family: str, idx: int, iteration: int
) -> dict[str, Any]:
    return {
        "hypothesis_id": f"H-iter{iteration}-int{idx}",
        "name": f"interaction_{it.get('col_a')}_{it.get('col_b')}",
        "summary": (
            f"Interaction between {it.get('col_a')} and {it.get('col_b')} "
            f"(MI={it.get('mutual_info', 0.0):.3f}); test it as an explicit feature."
        ),
        "technique_family": family,
        "area": "interactions",
        "model_hint": "lgbm_default",
        "features_dsl": [it.get("col_a"), it.get("col_b"), "engineered:interactions_top5"],
        "expected_info_gain": float(0.5 + 0.5 * float(it.get("mutual_info", 0.0))),
        "rationale": "Promoted by L2.",
        "source": "generator",
    }


def _hypothesis_from_regime(l3, family: str, iteration: int) -> Optional[dict[str, Any]]:
    if l3.n_regimes < 2:
        return None
    return {
        "hypothesis_id": f"H-iter{iteration}-regime",
        "name": "regime_specific_submodel_iter",
        "summary": f"L3 reports {l3.n_regimes} regimes; train per-regime submodels and score on out-of-regime data.",
        "technique_family": family,
        "area": "regimes",
        "model_hint": "lgbm_per_regime",
        "features_dsl": ["+all_allowed"],
        "expected_info_gain": 0.55,
        "rationale": f"L3 n_regimes={l3.n_regimes}",
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
            f"Restrict features to L6 causal neighbors of {target}: {nbrs[:5]}; "
            "tests whether the direct-effect set carries the signal."
        ),
        "technique_family": family,
        "area": "causal",
        "model_hint": "logreg_or_ridge",
        "features_dsl": list(nbrs[:5]),
        "expected_info_gain": 0.55,
        "rationale": "L6 causal neighbors of target.",
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
            f"Failure cluster {biggest.cluster_id} fired {biggest.n_observations} times; "
            "test a robustness-promoting variant (regularization / ensembling / outlier exclusion)."
        ),
        "technique_family": "ensemble",
        "area": "robustness",
        "model_hint": "lgbm_default",
        "features_dsl": ["+all_allowed"],
        "expected_info_gain": 0.45,
        "rationale": f"L7 cluster {biggest.cluster_id} dominant.",
        "source": "generator",
    }


# Family → list of registry keys, capability-agnostic. Each entry's keys are
# filtered through `lib.registry.is_available(model_key, capability_key)` so
# only models that can actually be built in the current environment surface.
# Replaces the prior single-key `family_to_model` mapping (which collapsed
# `neural` to `logreg` and made wildcards meaningless).
_FAMILY_TO_KEYS: dict[str, list[str]] = {
    "linear": ["elasticnet", "logreg_l1", "logreg_elastic", "ridge_classifier", "logreg", "ridge"],
    "tree": ["extra_trees", "random_forest", "decision_tree", "lgbm_binary"],
    "boosted_tree": ["xgboost_binary", "catboost_binary", "lgbm_focal", "lgbm_weighted", "xgboost_focal", "lgbm_binary"],
    "neural": ["mlp_tabular", "ft_transformer", "tabnet"],
    "ensemble": ["stacked_blend", "voting_soft", "bagging"],
    "rule_based": ["decision_tree", "logreg"],
    "survival": ["random_survival_forest", "lgbm_aft", "cox_ph", "lgbm_survival"],
    "anomaly": ["autoencoder_anomaly", "isolation_forest", "ocsvm", "lof"],
    "forecasting_classical": ["prophet", "theta", "ets", "naive_seasonal"],
    "forecasting_neural": ["ridge_lagged", "lgbm_regressor", "mlp_tabular_regressor"],
}


def _wildcard_keys_for(family: str, capability_key: str) -> list[str]:
    """Available registry keys for `family` filtered by capability +
    optional-import availability. Empty list means the wildcard generator
    skips this family — preferable to emitting unbuildable plans."""
    from lib.registry import is_available
    candidates = _FAMILY_TO_KEYS.get(family, [])
    return [k for k in candidates if is_available(k, capability_key)]


def _wildcard_hypothesis(
    family: str,
    iteration: int,
    *,
    model_key: Optional[str] = None,
    rationale_extra: str = "",
) -> dict[str, Any]:
    """One wildcard hypothesis. When `model_key` is provided the hypothesis
    is grounded in that specific registry key (real escape). When omitted,
    falls back to the legacy "let the researcher pick" shape — used outside
    breakthrough mode for back-compat."""
    if model_key is None:
        model_hint = "logreg"
        suffix = family
    else:
        model_hint = model_key
        suffix = f"{family}-{model_key}"
    summary_parts = [
        f"Wildcard arm: try a {family} variant we have not yet sampled.",
    ]
    if model_key:
        summary_parts.append(f"Concrete model: `{model_key}` (registry-resolved).")
    if rationale_extra:
        summary_parts.append(rationale_extra)
    return {
        "hypothesis_id": f"H-iter{iteration}-wild-{suffix}",
        "name": f"wildcard_{suffix}",
        "summary": " ".join(summary_parts),
        "technique_family": family,
        "area": "baseline",
        "model_hint": model_hint,
        "features_dsl": ["+all_allowed"],
        # Wildcards have an *exploratory* prior, not a high one.
        "expected_info_gain": 0.5,
        "rationale": f"Bandit underexplored '{family}'; novelty hypothesis.",
        "source": "generator_wildcard",
    }


def _expected_driver_hypotheses(
    mission: Mission, iteration: int, family: str
) -> list[dict[str, Any]]:
    """Mine MISSION.notes 'Expected drivers:' section into 1-2 hypotheses."""
    notes = mission.notes or ""
    if "Expected drivers" not in notes:
        return []
    # Pull the body of the "Expected drivers" section (until next blank line + label).
    m = re.search(
        r"Expected drivers:\s*\n([\s\S]+?)(\n\n[A-Z][a-z][^\n]*:|\Z)",
        notes,
    )
    if not m:
        return []
    body = m.group(1)
    # Identify column-name-shaped tokens. We can't be picky here; downstream
    # audit + DSL expansion will silently drop any token that isn't a real column.
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", body)
    # Heuristic stop-word filter for human prose.
    stop = {"the", "and", "or", "with", "a", "an", "of", "in", "on", "for", "to",
            "is", "are", "was", "were", "be", "by", "as", "at", "from", "into",
            "expected", "drivers"}
    tokens = [t for t in tokens if t.lower() not in stop]
    if len(tokens) < 2:
        return []
    out: list[dict[str, Any]] = []
    out.append(
        {
            "hypothesis_id": f"H-iter{iteration}-expected-drivers",
            "name": "user_expected_drivers_only",
            "summary": (
                f"Restrict features to the user's expected drivers from MISSION.notes "
                f"({tokens[:5]}); tests whether the practitioner's mental model holds."
            ),
            "technique_family": family,
            "area": "features",
            "model_hint": "logreg_or_ridge",
            "features_dsl": list(dict.fromkeys(tokens))[:5],
            "expected_info_gain": 0.5,
            "rationale": "Mined from MISSION.notes Expected drivers.",
            "source": "generator_user_priors",
        }
    )
    if len(tokens) >= 4:
        out.append(
            {
                "hypothesis_id": f"H-iter{iteration}-expected-driver-interactions",
                "name": "user_driver_interactions",
                "summary": (
                    f"Pairwise interactions among the user's expected drivers "
                    f"({tokens[:4]}) on top of +all_allowed; tests whether the "
                    "process-knowledge interactions carry residual signal."
                ),
                "technique_family": family,
                "area": "interactions",
                "model_hint": "lgbm_default",
                "features_dsl": ["+all_allowed", "engineered:interactions_top5"],
                "expected_info_gain": 0.55,
                "rationale": "Process-knowledge × pairwise interactions.",
                "source": "generator_user_priors",
            }
        )
    return out


def _cross_project_hypotheses(
    project_dir: Path,
    mission: Mission,
    iteration: int,
    family: str,
    *,
    workspace: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Pull the highest-info-gain hypothesis patterns from cross-project
    knowledge that match this project's domain + capability.

    Hydrates `model_hint` and `features_dsl` from the source entry when
    available (Pillar 3) so cross-project knowledge becomes a real plan to
    replay rather than a relabeled LightGBM run. Legacy entries without
    those fields fall back to `lgbm_default` + `+all_allowed`.
    """
    try:
        from lib.capabilities import composition_signature
        from lib.retrieval import query_hypotheses

        sig = composition_signature(mission.capability)
        rows = query_hypotheses(
            workspace=workspace,
            domain=mission.domain,
            capability_signature=sig,
            min_info_gain=0.2,
            top_k=4,
        )
    except Exception as e:  # noqa: BLE001 — never fail iteration on retrieval
        _log.debug("cross-project retrieval failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    cap_key = None
    try:
        from lib.capabilities import validate_composition
        cap_key = validate_composition(mission.capability).key
    except Exception:  # noqa: BLE001 — keep going on capability resolution failure
        cap_key = None
    for i, r in enumerate(rows):
        # Hydrate from the source entry when present and the model is
        # actually available for this capability; otherwise fall back to a
        # generic LGBM shape.
        src_model = getattr(r, "model", None)
        src_dsl = list(getattr(r, "feature_dsl", []) or [])
        src_params = dict(getattr(r, "params", {}) or {})
        if src_model and cap_key:
            try:
                from lib.registry import is_available
                if not is_available(src_model, cap_key):
                    src_model = None
            except Exception:  # noqa: BLE001
                pass
        out.append(
            {
                "hypothesis_id": f"H-iter{iteration}-xproj-{i}",
                "name": f"cross_project_{r.entry_id}",
                "summary": (
                    f"Cross-project pattern from `{r.source_project}` "
                    f"(info_gain={r.info_gain:.2f}): {r.pattern_summary}"
                ),
                "technique_family": r.technique_family or family,
                "area": "features",
                "model_hint": src_model or "lgbm_default",
                "features_dsl": src_dsl or ["+all_allowed"],
                "params": src_params,
                "expected_info_gain": float(min(0.85, r.info_gain)),
                "rationale": (
                    f"Cross-project retrieval. Source={r.source_project}; "
                    f"hydrated model={src_model or 'fallback'}, "
                    f"dsl_tokens={len(src_dsl)}."
                ),
                "source": "generator_cross_project",
            }
        )
    return out


# --- Stagnation-aware composition --------------------------------------


def _iterations_since_improvement(project_dir: Path) -> int:
    try:
        from lib.state import load_run_state

        return int(load_run_state(project_dir).iterations_since_improvement)
    except Exception:  # noqa: BLE001
        return 0


def _breakthrough_active(project_dir: Path) -> bool:
    try:
        from lib.state import load_run_state

        return bool(load_run_state(project_dir).breakthrough_mode_active)
    except Exception:  # noqa: BLE001
        return False


def _tried_models(project_dir: Path) -> set[str]:
    """Set of registry keys the project has actually fitted (non-FAIL only).
    Used by breakthrough wildcards so each new wildcard escapes a model the
    project has not used yet."""
    try:
        from lib.state import read_experiments

        return {
            e.model
            for e in read_experiments(project_dir)
            if e.skeptic.verdict != "FAIL" and e.model
        }
    except Exception:  # noqa: BLE001
        return set()


def _reviewer_directives(project_dir: Path) -> list[dict[str, Any]]:
    """Read all `source=reviewer_directive` hypotheses from HYPOTHESES.jsonl
    that haven't yet expired (the generator marks them consumed by setting
    `consumed=True` after first selection — see `mark_directive_consumed`)."""
    p = project_dir / "memory" / "HYPOTHESES.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("source") == "reviewer_directive" and not r.get("consumed"):
            out.append(r)
    return out


def generate(
    project_dir: Path,
    mission: Mission,
    iteration: int,
    *,
    cold_start_iterations: int = 5,
    seeds_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Return 5-12 candidate hypotheses for upcoming iterations.

    Cold-start (iter < cold_start_iterations) returns the universal seed
    list verbatim. Warm-start composes:

        - up to 3 L2 interaction hypotheses (rotating the family across top-3 bandit arms)
        - 1 L3 regime hypothesis (if regimes>=2)
        - 1 L6 causal-neighbors hypothesis
        - 1 L7 robustness hypothesis (if any failure clusters)
        - 1-2 user-prior hypotheses from MISSION.notes "Expected drivers"
        - up to 4 cross-project hypotheses
        - up to 2 wildcard hypotheses (least-tried bandit families)

    Under stagnation (iterations_since_improvement >= 3), the wildcard
    count doubles and the L7 robustness hypothesis is *always* emitted
    even without observed failures (with a generic "shake the tree"
    rationale).
    """
    project_dir = Path(project_dir)

    # Reviewer-directive hypotheses jump the queue regardless of cold/warm
    # state. They expire after first selection (consumed flag).
    directive_out: list[dict[str, Any]] = []
    for d in _reviewer_directives(project_dir):
        directive_out.append(d)

    if iteration < cold_start_iterations:
        seeds_path = seeds_path or (project_dir / "memory" / "HYPOTHESES.jsonl")
        if not seeds_path.exists():
            return directive_out
        out = list(directive_out)
        for line in seeds_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Don't double-emit reviewer directives that are already in
            # directive_out (HYPOTHESES.jsonl is appended to during /run).
            if rec.get("source") == "reviewer_directive":
                continue
            out.append(rec)
            if len(out) >= _MAX_HYPOTHESES:
                break
        return out

    manifest = load_manifest(project_dir)
    top_arms = _bandit_top_arms(project_dir, k=3) or ["boosted_tree", "linear", "tree"]
    untried = _bandit_untried_families(project_dir, n=4)
    stagnant = _iterations_since_improvement(project_dir) >= _STAGNATION_THRESHOLD
    in_breakthrough = _breakthrough_active(project_dir)

    # Resolve capability_key once for `is_available` filtering.
    cap_key: Optional[str] = None
    try:
        from lib.capabilities import validate_composition
        cap_key = validate_composition(mission.capability).key
    except Exception:  # noqa: BLE001
        cap_key = None

    # Reviewer directives lead the list (Pillar 9 priority).
    out: list[dict[str, Any]] = list(directive_out)

    # Sketch-driven hypotheses, rotating the family across the top arms
    # so diversity is structural, not an afterthought.
    l2 = load_l2(project_dir / manifest.l2_path)
    for idx, it in enumerate(l2.top_interactions[:3]):
        fam = top_arms[idx % len(top_arms)]
        out.append(_hypothesis_from_interaction(it, family=fam, idx=idx, iteration=iteration))

    l3 = load_l3(project_dir / manifest.l3_path)
    h = _hypothesis_from_regime(l3, family=top_arms[0], iteration=iteration)
    if h:
        out.append(h)

    l6 = load_l6(project_dir / manifest.l6_path)
    h = _hypothesis_from_causal(
        l6, target=mission.target_column, family=top_arms[1 % len(top_arms)], iteration=iteration
    )
    if h:
        out.append(h)

    catalog = load_l7(project_dir / manifest.l7_path)
    h = _hypothesis_from_failures(catalog, family=top_arms[0], iteration=iteration)
    if h:
        out.append(h)
    elif stagnant:
        # Force a robustness experiment under stagnation, even without observed failures.
        out.append(
            {
                "hypothesis_id": f"H-iter{iteration}-stagnation-robust",
                "name": "stagnation_robustness_probe",
                "summary": (
                    "Stagnation: emit a robustness-shaped baseline with stronger regularization "
                    "/ outlier-aware loss to surface whether overfitting is the limiting factor."
                ),
                "technique_family": "ensemble",
                "area": "robustness",
                "model_hint": "lgbm_default",
                "features_dsl": ["+all_allowed"],
                "expected_info_gain": 0.45,
                "rationale": "Stagnation-aware probe.",
                "source": "generator_stagnation",
            }
        )

    # User-prior hypotheses from MISSION.notes (process knowledge).
    out.extend(
        _expected_driver_hypotheses(mission, iteration, family=top_arms[0])
    )

    # Cross-project hypotheses.
    out.extend(
        _cross_project_hypotheses(
            project_dir, mission, iteration, family=top_arms[0], workspace=workspace
        )
    )

    # Wildcard hypotheses. Breakthrough mode emits one wildcard per untried
    # *registry key*, not per family — so the researcher actually has
    # somewhere different to escape to. Outside breakthrough mode the
    # behavior is the legacy "1, doubled under stagnation" shape so
    # back-compat is preserved.
    if in_breakthrough and cap_key:
        tried = _tried_models(project_dir)
        # Iterate untried families in bandit order; for each, emit a wildcard
        # per registry key not yet tried, capped at 4 total.
        breakthrough_quota = 4
        emitted = 0
        for fam in untried:
            if emitted >= breakthrough_quota:
                break
            keys = _wildcard_keys_for(fam, cap_key)
            for key in keys:
                if emitted >= breakthrough_quota:
                    break
                if key in tried:
                    continue
                out.append(
                    _wildcard_hypothesis(
                        fam,
                        iteration=iteration,
                        model_key=key,
                        rationale_extra="Breakthrough mode: forced novelty.",
                    )
                )
                emitted += 1
    else:
        n_wild = 2 if stagnant else 1
        for fam in untried[:n_wild]:
            # Pick the best available registry key for the family if we know
            # the capability; otherwise leave model_key=None (legacy shape).
            model_key = None
            if cap_key:
                keys = _wildcard_keys_for(fam, cap_key)
                tried = _tried_models(project_dir)
                untried_keys = [k for k in keys if k not in tried]
                model_key = untried_keys[0] if untried_keys else (keys[0] if keys else None)
            out.append(
                _wildcard_hypothesis(fam, iteration=iteration, model_key=model_key)
            )

    # Deduplicate by hypothesis_id and cap the list.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for h in out:
        hid = h.get("hypothesis_id")
        if hid in seen:
            continue
        seen.add(hid)
        deduped.append(h)

    # Top-up to the warm-start floor with extra wildcards if we somehow
    # came up short (e.g. tiny sketch).
    while len(deduped) < _MIN_HYPOTHESES_WARM and untried:
        fam = untried.pop(0)
        h = _wildcard_hypothesis(fam, iteration=iteration)
        if h["hypothesis_id"] not in seen:
            deduped.append(h)
            seen.add(h["hypothesis_id"])

    return deduped[:_MAX_HYPOTHESES]


def append_to_log(project_dir: Path, hypotheses: Iterable[dict[str, Any]]) -> Path:
    """Append generated hypotheses to memory/HYPOTHESES.jsonl."""
    p = Path(project_dir) / "memory" / "HYPOTHESES.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps(h) + "\n")
    return p
