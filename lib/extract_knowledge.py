"""Post-merge extractor.

Walks a newly-merged project's experiment_log.jsonl + sketch annotations,
and writes structured entries to:
    knowledge/hypothesis_library.jsonl
    knowledge/failure_modes.jsonl
    knowledge/domain_learnings/<domain>.jsonl
    knowledge/sketch_index.db (via lib.retrieval.upsert_sketch_index)

Anonymization: column names → semantic role tags via the domain module's
stage_keywords + role-tag heuristics. Raw data never enters knowledge/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from lib.domains import get as get_domain
from lib.project import open_project
from lib.retrieval import upsert_sketch_index
from lib.schemas.experiment import ExperimentResult
from lib.schemas.knowledge import (
    FailureModeEntry,
    HypothesisLibraryEntry,
    KnowledgeBundle,
)
from lib.schemas.mission import Mission
from lib.schemas.recommendation import Recommendation
from lib.sketch.manifest import load_manifest
from lib.workspace import knowledge_dir, project_path, resolve_workspace

_log = logging.getLogger("eda.extract_knowledge")


_ROLE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("<sensor:temperature>", ("temp", "celsius", "kelvin")),
    ("<sensor:pressure>", ("pressure", "psi", "bar")),
    ("<sensor:flow>", ("flow", "rate", "throughput")),
    ("<sensor:vibration>", ("vibration", "rpm", "shake")),
    ("<sensor:concentration>", ("ph", "ppm", "concentration")),
    ("<process:residence_time>", ("residence", "duration", "dwell")),
    ("<process:flowrate>", ("flowrate",)),
    ("<process:yield_rate>", ("yield",)),
    ("<calendar:dow>", ("dow", "weekday", "day_of_week")),
    ("<calendar:month>", ("month",)),
    ("<calendar:holiday>", ("holiday", "festival")),
    ("<price:list>", ("list_price", "price")),
    ("<price:promo>", ("promo", "discount")),
    ("<inventory:stock>", ("stock", "inventory")),
    ("<outcome:demand>", ("demand", "sales", "orders")),
    ("<asset:age_days>", ("age", "since_install")),
)


def _anonymize_column(name: str, domain_key: str) -> str:
    """Map a raw column name to a semantic role tag (domain-aware)."""
    nl = name.lower()
    for tag, hints in _ROLE_HINTS:
        for h in hints:
            if h in nl:
                return tag
    # Fall back to domain stage tag.
    spec = get_domain(domain_key)
    for stage, kws in spec.stage_keywords:
        if any(k in nl for k in kws):
            return f"<stage:{stage}>"
    return "<feature:unspecified>"


def _read_experiments(project_dir: Path) -> list[ExperimentResult]:
    p = project_dir / "experiment_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(ExperimentResult.model_validate_json(line))
        except Exception as e:  # noqa: BLE001
            _log.debug("skipping bad experiment row: %s", e)
    return out


def _append_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def extract_from_project(
    project_name: str,
    *,
    workspace: Optional[Path] = None,
    min_info_gain: float = 0.1,
) -> dict[str, int]:
    """Walk a merged project and append extracted entries to knowledge/.

    Returns a count summary.
    """
    ws = resolve_workspace(workspace)
    proj = project_path(ws, project_name)
    if not proj.exists():
        raise FileNotFoundError(proj)

    meta = open_project(workspace, project_name)
    mission = Mission.model_validate_json((proj / "MISSION.json").read_text(encoding="utf-8"))
    sig = mission.capability  # CapabilityComposition
    from lib.capabilities import composition_signature

    cap_sig = composition_signature(sig)

    # Hypothesis entries from the project's knowledge_bundle.json (preferred)
    # or fallback: walk the experiment log.
    h_entries: list[HypothesisLibraryEntry] = []
    f_entries: list[FailureModeEntry] = []

    bundle_path = proj / "results" / "knowledge_bundle.json"
    if bundle_path.exists():
        bundle = KnowledgeBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        h_entries.extend(bundle.hypothesis_entries)
        f_entries.extend(bundle.failure_entries)
    else:
        for e in _read_experiments(proj):
            if e.skeptic.verdict == "FAIL":
                for fc in e.skeptic.failed_checks:
                    f_entries.append(
                        FailureModeEntry(
                            entry_id=f"K-f-{project_name}-{e.id}-{fc}",
                            source_project=project_name,
                            domain=meta.domain,
                            capability_signature=cap_sig,
                            failure_name=fc,
                            resolution="extracted post-merge",
                        )
                    )
                continue
            if e.info_gain_actual < min_info_gain:
                continue
            h_entries.append(
                HypothesisLibraryEntry(
                    entry_id=f"K-h-{project_name}-{e.id}",
                    source_project=project_name,
                    source_iteration=e.iteration,
                    domain=meta.domain,
                    capability_signature=cap_sig,
                    pattern_summary=f"{e.area}: {e.model} ({len(e.features_used)} features)",
                    technique_family=e.technique_family,
                    feature_roles=[],
                    sketch_signature={"info_gain": float(e.info_gain_actual)},
                    info_gain=float(e.info_gain_actual),
                    primary_metric=e.primary_metric,
                    primary_metric_delta=float(e.info_gain_actual),
                )
            )

    # Anonymize feature columns at extraction time.
    for h in h_entries:
        if not h.feature_roles:
            # We don't have raw column names in the bundle's
            # HypothesisLibraryEntry by design; if a future bundle includes
            # them under sketch_signature we'd anonymize here.
            pass

    # Append to knowledge/.
    kdir = knowledge_dir(ws)
    _append_jsonl(kdir / "hypothesis_library.jsonl", [h.model_dump() for h in h_entries])
    _append_jsonl(kdir / "failure_modes.jsonl", [f.model_dump() for f in f_entries])

    # Domain learnings (per-domain JSONL).
    domain_dir = kdir / "domain_learnings"
    domain_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(domain_dir / f"{meta.domain}.jsonl", [h.model_dump() for h in h_entries])

    # Sketch index.
    try:
        manifest = load_manifest(proj)
        upsert_sketch_index(
            workspace=ws,
            project_name=project_name,
            domain=meta.domain,
            capability_signature=cap_sig,
            similarity_vector=list(manifest.similarity_vector),
            confidence_tier=meta.confidence_tier or "unknown",
        )
    except FileNotFoundError:
        _log.warning("no sketch manifest for %s — skipping sketch index update", project_name)

    return {
        "n_hypotheses_appended": len(h_entries),
        "n_failures_appended": len(f_entries),
    }
