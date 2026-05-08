"""Capability-driven web-search query helpers used by the `literature`
subagent in breakthrough mode.

This module does NOT call `WebSearch` — that tool is only available to the
LLM. The literature agent does the calling. Here we provide:

- `build_sota_queries(...)` — return capability- and family-aware query
  strings the agent should run, plus suggested year range for recency.
- `parse_search_hits(...)` — normalize raw `WebSearch` results into
  structured `PaperHit` dicts the researcher can fold into a plan's
  `prior_evidence`, `params`, and `features`.

Generic across domains: queries are built from `capability_key` +
`technique_family` + `problem_signature`. `domain_key` is appended only
when present and not equal to the placeholder `"general"`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


_DEFAULT_YEAR_MIN = 2022


_CAPABILITY_QUERY_HINTS: dict[str, list[str]] = {
    "tabular_classification": [
        "imbalanced classification techniques",
        "tabular classification state of the art",
        "calibration deep learning tabular",
    ],
    "temporal_classification": [
        "temporal classification deep learning",
        "imbalanced temporal classification",
    ],
    "tabular_regression": [
        "tabular regression state of the art",
        "gradient boosting tabular regression",
    ],
    "forecasting": [
        "time series forecasting state of the art",
        "deep learning forecasting benchmarks",
    ],
    "anomaly_detection": [
        "anomaly detection time series state of the art",
        "deep anomaly detection unsupervised",
    ],
    "predictive_maintenance": [
        "survival analysis machine learning",
        "deep learning predictive maintenance",
    ],
    "root_cause_attribution": [
        "feature attribution causal machine learning",
        "root cause analysis machine learning",
    ],
}


_FAMILY_QUERY_HINTS: dict[str, list[str]] = {
    "linear": ["sparse linear models", "elastic net regularization"],
    "tree": ["random forest tabular benchmark"],
    "boosted_tree": ["xgboost catboost lightgbm benchmark", "focal loss boosted trees"],
    "neural": ["FT-Transformer tabular", "MLP tabular benchmark", "TabNet tabular"],
    "ensemble": ["stacking blending tabular", "model ensembling techniques"],
    "rule_based": ["rule learning interpretable models"],
    "survival": ["random survival forest deep learning survival"],
    "anomaly": ["autoencoder anomaly detection", "isolation forest deep anomaly"],
    "forecasting_classical": ["ETS theta prophet forecasting"],
    "forecasting_neural": ["N-BEATS NHITS DLinear forecasting"],
}


@dataclass
class PaperHit:
    """A single normalized search result. Kept small so payloads passed
    between agents stay light."""

    url: str
    title: str
    abstract_snippet: str = ""
    technique_name: str = ""
    implementable_summary: str = ""
    year: Optional[int] = None
    score: float = 0.0
    raw_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "abstract_snippet": self.abstract_snippet,
            "technique_name": self.technique_name,
            "implementable_summary": self.implementable_summary,
            "year": self.year,
            "score": self.score,
            "raw_source": self.raw_source,
        }


def _maybe_year_filter(year_min: Optional[int]) -> str:
    if year_min is None or year_min < 1990:
        return ""
    return f" {year_min}..2030"


def build_sota_queries(
    capability_key: str,
    technique_family: Optional[str] = None,
    problem_signature: Optional[str] = None,
    *,
    domain_key: Optional[str] = None,
    year_min: Optional[int] = _DEFAULT_YEAR_MIN,
    k: int = 5,
) -> list[str]:
    """Return up to `k` web-search queries the literature agent should run.

    Queries combine capability hints, family hints, and (when present and
    non-generic) domain. Each query targets arxiv first, with one
    benchmark-flavored query appended.
    """
    cap_hints = list(_CAPABILITY_QUERY_HINTS.get(capability_key, []))
    fam_hints = list(_FAMILY_QUERY_HINTS.get(technique_family or "", []))
    if not cap_hints and not fam_hints:
        cap_hints = [f"{capability_key.replace('_', ' ')} state of the art"]
    domain_part = ""
    if domain_key and domain_key.lower() not in {"", "general"}:
        domain_part = f" {domain_key}"
    sig_part = f" ({problem_signature})" if problem_signature else ""
    yr = _maybe_year_filter(year_min)

    out: list[str] = []
    seen: set[str] = set()
    for h in cap_hints + fam_hints:
        q = f"{h}{domain_part}{sig_part} site:arxiv.org{yr}".strip()
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= k:
            break
    if len(out) < k:
        bench = f"{capability_key.replace('_', ' ')} benchmark{domain_part}{yr}".strip()
        if bench not in seen:
            out.append(bench)
    return out[:k]


_YEAR_RX = re.compile(r"(?:19|20)\d{2}")
_ARXIV_RX = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+)")


def _extract_year(text: str) -> Optional[int]:
    m = _YEAR_RX.search(text or "")
    if not m:
        return None
    try:
        y = int(m.group(0))
        return y if 1990 <= y <= 2099 else None
    except ValueError:
        return None


def _technique_name_from_title(title: str) -> str:
    """Best-effort technique label from the title — the part before a colon
    or the first dash. Used as a short pointer in `prior_evidence`."""
    t = (title or "").strip()
    for sep in (":", " — ", " - ", " – "):
        if sep in t:
            return t.split(sep, 1)[0].strip()
    return t[:80]


def parse_search_hits(raw_results: list[dict[str, Any]]) -> list[PaperHit]:
    """Normalize a list of raw `WebSearch` result dicts into `PaperHit`s.

    Tolerates schema variation across providers — looks at common keys
    (`url`, `link`, `title`, `snippet`, `description`, `body`).
    """
    out: list[PaperHit] = []
    for r in raw_results or []:
        url = (r.get("url") or r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or r.get("description") or r.get("body") or "").strip()
        if not url or not title:
            continue
        year = _extract_year(snippet) or _extract_year(title)
        out.append(
            PaperHit(
                url=url,
                title=title[:240],
                abstract_snippet=snippet[:600],
                technique_name=_technique_name_from_title(title),
                year=year,
                score=float(r.get("score", 0.0)),
                raw_source=(r.get("source") or r.get("provider") or ""),
            )
        )
    return out


def shortlist_hits(
    hits: list[PaperHit],
    *,
    k: int = 3,
    require_arxiv: bool = False,
) -> list[PaperHit]:
    """Pick the top `k` hits, preferring arxiv URLs when `require_arxiv`."""
    pool = list(hits)
    if require_arxiv:
        pool = [h for h in pool if "arxiv.org" in (h.url or "")]
    pool.sort(key=lambda h: (h.year or 0, h.score), reverse=True)
    return pool[:k]


__all__ = [
    "PaperHit",
    "build_sota_queries",
    "parse_search_hits",
    "shortlist_hits",
]
