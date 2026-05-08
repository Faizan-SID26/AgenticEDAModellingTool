"""`/init` implementation: profile data files, read domain documents,
propose joins.

Pure inspection. No questions asked. Reads every file under
`<project>/data/` (csv, parquet, jsonl) and profiles columns, then
*also* reads supported domain documents (md/txt/pdf/docx/rtf) and
extracts their text. Writes:

    memory/INIT_PROFILE.json     — column profile + proposed joins
    memory/DOMAIN_DOCS.md        — extracted text from PUDs/specs/SOPs
    results/init_report.md       — human-readable summary

The agent (planner role) consumes all three at /plan time so it can
ask precise, process-aware questions instead of generic ones.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from lib import __version__
from lib.documents import collect as collect_documents, write_domain_docs

_log = logging.getLogger("eda.inspect")


_TIME_HINTS = ("time", "ts", "date", "datetime", "timestamp", "created_at", "event_at", "_dt")
_TARGET_HINTS = (
    "target",
    "label",
    "y",
    "defect",
    "fail",
    "outcome",
    "yield",
    "demand",
    "sales",
    "is_",
    "_flag",
)
# Generic identifier-name detection. Designed to catch id-shaped columns
# across domains (manufacturing, retail, finance, healthcare, …) without
# leaking domain-specific vocabulary. Detection runs on a tokenized form of
# the name so snake_case, kebab-case, "Spaced Names", and CamelCase all work
# (e.g. `SerialNumber` → tokens [`serial`, `number`]).
_ID_TOKENS: frozenset[str] = frozenset(
    {"id", "uuid", "key", "code", "serial", "idx", "index", "lot", "batch", "sku", "hash", "token"}
)
# Compound suffix / numeric-tail patterns that the tokenizer can't catch on
# its own. Matched case-insensitively against the raw name.
_ID_COMPOUND_PATTERNS: tuple[str, ...] = (
    r"_run_idx$",
    r"^.*_[0-9]{6,}$",
)
_ID_COMPOUND_RX = tuple(re.compile(p, flags=re.IGNORECASE) for p in _ID_COMPOUND_PATTERNS)
# Legacy alias retained so any external import keeps working.
_ID_HINTS = tuple(sorted(_ID_TOKENS))


def _tokenize_column_name(name: str) -> list[str]:
    """Lowercase token list, splitting on underscore, hyphen, whitespace, and
    camelCase boundaries. Used by both `_matches_id_pattern` and the join
    proposer."""
    # Insert separator at lowercase/digit → Uppercase boundary.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"[\s\-]+", "_", s).lower()
    return [tok for tok in s.split("_") if tok]


def _read_file(path: Path) -> pd.DataFrame:
    """Read a single file (csv/parquet/jsonl) into a DataFrame."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported file type: {path.suffix} ({path})")


def _column_summary(s: pd.Series) -> dict[str, Any]:
    """Summarize one column for the INIT profile."""
    n = len(s)
    n_missing = int(s.isna().sum())
    if pd.api.types.is_datetime64_any_dtype(s) or _looks_like_datetime(s):
        dtype = "datetime"
    elif pd.api.types.is_bool_dtype(s):
        dtype = "boolean"
    elif pd.api.types.is_numeric_dtype(s):
        dtype = "numeric"
    elif pd.api.types.is_string_dtype(s) or s.dtype == object:
        # heuristic: low-cardinality strings → categorical, otherwise text
        nunq = int(s.nunique(dropna=True))
        dtype = "categorical" if nunq <= max(50, int(n**0.5)) else "text"
    else:
        dtype = "categorical"

    out: dict[str, Any] = {
        "name": s.name,
        "dtype": dtype,
        "n_rows": n,
        "n_missing": n_missing,
        "n_unique": int(s.nunique(dropna=True)),
    }
    if dtype == "numeric":
        try:
            out["min"] = float(s.min())
            out["max"] = float(s.max())
            out["mean"] = float(s.mean())
            out["stdev"] = float(s.std())
        except Exception:  # noqa: BLE001
            pass
    elif dtype == "categorical":
        vc = s.value_counts(dropna=True).head(5)
        out["top_categories"] = [(str(k), int(v)) for k, v in vc.items()]
    elif dtype == "datetime":
        try:
            out["min"] = str(pd.to_datetime(s, errors="coerce").min())
            out["max"] = str(pd.to_datetime(s, errors="coerce").max())
        except Exception:  # noqa: BLE001
            pass
    return out


def _looks_like_datetime(s: pd.Series) -> bool:
    """Heuristic: does this look like a parseable datetime column?"""
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return False
    sample = s.dropna().head(20)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        return float(parsed.notna().mean()) > 0.9
    except Exception:  # noqa: BLE001
        return False


def _is_likely(name: str, hints: tuple[str, ...]) -> bool:
    """Substring (lowercase) match against any hint."""
    nl = name.lower()
    return any(h in nl for h in hints)


def _matches_id_pattern(name: str) -> bool:
    """Return True if `name` looks like a generic identifier (id/uuid/serial/
    idx/index/lot/batch/sku/hash/token/key/code/digit-suffix). Tokenizes the
    name to handle snake_case, kebab-case, spaces, and camelCase uniformly."""
    if any(rx.search(name) for rx in _ID_COMPOUND_RX):
        return True
    return any(tok in _ID_TOKENS for tok in _tokenize_column_name(name))


def _infer_likely_columns(cols: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group columns into likely-time / likely-target / likely-id buckets."""
    likely_time = [c["name"] for c in cols if c["dtype"] == "datetime" or _is_likely(c["name"], _TIME_HINTS)]
    likely_target = [c["name"] for c in cols if _is_likely(c["name"], _TARGET_HINTS)]
    # Identifier detection: name pattern AND high cardinality (≥95% unique).
    # The 95% threshold catches near-unique identifiers without requiring
    # perfect uniqueness, which can be broken by NULL handling or duplicates.
    likely_id = [
        c["name"]
        for c in cols
        if _matches_id_pattern(c["name"])
        and c["n_unique"] >= max(1, int(0.95 * c["n_rows"]))
    ]
    return {
        "likely_time": likely_time,
        "likely_target": likely_target,
        "likely_id": likely_id,
    }


def _propose_joins(file_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Propose joins between any two files sharing column names with high overlap."""
    proposals: list[dict[str, Any]] = []
    for i in range(len(file_summaries)):
        for j in range(i + 1, len(file_summaries)):
            a = file_summaries[i]
            b = file_summaries[j]
            cols_a = {c["name"] for c in a["columns"]}
            cols_b = {c["name"] for c in b["columns"]}
            shared = sorted(cols_a & cols_b)
            keylike = [c for c in shared if _matches_id_pattern(c) or _is_likely(c, _TIME_HINTS)]
            if not keylike:
                continue
            proposals.append(
                {
                    "left_table": a["table_name"],
                    "right_table": b["table_name"],
                    "on": keylike,
                    "how": "inner",
                    "shared_columns": shared,
                    "rationale": (
                        f"{a['table_name']} and {b['table_name']} share key-like columns: {keylike}"
                    ),
                }
            )
    return proposals


def inspect_project(
    project_dir: Path,
    *,
    sample_rows: int = 100_000,
    write: bool = True,
) -> dict[str, Any]:
    """Walk `<project>/data/`, profile each file, write the init report.

    Returns the in-memory profile dict; if ``write`` is True, also persists
    `memory/INIT_PROFILE.json` and `results/init_report.md`.
    """
    project_dir = Path(project_dir).resolve()
    data_dir = project_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"no data directory at {data_dir}")

    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".parquet", ".jsonl", ".ndjson", ".json")
    )

    file_summaries: list[dict[str, Any]] = []
    for path in files:
        try:
            df = _read_file(path)
        except Exception as e:  # noqa: BLE001 — skip unreadable files but record them
            _log.warning("could not read %s: %s", path, e)
            file_summaries.append(
                {"table_name": path.stem, "path": str(path), "error": str(e), "columns": []}
            )
            continue
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=0)
        cols = [_column_summary(df[c]) for c in df.columns]
        likely = _infer_likely_columns(cols)
        file_summaries.append(
            {
                "table_name": path.stem,
                "path": str(path.relative_to(project_dir)),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
                "columns": cols,
                **likely,
            }
        )

    proposed_joins = _propose_joins(file_summaries)

    # Collect domain documents (PUDs, specs, SOPs, prior investigations)
    # so /plan can build process-aware questions instead of generic ones.
    corpus = collect_documents(data_dir)
    docs_summary = [
        {
            "path": d.path,
            "suffix": d.suffix,
            "n_chars": d.n_chars,
            "truncated": d.truncated,
            "error": d.error,
        }
        for d in corpus.documents
    ]

    profile: dict[str, Any] = {
        "schema_version": "1",
        "framework_version": __version__,
        "project_dir": str(project_dir),
        "n_files": len(file_summaries),
        "files": file_summaries,
        "proposed_joins": proposed_joins,
        "domain_documents": docs_summary,
        "domain_documents_parser_warnings": list(corpus.parser_warnings),
        "n_domain_documents": len(corpus.documents),
        "n_domain_documents_with_text": sum(1 for d in corpus.documents if d.text),
    }

    if write:
        memdir = project_dir / "memory"
        memdir.mkdir(parents=True, exist_ok=True)
        (memdir / "INIT_PROFILE.json").write_text(
            json.dumps(profile, indent=2, default=str), encoding="utf-8"
        )
        # memory/DOMAIN_DOCS.md (only if there were any documents).
        write_domain_docs(project_dir, corpus)

        results_dir = project_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "init_report.md").write_text(
            _render_report_md(profile), encoding="utf-8"
        )
    return profile


def _render_report_md(profile: dict[str, Any]) -> str:
    """Pretty-print the INIT profile as a markdown report."""
    lines: list[str] = []
    lines.append(f"# /init report\n")
    lines.append(f"_Project_: `{profile.get('project_dir', '')}`\n")
    lines.append(f"_Framework version_: `{profile.get('framework_version')}`\n")
    lines.append(f"\n## Files ({profile['n_files']})\n")
    for f in profile["files"]:
        lines.append(f"### {f['table_name']}\n")
        if "error" in f:
            lines.append(f"- error: `{f['error']}`\n")
            continue
        lines.append(f"- path: `{f.get('path')}`")
        lines.append(f"- rows: {f.get('n_rows')}  cols: {f.get('n_columns')}")
        if f.get("likely_time"):
            lines.append(f"- likely time columns: `{f['likely_time']}`")
        if f.get("likely_target"):
            lines.append(f"- likely target columns: `{f['likely_target']}`")
        if f.get("likely_id"):
            lines.append(f"- likely id columns: `{f['likely_id']}`")
        lines.append("")
        lines.append("| column | dtype | n_rows | n_missing | n_unique |")
        lines.append("|---|---|---|---|---|")
        for c in f["columns"]:
            lines.append(
                f"| `{c['name']}` | {c['dtype']} | {c['n_rows']} | {c['n_missing']} | {c['n_unique']} |"
            )
        lines.append("")
    if profile["proposed_joins"]:
        lines.append("\n## Proposed joins\n")
        for jp in profile["proposed_joins"]:
            lines.append(
                f"- {jp['left_table']} ⨝ {jp['right_table']} on `{jp['on']}` ({jp['how']})"
            )
            lines.append(f"  - rationale: {jp['rationale']}")
    docs = profile.get("domain_documents") or []
    if docs:
        lines.append(f"\n## Domain documents ({len(docs)})\n")
        for d in docs:
            chars = d.get("n_chars", 0)
            err = d.get("error")
            note = f" (parse error: `{err}`)" if err else ""
            trunc = " (truncated)" if d.get("truncated") else ""
            lines.append(f"- `{d.get('path')}` — {d.get('suffix')} — {chars} chars{trunc}{note}")
        warns = profile.get("domain_documents_parser_warnings") or []
        if warns:
            lines.append("")
            lines.append("Parser warnings (install the named extra to read these):")
            for w in warns:
                lines.append(f"- `{w}`")
        lines.append(
            "\n_Extracted text is in `memory/DOMAIN_DOCS.md`. The planner reads it before asking questions._"
        )
    else:
        lines.append("\n## Domain documents\n")
        lines.append(
            "_No supported documents in `data/`. Drop a `.md`, `.txt`, `.pdf`, "
            "`.docx`, or `.rtf` file (PUD / spec / SOP / prior investigation) "
            "and re-run `/init` to give the planner domain context._"
        )
    return "\n".join(lines) + "\n"
