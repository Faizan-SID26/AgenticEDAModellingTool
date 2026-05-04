"""L5: per-time-series-column SAX representation + matrix profile.

SAX is implemented inline (PAA + alphabet quantization). Matrix profile
is computed via `stumpy` if available; otherwise discords/motifs are
approximated via a simple sliding-window distance-to-mean criterion.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.schemas.sketch import L5TimeseriesSummary

_log = logging.getLogger("eda.sketch.l5")


def _paa(values: np.ndarray, word_length: int) -> np.ndarray:
    """Piecewise aggregate approximation: mean over `word_length` equal-sized chunks."""
    if values.size <= word_length:
        return values.copy()
    chunks = np.array_split(values, word_length)
    return np.array([float(c.mean()) for c in chunks])


def _sax_alphabet(paa: np.ndarray, alphabet_size: int) -> str:
    """Quantize PAA values into an alphabet. Uses gaussian-equiprobable breakpoints."""
    from scipy.stats import norm  # local import — scipy is heavy

    if paa.size == 0:
        return ""
    z = (paa - paa.mean()) / (paa.std(ddof=1) or 1.0)
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    letters = "abcdefghijklmnopqrstuvwxyz"[:alphabet_size]
    out = []
    for v in z:
        idx = int(np.searchsorted(breakpoints, v))
        out.append(letters[min(idx, alphabet_size - 1)])
    return "".join(out)


def _sax_window(values: np.ndarray, window: int, word_length: int, alphabet_size: int) -> list[str]:
    """Slide a window of size `window`, emit a SAX word per window."""
    out = []
    for i in range(0, len(values) - window + 1):
        out.append(_sax_alphabet(_paa(values[i : i + window], word_length), alphabet_size))
    return out


def _matrix_profile(values: np.ndarray, window: int) -> tuple[list[dict], list[dict]]:
    """Top motifs and discords. Use stumpy if available; else a fallback."""
    try:
        import stumpy  # type: ignore

        mp = stumpy.stump(values, m=window)
        # Distances are mp[:,0]. Smallest = motif (most similar match), largest = discord.
        dists = mp[:, 0].astype(float)
        motif_idx = list(map(int, np.argsort(dists)[:3]))
        discord_idx = list(map(int, np.argsort(dists)[::-1][:3]))
        motifs = [
            {"index": i, "distance": float(dists[i]), "match_index": int(mp[i, 1])}
            for i in motif_idx
        ]
        discords = [{"index": i, "distance": float(dists[i])} for i in discord_idx]
        return motifs, discords
    except ImportError:
        _log.debug("stumpy not available; using fallback matrix profile.")
    # Fallback: distance from each window's mean to global mean.
    if len(values) < window + 1:
        return [], []
    means = np.array([values[i : i + window].mean() for i in range(len(values) - window + 1)])
    dist = np.abs(means - means.mean())
    motif_idx = list(map(int, np.argsort(dist)[:3]))
    discord_idx = list(map(int, np.argsort(dist)[::-1][:3]))
    motifs = [{"index": i, "distance": float(dist[i])} for i in motif_idx]
    discords = [{"index": i, "distance": float(dist[i])} for i in discord_idx]
    return motifs, discords


def build_l5_for_column(
    s: pd.Series,
    *,
    window: int = 64,
    word_length: int = 8,
    alphabet_size: int = 8,
) -> L5TimeseriesSummary:
    """Build the L5 summary for a single time-series column."""
    values = s.dropna().astype(float).values
    name = str(s.name)
    if values.size < max(16, window):
        return L5TimeseriesSummary(
            column=name,
            sax_alphabet_size=alphabet_size,
            sax_word_length=word_length,
            matrix_profile_window=window,
        )
    words = _sax_window(values, window, word_length, alphabet_size)
    if words:
        ser = pd.Series(words)
        top = ser.value_counts().head(8)
        sax_top = [(str(k), int(v)) for k, v in top.items()]
    else:
        sax_top = []
    motifs, discords = _matrix_profile(values, window)
    return L5TimeseriesSummary(
        column=name,
        sax_alphabet_size=alphabet_size,
        sax_word_length=word_length,
        sax_top_motifs=sax_top,
        matrix_profile_window=window,
        matrix_profile_top_motifs=motifs,
        matrix_profile_top_discords=discords,
    )


def build_l5(
    df: pd.DataFrame,
    *,
    time_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    window: int = 64,
    word_length: int = 8,
    alphabet_size: int = 8,
) -> list[L5TimeseriesSummary]:
    """Build per-column L5 summaries for numeric columns (sorted by time if given)."""
    df_sorted = df
    if time_column and time_column in df.columns:
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
    num = df_sorted.select_dtypes(include=[np.number])
    cols = columns or [c for c in num.columns if c != time_column]
    summaries: list[L5TimeseriesSummary] = []
    for c in cols[:30]:  # cap on number of columns to avoid blowing past size budget
        if c not in num.columns:
            continue
        summaries.append(
            build_l5_for_column(num[c], window=window, word_length=word_length, alphabet_size=alphabet_size)
        )
    return summaries


def save_l5(summaries: list[L5TimeseriesSummary], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.model_dump() for s in summaries]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_l5(path: Path) -> list[L5TimeseriesSummary]:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [L5TimeseriesSummary.model_validate(r) for r in raw]
