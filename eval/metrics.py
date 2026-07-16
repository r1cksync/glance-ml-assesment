"""Pure ranking metrics for retrieval evaluation.

All functions are deterministic, dependency-free and side-effect-free so they
can be unit-tested with exact values (tests/test_metrics.py). Ranked lists are
deduplicated (first occurrence wins) before scoring, so an engine that
accidentally returns the same image twice is neither rewarded nor punished
twice for it.
"""
from __future__ import annotations

import math
from typing import Sequence

__all__ = ["recall_at_k", "mrr", "ndcg_at_k", "mean_metrics"]


def _unique(ranked: Sequence[str]) -> list[str]:
    """Preserve order, drop repeats (first occurrence wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for image_id in ranked:
        if image_id not in seen:
            seen.add(image_id)
            out.append(image_id)
    return out


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """|top-k ∩ relevant| / |relevant|.

    Returns 0.0 when there are no relevant ids or k <= 0. k larger than the
    ranked list simply scores whatever was returned.
    """
    rel = set(relevant)
    if not rel or k <= 0:
        return 0.0
    top = _unique(ranked)[:k]
    hits = sum(1 for image_id in top if image_id in rel)
    return hits / len(rel)


def mrr(relevant: set[str], ranked: list[str]) -> float:
    """Reciprocal rank of the first relevant id; 0.0 if none appears."""
    rel = set(relevant)
    if not rel:
        return 0.0
    for rank, image_id in enumerate(_unique(ranked), start=1):
        if image_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Binary-gain nDCG@k.

    DCG = sum of 1/log2(rank+1) over relevant hits within the top-k;
    normalized by the ideal DCG (all min(|relevant|, k) relevant ids packed at
    the top). Returns 0.0 when there are no relevant ids or k <= 0; a perfect
    prefix scores exactly 1.0.
    """
    rel = set(relevant)
    if not rel or k <= 0:
        return 0.0
    top = _unique(ranked)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, image_id in enumerate(top, start=1)
        if image_id in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Column-wise mean over per-query metric dicts.

    Rows may have heterogeneous keys; a key's mean averages only the rows that
    define it. Returns {} for an empty input.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in per_query:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}
