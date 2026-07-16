"""Engine-agnostic evaluation harness.

Everything here works against a plain ``search_fn(query_text, k) -> [image_id]``
callable, so the same harness scores the real SearchEngine, the whole-image
CLIP baselines, or any future system, without importing any of them.

Inputs:
- queries yaml (eval/queries.yaml): {queries: [{id, text, notes}, ...]}
- labels  yaml (eval/labels.yaml):  {labels: {q1: [image_id, ...], ...}}

Outputs: per-query Recall@k / MRR / nDCG metrics plus their means, and
json + markdown reports under eval/results/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from eval.metrics import mean_metrics, mrr, ndcg_at_k, recall_at_k

log = logging.getLogger(__name__)

#: The contract every evaluated system implements: (query_text, k) -> ranked image ids.
SearchFn = Callable[[str, int], list[str]]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── loading ──────────────────────────────────────────────────────────────────

def load_queries(path: str | Path) -> list[dict[str, str]]:
    """Load the acceptance queries. Returns [{id, text, notes}, ...]."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("queries") or []
    queries: list[dict[str, str]] = []
    for i, q in enumerate(raw):
        if not isinstance(q, Mapping) or not q.get("id") or not q.get("text"):
            raise ValueError(f"{p}: queries[{i}] must have non-empty 'id' and 'text'")
        queries.append({
            "id": str(q["id"]),
            "text": str(q["text"]),
            "notes": str(q.get("notes", "")),
        })
    if not queries:
        raise ValueError(f"{p}: no queries found under top-level 'queries' key")
    return queries


def load_labels(path: str | Path) -> dict[str, set[str]]:
    """Load hand-labeled relevance sets: {query_id: {image_id, ...}}.

    The labels file is produced by a human pass over the candidate pool; if it
    does not exist yet, the error explains how to bootstrap it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Labels file not found: {p}\n"
            "Bootstrap the labeling workflow first:\n"
            "    python -m eval.label_pool --config config/default.yaml\n"
            "which writes eval/label_pool.json (the candidate pool a human should "
            "label) and eval/labels.auto.yaml (WEAK attribute-derived auto-labels). "
            "Confirm / correct the auto labels against the pool, then save the "
            f"result as {p} in the form:\n"
            "    labels:\n"
            "      q1: [image_a, image_b]\n"
            "      q2: [image_c]\n"
        )
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("labels") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{p}: top-level 'labels' must be a mapping of query_id -> [image_id]")
    return {str(qid): {str(i) for i in (ids or [])} for qid, ids in raw.items()}


# ── running ──────────────────────────────────────────────────────────────────

def run_eval(
    search_fn: SearchFn,
    queries: Sequence[Mapping[str, str]],
    labels: Mapping[str, set[str]],
    k_values: Sequence[int] = (1, 5, 10),
) -> dict:
    """Score one system over all queries.

    Per query: recall@k for each k in k_values, MRR, and nDCG@max(k) — all
    computed over a single retrieval at depth max(k_values). Queries with no
    labels are still run (their rankings are recorded) but excluded from the
    means so unlabeled queries cannot silently drag scores to zero.
    """
    ks = sorted({int(k) for k in k_values})
    if not ks or ks[0] < 1:
        raise ValueError(f"k_values must be positive ints, got {k_values!r}")
    max_k = ks[-1]

    per_query: list[dict] = []
    scored_rows: list[dict[str, float]] = []
    for q in queries:
        qid, text = str(q["id"]), str(q["text"])
        ranked = [str(r) for r in search_fn(text, max_k)]
        relevant = set(labels.get(qid, set()))
        metrics: dict[str, float] = {f"recall@{k}": recall_at_k(relevant, ranked, k) for k in ks}
        metrics["mrr"] = mrr(relevant, ranked)
        metrics[f"ndcg@{max_k}"] = ndcg_at_k(relevant, ranked, max_k)
        row = {
            "id": qid,
            "text": text,
            "notes": str(q.get("notes", "")),
            "labeled": bool(relevant),
            "n_relevant": len(relevant),
            "ranked": ranked,
            "metrics": metrics,
        }
        per_query.append(row)
        if relevant:
            scored_rows.append(metrics)
        else:
            log.warning("query %s has no relevance labels — excluded from means", qid)

    return {
        "k_values": ks,
        "per_query": per_query,
        "means": mean_metrics(scored_rows),
        "n_queries": len(per_query),
        "n_labeled": len(scored_rows),
    }


# ── reporting ────────────────────────────────────────────────────────────────

def render_markdown_table(
    rows: Sequence[Mapping[str, object]],
    headers: Sequence[str] | None = None,
) -> str:
    """Render dict rows as a GitHub-flavored markdown table.

    Column order comes from `headers` (or first-seen key order). Floats are
    formatted to 4 decimals; missing cells render empty.
    """
    if not rows:
        return "_(no rows)_"
    if headers is None:
        cols: list[str] = []
        for row in rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
    else:
        cols = list(headers)

    def fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(" --- " for _ in cols) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _results_markdown(results: Mapping, name: str) -> str:
    """Per-query metrics table + means row for one run_eval() output."""
    metric_keys: list[str] = []
    rows: list[dict[str, object]] = []
    for row in results.get("per_query", []):
        metrics = row.get("metrics", {})
        for key in metrics:
            if key not in metric_keys:
                metric_keys.append(key)
        rows.append({
            "query": row.get("id", ""),
            "category": row.get("notes", ""),
            "labeled": "yes" if row.get("labeled") else "no",
            **metrics,
        })
    means = results.get("means", {})
    rows.append({"query": "**mean**", "category": "", "labeled": "", **means})
    headers = ["query", "category", "labeled"] + metric_keys
    return (
        f"# Evaluation results — {name}\n\n"
        f"{render_markdown_table(rows, headers=headers)}\n\n"
        f"_Means are over the {results.get('n_labeled', 0)} labeled "
        f"of {results.get('n_queries', 0)} queries._\n"
    )


def save_results(
    results: Mapping,
    name: str = "eval",
    results_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write one run_eval() output as {name}.json and {name}.md under eval/results/.

    Returns (json_path, md_path).
    """
    rd = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    rd.mkdir(parents=True, exist_ok=True)
    json_path = rd / f"{name}.json"
    md_path = rd / f"{name}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_results_markdown(results, name))
    log.info("saved results: %s, %s", json_path, md_path)
    return json_path, md_path
