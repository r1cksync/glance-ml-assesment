"""Latency benchmark and the 1M-image scaling projection.

Measures p50/p95/mean end-to-end query latency of any ``search_fn`` at the
current index size, and computes (not guesses) the memory / FLOPs math for a
synthetic 1M-image corpus: flat vs HNSW vs IVF-PQ index sizes, brute-force vs
graph-search distance evaluations, and when to leave FAISS-flat for
HNSW / OpenSearch / Atlas.

Writes eval/results/latency.md (+ latency.json for the raw numbers).

CLI:
    python -m eval.latency --config config/default.yaml
A persisted index is needed for live measurements; without one the report is
still written with the computed scaling projection.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

from core.config import load_config, resolve
from eval.harness import SearchFn

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── measurement ──────────────────────────────────────────────────────────────

def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile over an ascending-sorted sample."""
    if not sorted_vals:
        return 0.0
    idx = max(0, math.ceil(pct / 100.0 * len(sorted_vals)) - 1)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def measure_latency(
    search_fn: SearchFn,
    queries: Sequence[Mapping[str, str] | str],
    n_runs: int = 20,
    warmup: int = 3,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Wall-clock latency of search_fn cycled over the queries.

    Warmup runs (model lazy-loading, caches) are excluded from the sample.
    Returns {"p50_ms", "p95_ms", "mean_ms"} (nearest-rank percentiles).
    """
    texts = [q["text"] if isinstance(q, Mapping) else str(q) for q in queries]
    if not texts:
        raise ValueError("measure_latency needs at least one query")
    for i in range(max(0, int(warmup))):
        search_fn(texts[i % len(texts)], k)
    samples: list[float] = []
    for i in range(max(1, int(n_runs))):
        start = time.perf_counter()
        search_fn(texts[i % len(texts)], k)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "p50_ms": round(_percentile(samples, 50.0), 2),
        "p95_ms": round(_percentile(samples, 95.0), 2),
        "mean_ms": round(sum(samples) / len(samples), 2),
    }


# ── scaling math ─────────────────────────────────────────────────────────────

def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def scaling_projection(
    n_images: int = 1_000_000,
    regions_per_image: float = 4.0,
    dim: int = 512,
    *,
    hnsw_m: int = 32,
    ef_search: int = 128,
    pq_m: int = 64,
    nlist: int = 4096,
    cpu_gflops: float = 20.0,
) -> str:
    """Markdown section with the computed 1M-image index-size and query-cost math.

    Assumptions are parameters so the projection can be recomputed for any
    corpus shape. cpu_gflops is the effective single-core BLAS throughput used
    for the time estimates.
    """
    n_vec = int(n_images * regions_per_image)
    flat_garment = n_vec * dim * 4                       # float32
    flat_scene = n_images * dim * 4
    hnsw_total = flat_garment * 1.5 + n_vec * hnsw_m * 8  # ~1.5x vectors + link lists
    pq_total = n_vec * pq_m + n_vec * 8 + nlist * dim * 4  # codes + ids + centroids

    brute_flops = 2.0 * n_vec * dim                      # multiply+add per component
    hnsw_evals = ef_search * math.log2(max(2, n_vec))
    hnsw_flops = hnsw_evals * 2.0 * dim
    brute_ms = brute_flops / (cpu_gflops * 1e9) * 1e3
    hnsw_ms = hnsw_flops / (cpu_gflops * 1e9) * 1e3

    probes = 3  # typical parsed query: ~2 garment term vectors + 1 scene vector
    flat_p95 = probes * brute_ms * 1.5   # 1.5x tail padding over the mean
    hnsw_p95 = max(1.0, probes * hnsw_ms * 5)  # graph hop = pointer chasing, not BLAS

    compression = flat_garment / max(1, pq_total)

    return "\n".join([
        f"## Scaling projection — {n_images:,} images (computed, not measured)",
        "",
        f"Assumptions: {regions_per_image:g} garment regions/image → {n_vec:,} garment "
        f"vectors (+ {n_images:,} scene vectors), dim={dim}, float32; "
        f"~{cpu_gflops:g} GFLOP/s effective single-core CPU.",
        "",
        "### Index size (garment channel — the big one)",
        "",
        "| structure | size | math |",
        "| --- | --- | --- |",
        f"| FAISS flat (exact IP) | {_gb(flat_garment)} | {n_vec:,} × {dim} × 4 B |",
        f"| + scene flat index | {_gb(flat_scene)} | {n_images:,} × {dim} × 4 B |",
        f"| HNSW (M={hnsw_m}) | {_gb(hnsw_total)} | 1.5× vector storage + "
        f"{hnsw_m}×8 B links/vec ({_gb(n_vec * hnsw_m * 8)}) |",
        f"| IVF-PQ (m={pq_m}) | {_gb(pq_total)} | {pq_m} B codes + 8 B ids per vec + "
        f"{nlist:,}×{dim}×4 B centroids (~{compression:.0f}× smaller than flat) |",
        "",
        "### Query cost (per probe vector, garment channel)",
        "",
        "| method | distance evals | FLOPs | est. CPU time |",
        "| --- | --- | --- | --- |",
        f"| brute-force flat | {n_vec:,} | {brute_flops / 1e9:.2f} GFLOP | ~{brute_ms:.0f} ms |",
        f"| HNSW (ef={ef_search}) | ~{hnsw_evals:,.0f} (ef × log2 n) | "
        f"{hnsw_flops / 1e6:.1f} MFLOP | ~{hnsw_ms:.2f} ms math + graph overhead ≈ 1–3 ms |",
        "",
        f"A typical parsed query makes ~{probes} ANN probes (garment terms + scene). "
        f"Projected p95 for the vector stage alone: flat ≈ {flat_p95:,.0f} ms "
        f"(unacceptable at 1M), HNSW ≈ {hnsw_p95:.0f}–10 ms. End-to-end p95 is then "
        "dominated by query text encoding (~10–30 ms CPU) and the optional BLIP-ITM "
        "rerank (~100+ ms/image on CPU — cap `rerank.top_n` or serve int8-ONNX/GPU).",
        "",
        "### Recommendation",
        "",
        "| regime | index | why |",
        "| --- | --- | --- |",
        "| ≤ ~0.5M vectors (≈100K images) | FAISS `flat_ip` (current config) | "
        f"≤ ~{0.5e6 * dim * 4 / 1e9:.1f} GB, exact recall, brute force stays "
        "low-ms with BLAS batching — zero ops |",
        f"| 0.5M–50M vectors, one node | FAISS HNSW (M={hnsw_m}, ef≈{ef_search}) | "
        f"~{_gb(hnsw_total)} RAM at {n_vec / 1e6:.0f}M vecs, ms-level p95, "
        "recall ≥ 0.95 typical |",
        f"| RAM-bound single node | FAISS IVF-PQ (m={pq_m}) | {_gb(pq_total)} vs "
        f"{_gb(flat_garment)} (~{compression:.0f}×); recall loss is mitigated by our "
        "rerank stage |",
        "| multi-node / HA / filtered ANN at scale | OpenSearch k-NN or Atlas Vector "
        "Search | sharding + replication + metadata-prefilter pushdown next to the "
        "vectors; managed ops (both already behind our VectorStore ABC) |",
        "",
        f"At {n_images:,} images the attribute payloads (~1 KB each ≈ "
        f"{_gb(n_images * 1024)}) should also move from in-process dicts to "
        "DynamoDB/OpenSearch documents so the prefilter becomes an indexed query.",
    ])


# ── reporting ────────────────────────────────────────────────────────────────

def write_report(
    measurements: Mapping[str, Mapping[str, float]],
    *,
    results_dir: str | Path | None = None,
    context: str = "",
    projection: str | None = None,
) -> Path:
    """Write eval/results/latency.md (+ latency.json). Returns the md path."""
    rd = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    rd.mkdir(parents=True, exist_ok=True)
    md_path = rd / "latency.md"
    json_path = rd / "latency.json"

    parts: list[str] = ["# Latency benchmark", ""]
    if context:
        parts += [context, ""]
    if measurements:
        parts += [
            "| variant | p50 (ms) | p95 (ms) | mean (ms) |",
            "| --- | --- | --- | --- |",
        ]
        for name, m in measurements.items():
            parts.append(
                f"| {name} | {m.get('p50_ms', 0.0):.2f} | "
                f"{m.get('p95_ms', 0.0):.2f} | {m.get('mean_ms', 0.0):.2f} |")
    else:
        parts.append(
            "_No live measurements — no loaded index. Run the indexer, then "
            "re-run `python -m eval.latency`._")
    parts += ["", projection if projection is not None else scaling_projection(), ""]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"measurements": dict(measurements), "context": context},
                  f, indent=2, ensure_ascii=False, default=str)
    log.info("wrote %s and %s", md_path, json_path)
    return md_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    """CLI: measure engine latency on the acceptance queries + write the report."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.latency",
        description="p50/p95 latency benchmark + 1M-image scaling projection.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--variants", nargs="*",
                        default=["regions+attr-filter", "full+rerank"],
                        help="ablation variants to time (default: engine with and without rerank)")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--projection-only", action="store_true",
                        help="skip live measurement; just write the scaling math")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    results_dir = resolve(cfg, "eval.results_dir") if cfg.get_path("eval.results_dir") else None

    measurements: dict[str, dict[str, float]] = {}
    context = ""
    if not args.projection_only:
        try:
            from eval.ablations import AblationRunner, load_store  # lazy: heavy path
            from eval.harness import load_queries

            store = load_store(cfg, args.store_dir)
            images_dir = (Path(args.images_dir) if args.images_dir
                          else resolve(cfg, "paths.images_dir"))
            runner = AblationRunner(cfg, store, images_dir)
            queries = load_queries(args.queries or resolve(cfg, "eval.queries_file"))
            try:
                context = (f"Measured at k={args.k}, n_runs={args.n_runs}, "
                           f"warmup={args.warmup}; store stats: `{store.stats()}`.")
            except Exception:  # noqa: BLE001 — stats are cosmetic
                context = f"Measured at k={args.k}, n_runs={args.n_runs}, warmup={args.warmup}."
            for name in args.variants:
                try:
                    fn = runner.variant(name)
                    measurements[name] = measure_latency(
                        fn, queries, n_runs=args.n_runs, warmup=args.warmup, k=args.k)
                    log.info("%s: %s", name, measurements[name])
                except Exception:  # noqa: BLE001 — time whatever variants are available
                    log.exception("latency measurement failed for variant %s", name)
        except Exception:  # noqa: BLE001 — projection is still worth writing
            log.exception("could not build the engine for live measurement — "
                          "writing the scaling projection only")

    write_report(measurements, results_dir=results_dir, context=context)


if __name__ == "__main__":
    main()
