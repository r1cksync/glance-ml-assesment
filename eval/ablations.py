"""The ablation ladder — the eval story of this system.

Each variant is exposed as a plain ``search_fn(query_text, k) -> [image_id]``
so eval.harness.run_eval stays engine-agnostic. Weakest to strongest:

  a) vanilla-clip-whole   whole-image openai/clip-vit-base-patch32, cosine
                          against the raw query text (the baseline the
                          assignment says is not acceptable).
  b) fashionclip-whole    same recipe with patrickjohncyh/fashion-clip.
  c) fashionclip-regions  the real engine, garment channel ONLY
                          (weights {garment:1, scene:0, attribute:0},
                          prefilter off, no reranker).
  d) regions+attr-filter  configured fusion weights + attribute prefilter,
                          no reranker.
  e) full+rerank          everything on, incl. the BLIP-ITM reranker.

Whole-image baseline matrices are cached under
``<artifacts_dir>/eval_cache/wholeimg_<variant>_<model>.npz`` (keyed by
variant + model id) so re-runs never re-embed the corpus.

Determinism note: eval always parses with the rule parser (free, ~1ms,
reproducible) regardless of the configured parsing backend.

CLI:
    python -m eval.ablations --config config/default.yaml
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from core.config import Config, load_config, resolve
from eval.harness import (
    SearchFn,
    load_labels,
    load_queries,
    render_markdown_table,
    run_eval,
)
from indexer.models.base import EmbeddingModel
from indexer.storage.base import VectorStore
from retriever.parsing.rule_parser import RuleParser
from retriever.rerank.base import Reranker
from retriever.search.engine import SearchEngine

log = logging.getLogger(__name__)

VARIANTS: tuple[str, ...] = (
    "vanilla-clip-whole",
    "fashionclip-whole",
    "fashionclip-regions",
    "regions+attr-filter",
    "full+rerank",
)

VARIANT_NOTES: dict[str, str] = {
    "vanilla-clip-whole": "whole-image CLIP ViT-B/32, cosine vs raw query text (baseline)",
    "fashionclip-whole": "whole-image FashionCLIP, cosine vs raw query text",
    "fashionclip-regions": "engine, garment-region channel only (no scene, no attributes, no prefilter, no rerank)",
    "regions+attr-filter": "engine with configured fusion weights + conjunctive attribute prefilter (no rerank)",
    "full+rerank": "full engine incl. BLIP-ITM cross-modal rerank",
}

#: whole-image variant -> config backend name (model ids resolved via config)
_WHOLE_IMAGE_BACKENDS: dict[str, str] = {
    "vanilla-clip-whole": "clip-vit-b32",   # openai/clip-vit-base-patch32
    "fashionclip-whole": "fashion-clip",    # patrickjohncyh/fashion-clip
}

#: backend -> default HF model id (mirrors config/default.yaml and
#: eval._embedders.DEFAULT_MODEL_IDS — duplicated here so cache-path
#: resolution never has to import the torch-heavy embedder module).
_DEFAULT_MODEL_IDS: dict[str, str] = {
    "clip-vit-b32": "openai/clip-vit-base-patch32",
    "fashion-clip": "patrickjohncyh/fashion-clip",
    "marqo-siglip": "Marqo/marqo-fashionSigLIP",
}

_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ── small path helpers ────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def resolve_image_path(images_dir: Path, image_id: str) -> Path:
    """Map an image_id to its file (id may or may not carry an extension)."""
    direct = images_dir / image_id
    if direct.is_file():
        return direct
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
    hits = sorted(images_dir.glob(f"{image_id}.*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no image file for id {image_id!r} under {images_dir}")


# ── best-effort composition (store / reranker) ───────────────────────────────

def _default_store_dir(cfg: Config) -> Path:
    artifacts = resolve(cfg, "paths.artifacts_dir")
    candidate = artifacts / "index"
    return candidate if candidate.exists() else artifacts


def _instantiate_store(cfg: Config) -> VectorStore:
    """Instantiate the FAISS store implementation directly (fallback path)."""
    last_err: Exception | None = None
    for modname in ("indexer.storage.faiss_store", "indexer.storage.faiss"):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001 — module may not exist yet
            last_err = e
            continue
        classes = [
            obj for _, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, VectorStore) and obj is not VectorStore
            and obj.__module__ == mod.__name__
        ]
        if not classes:
            continue
        cls = classes[0]
        garment_backend = str(cfg.get_path("embedding.garment_backend", "fashion-clip"))
        scene_backend = str(cfg.get_path("embedding.scene_backend", "clip-vit-b32"))
        gdim = int(cfg.get_path(f"embedding.models.{garment_backend}.dim", 512))
        sdim = int(cfg.get_path(f"embedding.models.{scene_backend}.dim", 512))
        for kwargs in ({}, {"garment_dim": gdim, "scene_dim": sdim}, {"dim": gdim}):
            try:
                store = cls(**kwargs)
                log.info("instantiated %s.%s(%s)", modname, cls.__name__, kwargs)
                return store
            except TypeError:
                continue
        last_err = RuntimeError(f"could not construct {cls.__name__} with known signatures")
    raise RuntimeError(
        "Could not build a VectorStore automatically. Either expose "
        "core.bootstrap.build_store(cfg) / indexer.storage.factory.build_store(cfg), "
        "or construct the store yourself and pass it to AblationRunner."
    ) from last_err


def load_store(cfg: Config, store_dir: str | Path | None = None) -> VectorStore:
    """Construct the configured VectorStore and load its persisted artifacts.

    Best-effort composition: prefers the project composition root
    (core.bootstrap.build_store / indexer.storage factories) when present,
    then falls back to instantiating the FAISS implementation directly. This
    keeps eval runnable while sibling packages evolve.
    """
    store: VectorStore | None = None
    for modname, attr in (
        ("core.bootstrap", "build_store"),
        ("indexer.storage.factory", "build_store"),
        ("indexer.storage", "build_store"),
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            log.debug("store builder module %s unavailable: %s", modname, e)
            continue
        builder = getattr(mod, attr, None)
        if builder is None:
            continue
        try:
            store = builder(cfg)
            log.info("built store via %s.%s", modname, attr)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("store builder %s.%s failed (%s) — trying next", modname, attr, e)
    if store is None:
        store = _instantiate_store(cfg)

    path = Path(store_dir) if store_dir is not None else _default_store_dir(cfg)
    store.load(path)
    log.info("loaded store %r from %s — stats: %s",
             getattr(store, "name", "?"), path, store.stats())
    return store


def _build_reranker(cfg: Config) -> Optional[Reranker]:
    """Best-effort BLIP-ITM reranker construction (bootstrap first, then direct)."""
    try:
        mod = importlib.import_module("core.bootstrap")
        builder = getattr(mod, "build_reranker", None)
        if builder is not None:
            return builder(cfg)
    except Exception as e:  # noqa: BLE001
        log.debug("core.bootstrap.build_reranker unavailable: %s", e)

    model_id = str(cfg.get_path("rerank.model_id", "Salesforce/blip-itm-base-coco"))
    device = str(cfg.get_path("rerank.device", cfg.get_path("embedding.device", "auto")))
    try:
        mod = importlib.import_module("retriever.rerank.blip_itm")
        classes = [
            obj for _, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, Reranker) and obj is not Reranker
            and obj.__module__ == mod.__name__
        ]
        for cls in classes:
            for kwargs in ({"model_id": model_id, "device": device},
                           {"model_id": model_id}, {}):
                try:
                    return cls(**kwargs)
                except TypeError:
                    continue
    except Exception as e:  # noqa: BLE001
        log.warning("BLIP-ITM reranker unavailable (%s)", e)
    return None


# ── the runner ───────────────────────────────────────────────────────────────

class AblationRunner:
    """Builds shared eval resources once (embedders, parser, engines, cached
    whole-image matrices) and hands out each ablation variant as a SearchFn.

    Construction is cheap; models load lazily on first use of a variant and
    are then reused across variants and queries.
    """

    def __init__(
        self,
        cfg: Config,
        store: VectorStore,
        images_dir: str | Path,
        *,
        results_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.images_dir = Path(images_dir)
        self.artifacts_dir = resolve(cfg, "paths.artifacts_dir")
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        elif cfg.get_path("eval.results_dir"):
            self.results_dir = resolve(cfg, "eval.results_dir")
        else:
            self.results_dir = Path(__file__).resolve().parent / "results"
        self._parser = RuleParser()
        self._embedders: dict[str, EmbeddingModel] = {}
        self._engines: dict[str, SearchEngine] = {}
        self._matrices: dict[str, tuple[list[str], np.ndarray]] = {}
        self._reranker: Optional[Reranker] = None
        self._reranker_tried = False

    # ── shared resources ─────────────────────────────────────────────────────

    def _embedder(self, backend: str) -> EmbeddingModel:
        if backend not in self._embedders:
            from eval._embedders import build_eval_embedder  # heavy: torch
            mcfg = self.cfg.get_path(f"embedding.models.{backend}", {}) or {}
            self._embedders[backend] = build_eval_embedder(
                backend,
                model_id=mcfg.get("model_id") or _DEFAULT_MODEL_IDS.get(backend),
                dim=mcfg.get("dim"),
                device=str(self.cfg.get_path("embedding.device", "auto")),
                batch_size=int(self.cfg.get_path("embedding.batch_size", 16)),
            )
        return self._embedders[backend]

    def _get_reranker(self) -> Optional[Reranker]:
        if not self._reranker_tried:
            self._reranker_tried = True
            self._reranker = _build_reranker(self.cfg)
            if self._reranker is None:
                log.warning("no reranker available — 'full+rerank' degenerates "
                            "to 'regions+attr-filter'")
        return self._reranker

    def _image_resolver(self, image_id: str) -> "object":
        from PIL import Image  # lazy
        path = resolve_image_path(self.images_dir, image_id)
        with Image.open(path) as im:
            return im.convert("RGB")

    # ── whole-image baselines (variants a, b) ───────────────────────────────

    def _whole_image_matrix(self, variant: str, backend: str) -> tuple[list[str], np.ndarray]:
        """[N, dim] whole-image matrix over the indexed corpus, npz-cached
        under artifacts keyed by variant + model id."""
        if variant in self._matrices:
            return self._matrices[variant]

        mcfg = self.cfg.get_path(f"embedding.models.{backend}", {}) or {}
        model_id = str(mcfg.get("model_id") or _DEFAULT_MODEL_IDS[backend])
        corpus = sorted(str(i) for i in self.store.image_ids())
        if not corpus:
            raise RuntimeError("store has no indexed images — run the indexer first")

        cache_path = (self.artifacts_dir / "eval_cache"
                      / f"wholeimg_{_slug(variant)}_{_slug(model_id)}.npz")
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                if [str(x) for x in data["corpus"].tolist()] == corpus:
                    ids = [str(x) for x in data["image_ids"].tolist()]
                    mat = np.asarray(data["vectors"], dtype=np.float32)
                    self._matrices[variant] = (ids, mat)
                    log.info("loaded cached whole-image matrix for %s: %s", variant, mat.shape)
                    return self._matrices[variant]
                log.info("whole-image cache %s stale (corpus changed) — rebuilding", cache_path)
            except Exception as e:  # noqa: BLE001 — a bad cache should never block eval
                log.warning("unreadable cache %s (%s) — rebuilding", cache_path, e)

        from PIL import Image  # lazy
        embedder = self._embedder(backend)
        batch_size = int(self.cfg.get_path("embedding.batch_size", 16))
        kept: list[str] = []
        chunks: list[np.ndarray] = []
        batch: list = []
        for image_id in corpus:
            try:
                path = resolve_image_path(self.images_dir, image_id)
            except FileNotFoundError:
                log.warning("image file missing for %s — excluded from %s", image_id, variant)
                continue
            with Image.open(path) as im:
                batch.append(im.convert("RGB"))
            kept.append(image_id)
            if len(batch) >= batch_size:
                chunks.append(embedder.embed_images(batch))
                batch = []
        if batch:
            chunks.append(embedder.embed_images(batch))
        if not chunks:
            raise RuntimeError(f"no images embeddable for {variant} from {self.images_dir}")
        mat = np.vstack(chunks).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, corpus=np.asarray(corpus),
                 image_ids=np.asarray(kept), vectors=mat)
        log.info("built whole-image matrix for %s: %s -> %s", variant, mat.shape, cache_path)
        self._matrices[variant] = (kept, mat)
        return self._matrices[variant]

    def whole_image_variant(self, variant: str) -> SearchFn:
        """Variants (a)/(b): cosine between raw query text and whole-image vectors."""
        backend = _WHOLE_IMAGE_BACKENDS[variant]
        ids, mat = self._whole_image_matrix(variant, backend)
        embedder = self._embedder(backend)

        def search_fn(query: str, k: int) -> list[str]:
            qvec = embedder.embed_text(query).astype(np.float32)
            sims = mat @ qvec
            order = np.argsort(-sims, kind="stable")[: max(0, int(k))]
            return [ids[i] for i in order]

        return search_fn

    # ── engine variants (c, d, e) ────────────────────────────────────────────

    def _build_engine(
        self,
        *,
        store: VectorStore,
        garment_backend: str,
        weights: Mapping[str, float],
        prefilter_enabled: bool,
        use_reranker: bool,
    ) -> SearchEngine:
        scene_backend = str(self.cfg.get_path("embedding.scene_backend", "clip-vit-b32"))
        pf = self.cfg.get_path("search.prefilter", {}) or {}
        return SearchEngine(
            store=store,
            garment_embedder=self._embedder(garment_backend),
            scene_embedder=self._embedder(scene_backend),
            parser=self._parser,
            reranker=self._get_reranker() if use_reranker else None,
            image_resolver=self._image_resolver,
            weights={str(ch): float(w) for ch, w in weights.items()},
            candidates_k=int(self.cfg.get_path("search.candidates_k", 50)),
            prefilter_enabled=bool(prefilter_enabled),
            prefilter_min_candidates=int(pf.get("min_candidates", 12)),
            prefilter_soft_fallback=bool(pf.get("soft_fallback", True)),
            rerank_top_n=int(self.cfg.get_path("rerank.top_n", 50)),
            rerank_weight=float(self.cfg.get_path("rerank.weight", 0.35)),
        )

    def _engine(self, variant: str) -> SearchEngine:
        if variant in self._engines:
            return self._engines[variant]
        garment_backend = str(self.cfg.get_path("embedding.garment_backend", "fashion-clip"))
        cfg_weights = self.cfg.get_path("search.weights", {}) or \
            {"garment": 0.55, "scene": 0.25, "attribute": 0.20}
        cfg_prefilter = bool(self.cfg.get_path("search.prefilter.enabled", True))

        if variant == "fashionclip-regions":
            engine = self._build_engine(
                store=self.store, garment_backend=garment_backend,
                weights={"garment": 1.0, "scene": 0.0, "attribute": 0.0},
                prefilter_enabled=False, use_reranker=False)
        elif variant == "regions+attr-filter":
            engine = self._build_engine(
                store=self.store, garment_backend=garment_backend,
                weights=cfg_weights, prefilter_enabled=cfg_prefilter,
                use_reranker=False)
        elif variant == "full+rerank":
            engine = self._build_engine(
                store=self.store, garment_backend=garment_backend,
                weights=cfg_weights, prefilter_enabled=cfg_prefilter,
                use_reranker=True)
        else:
            raise ValueError(f"unknown engine variant {variant!r}")
        self._engines[variant] = engine
        return engine

    def engine_variant(self, variant: str) -> SearchFn:
        engine = self._engine(variant)
        use_rerank = variant == "full+rerank"

        def search_fn(query: str, k: int) -> list[str]:
            _, results = engine.search(query, k=int(k), use_rerank=use_rerank)
            return [r.image_id for r in results]

        return search_fn

    # ── dispatch ─────────────────────────────────────────────────────────────

    def variant(self, name: str) -> SearchFn:
        """Return the search_fn for one ablation variant by its ladder name."""
        if name in _WHOLE_IMAGE_BACKENDS:
            return self.whole_image_variant(name)
        if name in ("fashionclip-regions", "regions+attr-filter", "full+rerank"):
            return self.engine_variant(name)
        raise ValueError(f"unknown ablation variant {name!r}; known: {VARIANTS}")

    # ── running + reporting ──────────────────────────────────────────────────

    def run_all(
        self,
        queries: Sequence[Mapping[str, str]],
        labels: Mapping[str, set[str]],
        k_values: Sequence[int] = (1, 5, 10),
        variants: Sequence[str] = VARIANTS,
    ) -> dict[str, dict]:
        """Evaluate every variant; write eval/results/ablations.{md,json}.

        A variant failure (missing optional dep, unbuilt module) is recorded
        as {"error": ...} instead of aborting the whole ladder.
        """
        ks = tuple(sorted({int(k) for k in k_values}))
        results: dict[str, dict] = {}
        for name in variants:
            log.info("── ablation variant: %s", name)
            try:
                fn = self.variant(name)
                results[name] = run_eval(fn, queries, labels, k_values=ks)
                log.info("%s means: %s", name, results[name]["means"])
            except Exception as e:  # noqa: BLE001 — one rung must not kill the ladder
                log.exception("variant %s failed", name)
                results[name] = {"error": f"{type(e).__name__}: {e}"}
        self.write_reports(results, ks)
        return results

    def write_reports(
        self, results: Mapping[str, Mapping], k_values: Sequence[int]
    ) -> tuple[Path, Path]:
        """Write ablations.json (full detail) + ablations.md (the single table)."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.results_dir / "ablations.json"
        md_path = self.results_dir / "ablations.md"

        payload = {
            "meta": {
                "garment_backend": self.cfg.get_path("embedding.garment_backend"),
                "scene_backend": self.cfg.get_path("embedding.scene_backend"),
                "fusion_weights": self.cfg.get_path("search.weights"),
                "k_values": list(k_values),
            },
            "variants": results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        parts = [
            "# Ablation ladder",
            "",
            "Mean metrics over the labeled acceptance queries "
            "(see `ablations.json` for per-query detail).",
            "",
            _means_table(results, k_values),
            "",
            "Variant definitions:",
        ]
        parts += [f"- **{name}** — {VARIANT_NOTES.get(name, '')}" for name in results]
        errors = [f"> `{name}` failed: {res['error']}"
                  for name, res in results.items() if "error" in res]
        if errors:
            parts += [""] + errors
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts) + "\n")
        log.info("wrote %s and %s", md_path, json_path)
        return json_path, md_path

    # ── embedder benchmark ───────────────────────────────────────────────────

    def benchmark_embedders(
        self,
        queries: Sequence[Mapping[str, str]],
        labels: Mapping[str, set[str]],
        *,
        stores: Mapping[str, "VectorStore | str | Path"] | None = None,
        k_values: Sequence[int] = (1, 5, 10),
    ) -> dict[str, dict]:
        """Compare garment embedders on the regions-only variant (c).

        Rebuilding region vectors per embedder is expensive (detection + crop
        + embedding over the whole corpus), so this deliberately does NOT
        re-embed anything: pass PREBUILT stores (instances or persisted store
        directories), one per embedder, e.g.::

            runner.benchmark_embedders(queries, labels, stores={
                "fashion-clip": runner.store,  # implied default
                "marqo-siglip": "data/artifacts/index-marqo-siglip",
            })

        The runner's own store is used for the configured garment backend when
        that backend has no explicit entry. Backends without a store are
        skipped with a warning. Writes eval/results/embedder_benchmark.{md,json}.
        """
        ks = tuple(sorted({int(k) for k in k_values}))
        max_k = ks[-1]
        default_backend = str(self.cfg.get_path("embedding.garment_backend", "fashion-clip"))
        sources: dict[str, "VectorStore | str | Path"] = dict(stores or {})
        sources.setdefault(default_backend, self.store)

        results: dict[str, dict] = {}
        for backend in ("fashion-clip", "marqo-siglip"):
            src = sources.get(backend)
            if src is None:
                log.warning("no prebuilt store for %r — skipped "
                            "(see benchmark_embedders docstring)", backend)
                results[backend] = {"error": "no prebuilt store provided"}
                continue
            try:
                store = src if isinstance(src, VectorStore) else load_store(self.cfg, src)
                engine = self._build_engine(
                    store=store, garment_backend=backend,
                    weights={"garment": 1.0, "scene": 0.0, "attribute": 0.0},
                    prefilter_enabled=False, use_reranker=False)

                def search_fn(query: str, k: int, _e: SearchEngine = engine) -> list[str]:
                    _, res = _e.search(query, k=int(k), use_rerank=False)
                    return [r.image_id for r in res]

                results[backend] = run_eval(search_fn, queries, labels, k_values=ks)
            except Exception as e:  # noqa: BLE001
                log.exception("embedder benchmark failed for %s", backend)
                results[backend] = {"error": f"{type(e).__name__}: {e}"}

        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.results_dir / "embedder_benchmark.json"
        md_path = self.results_dir / "embedder_benchmark.md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        table = _means_table(results, ks, key_col="embedder")
        scored = {b: r.get("means", {}).get(f"ndcg@{max_k}", 0.0)
                  for b, r in results.items() if "error" not in r}
        winner_line = ""
        if len(scored) >= 2:
            winner = max(scored, key=lambda b: scored[b])
            winner_line = (f"\n**Winner: `{winner}`** by nDCG@{max_k} "
                           f"({scored[winner]:.4f}); this is what "
                           "`embedding.garment_backend` should be set to.\n")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                "# Garment embedder benchmark\n\n"
                "Regions-only retrieval (ablation variant c) with each garment "
                "embedder against its own prebuilt index — region vectors are "
                "never rebuilt here.\n\n" + table + "\n" + winner_line
            )
        log.info("wrote %s and %s", md_path, json_path)
        return results


# ── report table ─────────────────────────────────────────────────────────────

def _means_table(
    results: Mapping[str, Mapping],
    k_values: Sequence[int],
    *,
    key_col: str = "variant",
) -> str:
    """Single table: rows = variants, cols = R@k..., MRR, nDCG@max_k."""
    ks = sorted({int(k) for k in k_values})
    max_k = ks[-1]
    headers = [key_col] + [f"R@{k}" for k in ks] + ["MRR", f"nDCG@{max_k}"]
    rows: list[dict[str, object]] = []
    for name, res in results.items():
        row: dict[str, object] = {key_col: name}
        if "error" in res:
            for h in headers[1:]:
                row[h] = "—"
        else:
            means = res.get("means", {})
            for k in ks:
                row[f"R@{k}"] = float(means.get(f"recall@{k}", 0.0))
            row["MRR"] = float(means.get("mrr", 0.0))
            row[f"nDCG@{max_k}"] = float(means.get(f"ndcg@{max_k}", 0.0))
        rows.append(row)
    return render_markdown_table(rows, headers=headers)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    """CLI: run the full ablation ladder and write eval/results/ablations.{md,json}."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.ablations",
        description="Run the retrieval ablation ladder over the labeled queries.")
    parser.add_argument("--config", default=None, help="YAML config path (default: config/default.yaml)")
    parser.add_argument("--store-dir", default=None,
                        help="persisted index dir (default: <artifacts_dir>/index)")
    parser.add_argument("--images-dir", default=None,
                        help="image files dir (default: paths.images_dir)")
    parser.add_argument("--queries", default=None, help="queries yaml (default: eval.queries_file)")
    parser.add_argument("--labels", default=None, help="labels yaml (default: eval.labels_file)")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS),
                        help=f"subset of variants to run (default: all of {VARIANTS})")
    parser.add_argument("--benchmark-embedders", action="store_true",
                        help="also run the fashion-clip vs marqo-siglip benchmark")
    parser.add_argument("--marqo-store-dir", default=None,
                        help="prebuilt marqo-siglip index dir for --benchmark-embedders")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    queries = load_queries(args.queries or resolve(cfg, "eval.queries_file"))
    labels = load_labels(args.labels or resolve(cfg, "eval.labels_file"))
    k_values = tuple(cfg.get_path("eval.k_values", [1, 5, 10]))

    store = load_store(cfg, args.store_dir)
    images_dir = Path(args.images_dir) if args.images_dir else resolve(cfg, "paths.images_dir")
    runner = AblationRunner(cfg, store, images_dir)

    runner.run_all(queries, labels, k_values=k_values, variants=args.variants)
    if args.benchmark_embedders:
        stores: dict[str, "VectorStore | str | Path"] | None = None
        if args.marqo_store_dir:
            stores = {"marqo-siglip": args.marqo_store_dir}
        runner.benchmark_embedders(queries, labels, stores=stores, k_values=k_values)


if __name__ == "__main__":
    main()
