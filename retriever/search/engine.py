"""The retrieval engine: structured intent → metadata pre-filter → multi-vector
ANN → weighted fusion → cross-encoder rerank.

Compositionality is handled structurally, not in embedding space:
"a red tie and a white shirt" becomes two AttributePredicates that must each be
satisfied by a single garment record on the same image — so an image with a
white tie and a red shirt is filtered out before any vector math happens.

No AWS imports here. Everything injected: store, embedders, parser, reranker.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np

from core.lexicon import (
    STYLE_TO_FORMALITY,
    canonical_color,
    canonical_garment,
    canonical_material,
    canonical_scene,
    expand_color,
)
from core.schemas import (
    AttributePredicate,
    ComponentScores,
    ImageAttributes,
    MatchExplanation,
    ParsedQuery,
    SearchResult,
)
from indexer.models.base import EmbeddingModel
from indexer.storage.base import VectorStore
from retriever.parsing.base import QueryParser
from retriever.rerank.base import Reranker

log = logging.getLogger(__name__)

GARMENT_PROMPT_TEMPLATE = "a photo of a person wearing {term}"
SCENE_PROMPT_TEMPLATE = "a photo taken in {scene}"


# ── query → predicates ───────────────────────────────────────────────────────

def predicates_from_parsed(parsed: ParsedQuery
                           ) -> tuple[list[AttributePredicate], list[AttributePredicate]]:
    """Expand parsed intent into conjunctive (required) and exclusion predicates,
    canonicalized into the shared lexicon."""
    required: list[AttributePredicate] = []
    for g in parsed.garments:
        t = canonical_garment(g.type) if g.type else None
        c = canonical_color(g.color) if g.color else None
        if not t and not c:
            continue
        required.append(AttributePredicate(
            types=[t] if t else [],
            colors=expand_color(c) if c else []))  # neighbor-tolerant on query side

    excluded: list[AttributePredicate] = []
    for n in parsed.negations:
        t = canonical_garment(n.type) or canonical_garment(n.term)
        c = canonical_color(n.color) if n.color else None
        m = canonical_material(n.material) or canonical_material(n.term)
        pred = AttributePredicate(
            types=[t] if t else [], colors=[c] if c else [],
            materials=[m] if m else [])
        if pred.types or pred.colors or pred.materials:
            excluded.append(pred)
    return required, excluded


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    if span < 1e-9:
        return {k: (1.0 if hi > 0 else 0.0) for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


class SearchEngine:
    """Composition of all retrieval stages. Construct via core.bootstrap.build_engine."""

    def __init__(
        self,
        store: VectorStore,
        garment_embedder: EmbeddingModel,
        scene_embedder: EmbeddingModel,
        parser: QueryParser,
        reranker: Optional[Reranker] = None,
        image_resolver: Optional[Callable[[str], "object"]] = None,  # image_id -> PIL.Image
        *,
        weights: Optional[dict[str, float]] = None,
        candidates_k: int = 50,
        prefilter_enabled: bool = True,
        prefilter_min_candidates: int = 12,
        prefilter_soft_fallback: bool = True,
        rerank_top_n: int = 50,
        rerank_weight: float = 0.35,
    ):
        self.store = store
        self.garment_embedder = garment_embedder
        self.scene_embedder = scene_embedder
        self.parser = parser
        self.reranker = reranker
        self.image_resolver = image_resolver
        self.weights = weights or {"garment": 0.55, "scene": 0.25, "attribute": 0.20}
        self.candidates_k = candidates_k
        self.prefilter_enabled = prefilter_enabled
        self.prefilter_min_candidates = prefilter_min_candidates
        self.prefilter_soft_fallback = prefilter_soft_fallback
        self.rerank_top_n = rerank_top_n
        self.rerank_weight = rerank_weight

    # ── public API ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10,
               parsed: Optional[ParsedQuery] = None,
               use_rerank: bool = True) -> tuple[ParsedQuery, list[SearchResult]]:
        parsed = parsed or self.parser.parse(query)
        garment_qvecs = self._garment_query_vecs(parsed)
        scene_vec = self._scene_query_vec(parsed)
        results = self._retrieve(parsed, garment_qvecs, scene_vec,
                                 rerank_text=query, k=k, use_rerank=use_rerank)
        return parsed, results

    def search_by_image(self, image, refinement: Optional[str] = None,
                        k: int = 10, use_rerank: bool = False
                        ) -> tuple[Optional[ParsedQuery], list[SearchResult]]:
        """Query-by-example with optional text refinement ('but in green').

        The uploaded image is embedded in both spaces; refinement garment terms are
        mixed into the garment query vector (equal-weight average in embedding
        space, renormalized) and refinement negations/predicates apply as usual.
        """
        img_gvec = self.garment_embedder.embed_image(image)
        img_svec = self.scene_embedder.embed_image(image)

        parsed = self.parser.parse(refinement) if refinement else None
        garment_qvecs: list[tuple[str, np.ndarray]] = [("query image", img_gvec)]
        if parsed and parsed.has_garment_terms():
            term_vecs = self._garment_query_vecs(parsed)
            for term, tvec in term_vecs:
                mixed = img_gvec + tvec
                mixed = mixed / max(float(np.linalg.norm(mixed)), 1e-12)
                garment_qvecs.append((f"image + {term}", mixed))
        results = self._retrieve(parsed or ParsedQuery(raw=refinement or ""),
                                 garment_qvecs, ("query image scene", img_svec),
                                 rerank_text=refinement, k=k, use_rerank=use_rerank)
        return parsed, results

    # ── query vector construction ────────────────────────────────────────────

    def _garment_query_vecs(self, parsed: ParsedQuery) -> list[tuple[str, np.ndarray]]:
        terms = [g.text() for g in parsed.garments if g.text()]
        if not terms:
            return []
        prompts = [GARMENT_PROMPT_TEMPLATE.format(term=t) for t in terms]
        vecs = self.garment_embedder.embed_texts(prompts)
        return list(zip(terms, vecs))

    def _scene_query_vec(self, parsed: ParsedQuery) -> tuple[str, np.ndarray]:
        if parsed.scene:
            label = parsed.scene
            text = SCENE_PROMPT_TEMPLATE.format(scene=parsed.scene)
        elif parsed.style:
            label = f"{parsed.style} scene"
            text = parsed.raw  # whole-image CLIP reads style holistically
        else:
            label = "query context"
            text = parsed.raw
        return label, self.scene_embedder.embed_text(text)

    # ── core retrieval ───────────────────────────────────────────────────────

    def _retrieve(self, parsed: ParsedQuery,
                  garment_qvecs: list[tuple[str, np.ndarray]],
                  scene_query: Optional[tuple[str, np.ndarray]],
                  *, rerank_text: Optional[str], k: int,
                  use_rerank: bool) -> list[SearchResult]:
        required, excluded = predicates_from_parsed(parsed)

        # 1 — metadata pre-filter (the structural compositionality gate)
        allowed: Optional[set[str]] = None
        prefilter_soft = False
        if self.prefilter_enabled and (required or excluded):
            candidate_ids = self.store.find_images(required, excluded)
            if len(candidate_ids) >= self.prefilter_min_candidates or not self.prefilter_soft_fallback:
                allowed = candidate_ids
            else:
                # too few survivors (attribute extraction is imperfect) — keep
                # exclusions hard, demote requirements to a scoring feature
                prefilter_soft = True
                if excluded:
                    allowed = self.store.find_images([], excluded)
                log.info("prefilter softened: %d hard survivors < %d for %r",
                         len(candidate_ids), self.prefilter_min_candidates, parsed.raw)
        if allowed is not None and not allowed:
            return []

        # 2 — per-channel ANN, first pass to build the candidate pool
        term_hits: dict[str, dict[str, tuple[str, float]]] = {}
        pool: set[str] = set()
        for term, vec in garment_qvecs:
            hits = self.store.search_garments(vec, self.candidates_k, allowed)
            best: dict[str, tuple[str, float]] = {}
            for image_id, region_id, sim in hits:
                if image_id not in best or sim > best[image_id][1]:
                    best[image_id] = (region_id, sim)
            term_hits[term] = best
            pool.update(best)
        scene_hits: dict[str, float] = {}
        if scene_query is not None:
            for image_id, sim in self.store.search_scene(
                    scene_query[1], self.candidates_k, allowed):
                scene_hits[image_id] = sim
            pool.update(scene_hits)
        if not pool:
            return []

        # 3 — second restricted pass so every candidate has every channel score
        fill_k = max(self.candidates_k, 4 * len(pool))
        for term, vec in garment_qvecs:
            missing = pool - set(term_hits[term])
            if missing:
                for image_id, region_id, sim in self.store.search_garments(
                        vec, fill_k, pool):
                    cur = term_hits[term].get(image_id)
                    if cur is None or sim > cur[1]:
                        term_hits[term][image_id] = (region_id, sim)
        if scene_query is not None and pool - set(scene_hits):
            for image_id, sim in self.store.search_scene(scene_query[1], fill_k, pool):
                scene_hits.setdefault(image_id, sim)

        # 4 — channel scores
        garment_raw = {
            img: float(np.mean([term_hits[t].get(img, (None, 0.0))[1]
                                for t, _ in garment_qvecs]))
            for img in pool
        } if garment_qvecs else {}
        scene_raw = {img: scene_hits.get(img, 0.0) for img in pool} if scene_query else {}
        attr_scores = {img: self._attribute_score(img, parsed, required) for img in pool}

        garment_n = _minmax(garment_raw)
        scene_n = _minmax(scene_raw)

        active = {
            "garment": bool(garment_qvecs),
            "scene": bool(scene_query),
            "attribute": bool(required or parsed.style or parsed.scene),
        }
        wsum = sum(w for ch, w in self.weights.items() if active.get(ch)) or 1.0

        fused: dict[str, float] = {}
        for img in pool:
            s = 0.0
            if active["garment"]:
                s += self.weights["garment"] * garment_n.get(img, 0.0)
            if active["scene"]:
                s += self.weights["scene"] * scene_n.get(img, 0.0)
            if active["attribute"]:
                s += self.weights["attribute"] * attr_scores.get(img, 0.0)
            fused[img] = s / wsum

        ranked = sorted(fused, key=fused.get, reverse=True)

        # 5 — rerank top-N with the cross-modal ITM head
        rerank_scores: dict[str, float] = {}
        if (use_rerank and self.reranker and self.image_resolver
                and rerank_text and rerank_text.strip()):
            top = ranked[: self.rerank_top_n]
            images = []
            kept = []
            for img in top:
                try:
                    images.append(self.image_resolver(img))
                    kept.append(img)
                except Exception as e:  # noqa: BLE001 — a missing file shouldn't kill the query
                    log.warning("image_resolver failed for %s: %s", img, e)
            if not images and top:
                log.warning("rerank requested but no candidate images resolvable "
                            "(image files not available locally?) — skipping rerank")
            if images:
                probs = self.reranker.score(rerank_text, images)
                rerank_scores = dict(zip(kept, (float(p) for p in probs)))
                w = self.rerank_weight
                for img in kept:
                    fused[img] = (1 - w) * fused[img] + w * rerank_scores[img]
                ranked = sorted(fused, key=fused.get, reverse=True)

        # 6 — assemble explainable results
        results: list[SearchResult] = []
        for img in ranked[:k]:
            regions = {r.region_id: r for r in self.store.get_regions(img)}
            matches: list[MatchExplanation] = []
            for term, _ in garment_qvecs:
                hit = term_hits.get(term, {}).get(img)
                if hit is None:
                    continue
                region_id, sim = hit
                region = regions.get(region_id)
                if region is None:
                    continue
                matches.append(MatchExplanation(
                    region_id=region_id, bbox=region.bbox, label=region.label,
                    query_term=term, similarity=round(sim, 4)))
            results.append(SearchResult(
                image_id=img,
                score=round(fused[img], 4),
                components=ComponentScores(
                    garment=round(garment_n.get(img, 0.0), 4),
                    scene=round(scene_n.get(img, 0.0), 4),
                    attribute=round(attr_scores.get(img, 0.0), 4),
                    rerank=(round(rerank_scores[img], 4)
                            if img in rerank_scores else None),
                ),
                matches=matches,
                attributes=self.store.get_attributes(img),
            ))
        return results

    # ── attribute channel ────────────────────────────────────────────────────

    def _attribute_score(self, image_id: str, parsed: ParsedQuery,
                         required: list[AttributePredicate]) -> float:
        """Soft attribute agreement in [0,1]: predicate satisfaction + formality
        fit + scene-type fit, averaged over the signals the query actually has."""
        attrs = self.store.get_attributes(image_id)
        if attrs is None:
            return 0.0
        parts: list[float] = []
        if required:
            sat = sum(1.0 for p in required if p.matches_image(attrs)) / len(required)
            parts.extend([sat, sat])  # predicate satisfaction dominates (2 shares)
        if parsed.style:
            compatible = STYLE_TO_FORMALITY.get(parsed.style, {parsed.style})
            fits = [g for g in attrs.garments if g.formality in compatible]
            parts.append(len(fits) / len(attrs.garments) if attrs.garments else 0.0)
        target_scene = canonical_scene(parsed.scene) if parsed.scene else None
        if target_scene:
            parts.append(1.0 if attrs.scene_type == target_scene else 0.0)
        return float(np.mean(parts)) if parts else 0.0
