"""OpenSearch k-NN implementation of the VectorStore ABC.

Code-complete but NOT provisioned by default. Cost tradeoff: the smallest
sensible managed domain (t3.small.search) runs ~$26/mo, versus $0 for the
in-process FAISS backend at the ~1K-image scale of this project — so FAISS
stays the default and this backend is enabled by pointing
``storage.backend: opensearch`` at a live endpoint (terraform fills
``storage.opensearch.endpoint``). At 1M+ images the calculus flips: a managed
HNSW cluster gives horizontal scale, replication and server-side filtering.

Layout — two indices per prefix:

* ``{prefix}-garments`` — one doc per garment region: ``knn_vector`` (lucene
  HNSW, cosinesimil) + image_id/region_id keywords + bbox/label/det_score.
* ``{prefix}-scenes`` — one doc per image: scene ``knn_vector`` + the full
  attribute payload, with ``attributes.garments`` mapped as ``nested`` so a
  type∧color∧material conjunction must hold inside a SINGLE garment record —
  the same structural-binding guarantee the FAISS backend gets from
  ``AttributePredicate.matches_garment``.

The ``opensearchpy`` client is imported lazily in ``__init__`` so importing
this module never pulls the dependency for users of other backends.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.schemas import AttributePredicate, ImageAttributes, RegionRecord
from indexer.storage.base import VectorStore

log = logging.getLogger(__name__)


def _lucene_score_to_cosine(score: float) -> float:
    """Invert OpenSearch's lucene cosinesimil scoring: score = (1 + cos) / 2."""
    return 2.0 * float(score) - 1.0


def _to_list(vec: np.ndarray) -> list[float]:
    return np.asarray(vec, dtype=np.float32).reshape(-1).tolist()


class OpenSearchStore(VectorStore):
    """Multi-vector store on OpenSearch k-NN (lucene HNSW engine)."""

    name = "opensearch"

    def __init__(self, endpoint: str, index_prefix: str = "fashion",
                 garment_dim: int = 512, scene_dim: int = 512,
                 username: str | None = None, password: str | None = None,
                 use_ssl: bool = True) -> None:
        if not endpoint:
            raise ValueError(
                "OpenSearchStore requires a non-empty endpoint "
                "(config storage.opensearch.endpoint, filled by terraform output)")
        # Lazy import: only this backend needs the client dependency.
        from opensearchpy import NotFoundError, OpenSearch, helpers

        self._helpers = helpers
        self._not_found = NotFoundError
        self.garment_dim = int(garment_dim)
        self.scene_dim = int(scene_dim)
        self._garments_index = f"{index_prefix}-garments"
        self._scenes_index = f"{index_prefix}-scenes"

        kwargs: dict[str, Any] = {
            "use_ssl": use_ssl,
            "verify_certs": use_ssl,
            "timeout": 30,
        }
        if username and password:
            kwargs["http_auth"] = (username, password)
        if endpoint.startswith(("http://", "https://")):
            self._client = OpenSearch(hosts=[endpoint], **kwargs)
        else:
            self._client = OpenSearch(
                hosts=[{"host": endpoint, "port": 443}], **kwargs)
        self.ensure_indices()

    # ── index bootstrap ──────────────────────────────────────────────────────

    def ensure_indices(self) -> None:
        """Create the two indices with explicit mappings if they do not exist."""
        if not self._client.indices.exists(index=self._garments_index):
            self._client.indices.create(index=self._garments_index,
                                        body=self._garments_index_body())
            log.info("created index %s", self._garments_index)
        if not self._client.indices.exists(index=self._scenes_index):
            self._client.indices.create(index=self._scenes_index,
                                        body=self._scenes_index_body())
            log.info("created index %s", self._scenes_index)

    def _knn_field(self, dim: int) -> dict[str, Any]:
        return {
            "type": "knn_vector",
            "dimension": dim,
            "method": {
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "cosinesimil",
                "parameters": {"m": 16, "ef_construction": 128},
            },
        }

    def _garments_index_body(self) -> dict[str, Any]:
        return {
            "settings": {"index": {"knn": True}},
            "mappings": {"properties": {
                "vec": self._knn_field(self.garment_dim),
                "image_id": {"type": "keyword"},
                "region_id": {"type": "keyword"},
                "bbox": {"type": "float"},
                "label": {"type": "keyword"},
                "det_score": {"type": "float"},
            }},
        }

    def _scenes_index_body(self) -> dict[str, Any]:
        return {
            "settings": {"index": {"knn": True}},
            "mappings": {"properties": {
                "vec": self._knn_field(self.scene_dim),
                "image_id": {"type": "keyword"},
                "attributes": {
                    "type": "object",
                    "enabled": True,
                    "properties": {
                        # nested: per-garment conjunctions bind structurally
                        "garments": {
                            "type": "nested",
                            "properties": {
                                "type": {"type": "keyword"},
                                "color": {"type": "keyword"},
                                "material": {"type": "keyword"},
                                "formality": {"type": "keyword"},
                            },
                        },
                    },
                },
                "regions": {"type": "object"},
            }},
        }

    # ── writes ───────────────────────────────────────────────────────────────

    def add_image(self, image_id: str, scene_vec: np.ndarray,
                  garment_vecs: list[tuple[RegionRecord, np.ndarray]],
                  attributes: ImageAttributes) -> None:
        """Idempotent upsert: delete-by-query on image_id, then bulk re-index."""
        for index in (self._garments_index, self._scenes_index):
            self._client.delete_by_query(
                index=index,
                body={"query": {"term": {"image_id": image_id}}},
                conflicts="proceed", refresh=True)

        if not attributes.image_id:
            attributes = attributes.model_copy(update={"image_id": image_id})

        actions: list[dict[str, Any]] = []
        for region, vec in garment_vecs:
            actions.append({
                "_op_type": "index",
                "_index": self._garments_index,
                "_id": region.region_id,
                "_source": {
                    "image_id": image_id,
                    "region_id": region.region_id,
                    "bbox": [float(x) for x in region.bbox],
                    "label": region.label,
                    "det_score": float(region.det_score),
                    "vec": _to_list(vec),
                },
            })
        actions.append({
            "_op_type": "index",
            "_index": self._scenes_index,
            "_id": image_id,
            "_source": {
                "image_id": image_id,
                "vec": _to_list(scene_vec),
                "attributes": attributes.model_dump(),
                "regions": [region.model_dump() for region, _ in garment_vecs],
            },
        })
        self._helpers.bulk(self._client, actions, refresh=True)

    # ── vector search ────────────────────────────────────────────────────────

    def _knn_search(self, index: str, source: list[str], query_vec: np.ndarray,
                    k: int, allowed_ids: Optional[set[str]]) -> list[dict[str, Any]]:
        knn: dict[str, Any] = {"vector": _to_list(query_vec), "k": k}
        if allowed_ids is not None:
            # lucene engine applies this as an efficient pre-filter
            knn["filter"] = {"terms": {"image_id": sorted(allowed_ids)}}
        body = {"size": k, "_source": source, "query": {"knn": {"vec": knn}}}
        resp = self._client.search(index=index, body=body)
        return resp["hits"]["hits"]

    def search_garments(self, query_vec: np.ndarray, k: int,
                        allowed_ids: Optional[set[str]] = None
                        ) -> list[tuple[str, str, float]]:
        if k <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        hits = self._knn_search(self._garments_index, ["image_id", "region_id"],
                                query_vec, k, allowed_ids)
        return [(h["_source"]["image_id"], h["_source"]["region_id"],
                 _lucene_score_to_cosine(h["_score"])) for h in hits][:k]

    def search_scene(self, query_vec: np.ndarray, k: int,
                     allowed_ids: Optional[set[str]] = None
                     ) -> list[tuple[str, float]]:
        if k <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        hits = self._knn_search(self._scenes_index, ["image_id"],
                                query_vec, k, allowed_ids)
        return [(h["_source"]["image_id"],
                 _lucene_score_to_cosine(h["_score"])) for h in hits][:k]

    # ── attribute / metadata access ──────────────────────────────────────────

    @staticmethod
    def _nested_predicate(p: AttributePredicate) -> dict[str, Any]:
        """type∧color∧material inside ONE nested garment doc (structural binding)."""
        must: list[dict[str, Any]] = []
        if p.types:
            must.append({"terms": {"attributes.garments.type": p.types}})
        if p.colors:
            must.append({"terms": {"attributes.garments.color": p.colors}})
        if p.materials:
            must.append({"terms": {"attributes.garments.material": p.materials}})
        return {"nested": {"path": "attributes.garments",
                           "query": {"bool": {"must": must}}}}

    def find_images(self, required: list[AttributePredicate],
                    excluded: list[AttributePredicate]) -> set[str]:
        must = [self._nested_predicate(p) for p in required
                if p.types or p.colors or p.materials]
        must_not = [self._nested_predicate(p) for p in excluded
                    if p.types or p.colors or p.materials]
        bool_q: dict[str, Any] = {}
        if must:
            bool_q["must"] = must
        if must_not:
            bool_q["must_not"] = must_not
        query: dict[str, Any] = {"bool": bool_q} if bool_q else {"match_all": {}}
        found: set[str] = set()
        for hit in self._helpers.scan(self._client, index=self._scenes_index,
                                      query={"query": query,
                                             "_source": ["image_id"]}):
            found.add(hit["_source"]["image_id"])
        return found

    def get_attributes(self, image_id: str) -> Optional[ImageAttributes]:
        try:
            doc = self._client.get(index=self._scenes_index, id=image_id,
                                   _source=["attributes"])
        except self._not_found:
            return None
        return ImageAttributes.model_validate(doc["_source"]["attributes"])

    def get_regions(self, image_id: str) -> list[RegionRecord]:
        try:
            doc = self._client.get(index=self._scenes_index, id=image_id,
                                   _source=["regions"])
        except self._not_found:
            return []
        return [RegionRecord.model_validate(r)
                for r in doc["_source"].get("regions", [])]

    def iter_attributes(self) -> Iterable[ImageAttributes]:
        for hit in self._helpers.scan(self._client, index=self._scenes_index,
                                      query={"query": {"match_all": {}},
                                             "_source": ["attributes"]}):
            yield ImageAttributes.model_validate(hit["_source"]["attributes"])

    # ── lifecycle ────────────────────────────────────────────────────────────

    def image_ids(self) -> list[str]:
        return [hit["_source"]["image_id"] for hit in self._helpers.scan(
            self._client, index=self._scenes_index,
            query={"query": {"match_all": {}}, "_source": ["image_id"]})]

    def stats(self) -> dict:
        n_garments = int(self._client.count(index=self._garments_index)["count"])
        n_scenes = int(self._client.count(index=self._scenes_index)["count"])
        type_counts: Counter[str] = Counter()
        color_counts: Counter[str] = Counter()
        for attrs in self.iter_attributes():
            for g in attrs.garments:
                if g.type:
                    type_counts[g.type] += 1
                if g.color:
                    color_counts[g.color] += 1
        return {
            "backend": self.name,
            "n_images": n_scenes,
            "n_garment_vectors": n_garments,
            "n_scene_vectors": n_scenes,
            "garment_dim": self.garment_dim,
            "scene_dim": self.scene_dim,
            "top_garment_types": dict(type_counts.most_common(10)),
            "top_colors": dict(color_counts.most_common(10)),
        }

    def persist(self, path: str | Path) -> None:
        """No-op: persistence is server-side (index replicas + snapshots)."""
        log.info("OpenSearchStore.persist(%s): no-op — data lives in the domain "
                 "(%s / %s)", path, self._garments_index, self._scenes_index)

    def load(self, path: str | Path) -> None:
        """No-op: the domain already holds the indices; nothing to load locally."""
        log.info("OpenSearchStore.load(%s): no-op — reading live indices "
                 "(%s / %s)", path, self._garments_index, self._scenes_index)
