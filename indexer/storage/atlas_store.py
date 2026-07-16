"""MongoDB Atlas Vector Search implementation of the VectorStore ABC.

Code-complete but NOT provisioned by default: the free M0 tier requires a
manual Atlas signup (no API-driven provisioning without an org), so FAISS
stays the default backend. Enable with ``storage.backend: atlas`` and the
connection string in the env var named by ``storage.atlas.uri_env``
(default ``ATLAS_URI`` — never in code or config files).

Layout — two collections in one database:

* ``garments`` — one doc per region: image_id, region_id, bbox, label,
  det_score, vec.
* ``scenes`` — one doc per image: image_id, vec, attributes (plain dict of the
  ``ImageAttributes`` payload).

Vector search runs through ``$vectorSearch`` aggregation stages against Atlas
Search indexes named ``garment_vec_idx`` / ``scene_vec_idx``.
:meth:`AtlasStore.ensure_indexes` creates them via the driver where permitted;
on clusters where programmatic search-index creation is not allowed, paste
these definitions in the Atlas UI (Search -> Create Search Index -> JSON
editor, type "Vector Search") — adjust ``numDimensions`` to your configured
embedding dims (512 for fashion-clip / clip-vit-b32, 768 for marqo-siglip):

``garments`` collection, index name ``garment_vec_idx``::

    {"fields": [
        {"type": "vector", "path": "vec", "numDimensions": 512,
         "similarity": "cosine"},
        {"type": "filter", "path": "image_id"}
    ]}

``scenes`` collection, index name ``scene_vec_idx``::

    {"fields": [
        {"type": "vector", "path": "vec", "numDimensions": 512,
         "similarity": "cosine"},
        {"type": "filter", "path": "image_id"}
    ]}

Attribute pre-filtering uses ``$elemMatch`` on ``attributes.garments`` so a
type∧color∧material conjunction must hold inside a SINGLE garment record —
the same structural-binding guarantee as the FAISS and OpenSearch backends.

``pymongo`` is imported lazily in ``__init__`` so importing this module never
pulls the dependency for users of other backends.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.schemas import AttributePredicate, ImageAttributes, RegionRecord
from indexer.storage.base import VectorStore

log = logging.getLogger(__name__)

GARMENT_VEC_INDEX = "garment_vec_idx"
SCENE_VEC_INDEX = "scene_vec_idx"


def _atlas_score_to_cosine(score: float) -> float:
    """Invert Atlas's cosine vectorSearchScore: score = (1 + cos) / 2."""
    return 2.0 * float(score) - 1.0


def _to_list(vec: np.ndarray) -> list[float]:
    return np.asarray(vec, dtype=np.float32).reshape(-1).tolist()


class AtlasStore(VectorStore):
    """Multi-vector store on MongoDB Atlas Vector Search."""

    name = "atlas"

    def __init__(self, uri: str | None = None, db: str = "fashion",
                 garment_dim: int = 512, scene_dim: int = 512) -> None:
        uri = uri or os.environ.get("ATLAS_URI")
        if not uri:
            raise ValueError(
                "AtlasStore needs a connection string: pass uri= or set the "
                "ATLAS_URI environment variable (see storage.atlas.uri_env)")
        # Lazy import: only this backend needs the driver dependency.
        import pymongo

        self._client = pymongo.MongoClient(uri)
        self._db = self._client[db]
        self._garments = self._db["garments"]
        self._scenes = self._db["scenes"]
        self.garment_dim = int(garment_dim)
        self.scene_dim = int(scene_dim)

    # ── index bootstrap ──────────────────────────────────────────────────────

    def _vector_index_definition(self, dim: int) -> dict[str, Any]:
        return {"fields": [
            {"type": "vector", "path": "vec", "numDimensions": dim,
             "similarity": "cosine"},
            {"type": "filter", "path": "image_id"},
        ]}

    def ensure_indexes(self) -> None:
        """Create the two Atlas vector search indexes if the cluster permits it.

        On tiers/roles where programmatic search-index creation is refused,
        this logs the JSON definitions to paste in the Atlas UI (also in the
        module docstring) instead of raising.
        """
        try:
            from pymongo.operations import SearchIndexModel

            for coll, index_name, dim in (
                (self._garments, GARMENT_VEC_INDEX, self.garment_dim),
                (self._scenes, SCENE_VEC_INDEX, self.scene_dim),
            ):
                existing = {ix["name"] for ix in coll.list_search_indexes()}
                if index_name in existing:
                    continue
                coll.create_search_index(SearchIndexModel(
                    definition=self._vector_index_definition(dim),
                    name=index_name, type="vectorSearch"))
                log.info("created Atlas vector search index %s on %s.%s",
                         index_name, self._db.name, coll.name)
        except Exception as exc:  # noqa: BLE001 — M0/permissions may refuse this
            log.warning(
                "could not create Atlas vector search indexes (%s); create them "
                "manually in the Atlas UI: %s on 'garments' -> %s ; %s on "
                "'scenes' -> %s", exc,
                GARMENT_VEC_INDEX, self._vector_index_definition(self.garment_dim),
                SCENE_VEC_INDEX, self._vector_index_definition(self.scene_dim))

    # ── writes ───────────────────────────────────────────────────────────────

    def add_image(self, image_id: str, scene_vec: np.ndarray,
                  garment_vecs: list[tuple[RegionRecord, np.ndarray]],
                  attributes: ImageAttributes) -> None:
        """Idempotent upsert: delete all docs for image_id, then re-insert."""
        self._garments.delete_many({"image_id": image_id})
        self._scenes.delete_many({"image_id": image_id})

        if not attributes.image_id:
            attributes = attributes.model_copy(update={"image_id": image_id})

        garment_docs = [{
            "image_id": image_id,
            "region_id": region.region_id,
            "bbox": [float(x) for x in region.bbox],
            "label": region.label,
            "det_score": float(region.det_score),
            "vec": _to_list(vec),
        } for region, vec in garment_vecs]
        if garment_docs:
            self._garments.insert_many(garment_docs)
        self._scenes.insert_one({
            "image_id": image_id,
            "vec": _to_list(scene_vec),
            "attributes": attributes.model_dump(),
        })

    # ── vector search ────────────────────────────────────────────────────────

    def _vector_search(self, coll: Any, index_name: str, query_vec: np.ndarray,
                       k: int, allowed_ids: Optional[set[str]],
                       project: dict[str, Any]) -> list[dict[str, Any]]:
        stage: dict[str, Any] = {
            "index": index_name,
            "path": "vec",
            "queryVector": _to_list(query_vec),
            "numCandidates": max(200, 10 * k),
            "limit": k,
        }
        if allowed_ids is not None:
            stage["filter"] = {"image_id": {"$in": sorted(allowed_ids)}}
        pipeline = [
            {"$vectorSearch": stage},
            {"$project": {"_id": 0, "score": {"$meta": "vectorSearchScore"},
                          **project}},
        ]
        return list(coll.aggregate(pipeline))

    def search_garments(self, query_vec: np.ndarray, k: int,
                        allowed_ids: Optional[set[str]] = None
                        ) -> list[tuple[str, str, float]]:
        if k <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        docs = self._vector_search(self._garments, GARMENT_VEC_INDEX, query_vec,
                                   k, allowed_ids,
                                   {"image_id": 1, "region_id": 1})
        return [(d["image_id"], d["region_id"],
                 _atlas_score_to_cosine(d["score"])) for d in docs][:k]

    def search_scene(self, query_vec: np.ndarray, k: int,
                     allowed_ids: Optional[set[str]] = None
                     ) -> list[tuple[str, float]]:
        if k <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        docs = self._vector_search(self._scenes, SCENE_VEC_INDEX, query_vec,
                                   k, allowed_ids, {"image_id": 1})
        return [(d["image_id"], _atlas_score_to_cosine(d["score"]))
                for d in docs][:k]

    # ── attribute / metadata access ──────────────────────────────────────────

    @staticmethod
    def _elem_match(p: AttributePredicate) -> dict[str, Any]:
        """type∧color∧material inside ONE garment element (structural binding)."""
        elem: dict[str, Any] = {}
        if p.types:
            elem["type"] = {"$in": p.types}
        if p.colors:
            elem["color"] = {"$in": p.colors}
        if p.materials:
            elem["material"] = {"$in": p.materials}
        return elem

    def find_images(self, required: list[AttributePredicate],
                    excluded: list[AttributePredicate]) -> set[str]:
        must: list[dict[str, Any]] = []
        for p in required:
            elem = self._elem_match(p)
            if elem:
                must.append({"attributes.garments": {"$elemMatch": elem}})
        must_not: list[dict[str, Any]] = []
        for p in excluded:
            elem = self._elem_match(p)
            if elem:
                must_not.append({"attributes.garments": {"$elemMatch": elem}})
        query: dict[str, Any] = {}
        if must:
            query["$and"] = must
        if must_not:
            query["$nor"] = must_not
        cursor = self._scenes.find(query, {"_id": 0, "image_id": 1})
        return {doc["image_id"] for doc in cursor}

    def get_attributes(self, image_id: str) -> Optional[ImageAttributes]:
        doc = self._scenes.find_one({"image_id": image_id},
                                    {"_id": 0, "attributes": 1})
        if doc is None:
            return None
        return ImageAttributes.model_validate(doc["attributes"])

    def get_regions(self, image_id: str) -> list[RegionRecord]:
        cursor = self._garments.find({"image_id": image_id},
                                     {"_id": 0, "vec": 0}).sort("region_id", 1)
        return [RegionRecord.model_validate(doc) for doc in cursor]

    def iter_attributes(self) -> Iterable[ImageAttributes]:
        for doc in self._scenes.find({}, {"_id": 0, "attributes": 1}):
            yield ImageAttributes.model_validate(doc["attributes"])

    # ── lifecycle ────────────────────────────────────────────────────────────

    def image_ids(self) -> list[str]:
        return [doc["image_id"]
                for doc in self._scenes.find({}, {"_id": 0, "image_id": 1})]

    def stats(self) -> dict:
        n_garments = int(self._garments.count_documents({}))
        n_scenes = int(self._scenes.count_documents({}))
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
        """No-op: persistence is server-side (Atlas cluster storage + backups)."""
        log.info("AtlasStore.persist(%s): no-op — data lives in Atlas db %r",
                 path, self._db.name)

    def load(self, path: str | Path) -> None:
        """No-op: the cluster already holds the collections; nothing to load."""
        log.info("AtlasStore.load(%s): no-op — reading live Atlas db %r",
                 path, self._db.name)
