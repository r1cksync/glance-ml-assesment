"""In-process FAISS implementation of the VectorStore ABC (default backend).

Two exact inner-product indexes (garments, scenes), each wrapped in an
``IndexIDMap2`` so vectors carry stable int64 ids that survive removals —
which is what makes :meth:`FaissStore.add_image` an idempotent upsert.
All embedding vectors are L2-normalized upstream
(``indexer.models.base.EmbeddingModel``), so inner product == cosine similarity.

Exact search (``IndexFlatIP``) is the right call at the ~1K-image scale of this
project: a couple of thousand 512-d float32 rows is a few MB and a brute-force
scan is sub-millisecond. The ABC boundary is what makes swapping in HNSW /
OpenSearch / Atlas at larger scale a config change, not a refactor.

Filtered search uses ``faiss.SearchParameters(sel=IDSelectorBatch(...))``
(supported on IndexFlat in faiss>=1.7.4; IndexIDMap2 translates the selector to
external ids). If the ``params`` kwarg is unavailable at runtime, we fall back
to over-fetching ``k*10`` and filtering in Python — both paths are implemented.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from core.schemas import AttributePredicate, ImageAttributes, RegionRecord
from indexer.storage.base import VectorStore

log = logging.getLogger(__name__)

_GARMENTS_FILE = "garments.faiss"
_SCENES_FILE = "scenes.faiss"
_META_FILE = "meta.json"


def _as_matrix(vec: np.ndarray, dim: int, what: str) -> np.ndarray:
    """Coerce a vector/matrix to contiguous float32 [N, dim]."""
    m = np.ascontiguousarray(np.asarray(vec, dtype=np.float32))
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.ndim != 2 or m.shape[1] != dim:
        raise ValueError(f"{what} vectors have shape {m.shape}, expected [N, {dim}]")
    return m


def _id_selector(ids: np.ndarray) -> "faiss.IDSelectorBatch":
    """Build an IDSelectorBatch across faiss python API variants."""
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    try:
        return faiss.IDSelectorBatch(ids)
    except TypeError:
        return faiss.IDSelectorBatch(int(ids.size), faiss.swig_ptr(ids))


class FaissStore(VectorStore):
    """Multi-vector store: N garment vectors + 1 scene vector + attributes per image."""

    name = "faiss"

    def __init__(self, garment_dim: int, scene_dim: int) -> None:
        self.garment_dim = int(garment_dim)
        self.scene_dim = int(scene_dim)
        self._garment_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.garment_dim))
        self._scene_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.scene_dim))
        # sequential int64 id counters (never reused, so ids stay unique forever)
        self._next_garment_id: int = 0
        self._next_scene_id: int = 0
        # faiss id -> (image_id, region_id)
        self._garment_meta: dict[int, tuple[str, str]] = {}
        # faiss id -> image_id
        self._scene_meta: dict[int, str] = {}
        # image_id -> {"garment_ids": [int], "scene_id": int,
        #              "regions": [RegionRecord], "attributes": ImageAttributes}
        self._images: dict[str, dict[str, Any]] = {}

    # ── writes ───────────────────────────────────────────────────────────────

    def add_image(self, image_id: str, scene_vec: np.ndarray,
                  garment_vecs: list[tuple[RegionRecord, np.ndarray]],
                  attributes: ImageAttributes) -> None:
        """Idempotent upsert: re-adding an image first removes its old vectors."""
        old = self._images.pop(image_id, None)
        if old is not None:
            self._remove_vectors(old)
            log.debug("re-indexing %s: removed %d old garment vectors",
                      image_id, len(old["garment_ids"]))

        scene_id = self._next_scene_id
        self._next_scene_id += 1
        sv = _as_matrix(scene_vec, self.scene_dim, "scene")
        self._scene_index.add_with_ids(sv, np.asarray([scene_id], dtype=np.int64))
        self._scene_meta[scene_id] = image_id

        garment_ids: list[int] = []
        regions: list[RegionRecord] = []
        if garment_vecs:
            mat = _as_matrix(
                np.stack([np.asarray(v, dtype=np.float32).reshape(-1)
                          for _, v in garment_vecs]),
                self.garment_dim, "garment")
            ids = np.arange(self._next_garment_id,
                            self._next_garment_id + len(garment_vecs), dtype=np.int64)
            self._next_garment_id += len(garment_vecs)
            self._garment_index.add_with_ids(mat, ids)
            for (region, _), fid in zip(garment_vecs, ids):
                self._garment_meta[int(fid)] = (image_id, region.region_id)
                garment_ids.append(int(fid))
                regions.append(region)

        if not attributes.image_id:
            attributes = attributes.model_copy(update={"image_id": image_id})
        self._images[image_id] = {
            "garment_ids": garment_ids,
            "scene_id": scene_id,
            "regions": regions,
            "attributes": attributes,
        }

    def _remove_vectors(self, record: dict[str, Any]) -> None:
        gids = np.asarray(record["garment_ids"], dtype=np.int64)
        if gids.size:
            self._garment_index.remove_ids(_id_selector(gids))
        self._scene_index.remove_ids(
            _id_selector(np.asarray([record["scene_id"]], dtype=np.int64)))
        for gid in record["garment_ids"]:
            self._garment_meta.pop(int(gid), None)
        self._scene_meta.pop(int(record["scene_id"]), None)

    # ── vector search ────────────────────────────────────────────────────────

    def search_garments(self, query_vec: np.ndarray, k: int,
                        allowed_ids: Optional[set[str]] = None
                        ) -> list[tuple[str, str, float]]:
        if k <= 0 or self._garment_index.ntotal == 0:
            return []
        faiss_ids: Optional[np.ndarray] = None
        if allowed_ids is not None:
            ids = sorted(
                gid for iid in allowed_ids if iid in self._images
                for gid in self._images[iid]["garment_ids"])
            if not ids:
                return []
            faiss_ids = np.asarray(ids, dtype=np.int64)
        out: list[tuple[str, str, float]] = []
        for fid, score in self._knn(self._garment_index, query_vec,
                                    self.garment_dim, k, faiss_ids):
            meta = self._garment_meta.get(fid)
            if meta is None:
                continue
            out.append((meta[0], meta[1], score))
            if len(out) >= k:
                break
        return out

    def search_scene(self, query_vec: np.ndarray, k: int,
                     allowed_ids: Optional[set[str]] = None
                     ) -> list[tuple[str, float]]:
        if k <= 0 or self._scene_index.ntotal == 0:
            return []
        faiss_ids: Optional[np.ndarray] = None
        if allowed_ids is not None:
            ids = sorted(self._images[iid]["scene_id"]
                         for iid in allowed_ids if iid in self._images)
            if not ids:
                return []
            faiss_ids = np.asarray(ids, dtype=np.int64)
        out: list[tuple[str, float]] = []
        for fid, score in self._knn(self._scene_index, query_vec,
                                    self.scene_dim, k, faiss_ids):
            image_id = self._scene_meta.get(fid)
            if image_id is None:
                continue
            out.append((image_id, score))
            if len(out) >= k:
                break
        return out

    def _knn(self, index: "faiss.Index", query_vec: np.ndarray, dim: int, k: int,
             faiss_ids: Optional[np.ndarray]) -> list[tuple[int, float]]:
        """Top-k (faiss_id, score), optionally restricted to `faiss_ids`.

        Selector-parameter path first (faiss>=1.7.4); over-fetch fallback second.
        Ids of -1 (padding for k > matches) are skipped.
        """
        q = _as_matrix(query_vec, dim, "query")
        if faiss_ids is None:
            scores, ids = index.search(q, k)
            return [(int(i), float(s))
                    for s, i in zip(scores[0], ids[0]) if i != -1][:k]
        try:
            params = faiss.SearchParameters(sel=_id_selector(faiss_ids))
            scores, ids = index.search(q, k, params=params)
            return [(int(i), float(s))
                    for s, i in zip(scores[0], ids[0]) if i != -1][:k]
        except Exception as exc:  # noqa: BLE001 — older faiss builds lack params=
            log.debug("faiss selector search unavailable (%s); "
                      "falling back to over-fetch + python filter", exc)
            allowed = {int(x) for x in faiss_ids}
            fetch = min(max(k * 10, k), int(index.ntotal))
            scores, ids = index.search(q, fetch)
            return [(int(i), float(s)) for s, i in zip(scores[0], ids[0])
                    if i != -1 and int(i) in allowed][:k]

    # ── attribute / metadata access ──────────────────────────────────────────

    def find_images(self, required: list[AttributePredicate],
                    excluded: list[AttributePredicate]) -> set[str]:
        out: set[str] = set()
        for image_id, record in self._images.items():
            attrs: ImageAttributes = record["attributes"]
            if required and not all(p.matches_image(attrs) for p in required):
                continue
            if excluded and any(p.matches_garment(g)
                                for p in excluded for g in attrs.garments):
                continue
            out.add(image_id)
        return out

    def update_attributes(self, image_id: str, attributes: ImageAttributes) -> None:
        record = self._images.get(image_id)
        if record is None:
            raise KeyError(f"image not indexed: {image_id}")
        if not attributes.image_id:
            attributes = attributes.model_copy(update={"image_id": image_id})
        record["attributes"] = attributes

    def get_attributes(self, image_id: str) -> Optional[ImageAttributes]:
        record = self._images.get(image_id)
        return record["attributes"] if record is not None else None

    def get_regions(self, image_id: str) -> list[RegionRecord]:
        record = self._images.get(image_id)
        return list(record["regions"]) if record is not None else []

    def iter_attributes(self) -> Iterable[ImageAttributes]:
        for record in self._images.values():
            yield record["attributes"]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def image_ids(self) -> list[str]:
        return list(self._images.keys())

    def stats(self) -> dict:
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
            "n_images": len(self._images),
            "n_garment_vectors": int(self._garment_index.ntotal),
            "n_scene_vectors": int(self._scene_index.ntotal),
            "garment_dim": self.garment_dim,
            "scene_dim": self.scene_dim,
            "top_garment_types": dict(type_counts.most_common(10)),
            "top_colors": dict(color_counts.most_common(10)),
        }

    def persist(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._garment_index, str(path / _GARMENTS_FILE))
        faiss.write_index(self._scene_index, str(path / _SCENES_FILE))
        meta = {
            "garment_dim": self.garment_dim,
            "scene_dim": self.scene_dim,
            "next_garment_id": self._next_garment_id,
            "next_scene_id": self._next_scene_id,
            "garment_meta": {str(fid): list(pair)
                             for fid, pair in self._garment_meta.items()},
            "scene_meta": {str(fid): iid for fid, iid in self._scene_meta.items()},
            "images": {
                image_id: {
                    "garment_ids": record["garment_ids"],
                    "scene_id": record["scene_id"],
                    "regions": [r.model_dump() for r in record["regions"]],
                    "attributes": record["attributes"].model_dump(),
                }
                for image_id, record in self._images.items()
            },
        }
        with open(path / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        log.info("persisted %d images (%d garment vecs, %d scene vecs) to %s",
                 len(self._images), self._garment_index.ntotal,
                 self._scene_index.ntotal, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._garment_index = faiss.read_index(str(path / _GARMENTS_FILE))
        self._scene_index = faiss.read_index(str(path / _SCENES_FILE))
        with open(path / _META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        loaded_gdim, loaded_sdim = int(meta["garment_dim"]), int(meta["scene_dim"])
        if (loaded_gdim, loaded_sdim) != (self.garment_dim, self.scene_dim):
            log.warning("loaded index dims (%d, %d) differ from configured (%d, %d);"
                        " using loaded dims", loaded_gdim, loaded_sdim,
                        self.garment_dim, self.scene_dim)
        self.garment_dim, self.scene_dim = loaded_gdim, loaded_sdim
        self._next_garment_id = int(meta["next_garment_id"])
        self._next_scene_id = int(meta["next_scene_id"])
        self._garment_meta = {int(fid): (pair[0], pair[1])
                              for fid, pair in meta["garment_meta"].items()}
        self._scene_meta = {int(fid): iid for fid, iid in meta["scene_meta"].items()}
        self._images = {
            image_id: {
                "garment_ids": [int(g) for g in record["garment_ids"]],
                "scene_id": int(record["scene_id"]),
                "regions": [RegionRecord.model_validate(r)
                            for r in record["regions"]],
                "attributes": ImageAttributes.model_validate(record["attributes"]),
            }
            for image_id, record in meta["images"].items()
        }
        log.info("loaded %d images (%d garment vecs, %d scene vecs) from %s",
                 len(self._images), self._garment_index.ntotal,
                 self._scene_index.ntotal, path)
