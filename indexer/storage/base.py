"""Multi-vector store interface.

One image = N garment vectors + 1 scene vector + a structured attribute payload,
all joined on image_id. Backends: FAISS (default, in-process), OpenSearch k-NN,
MongoDB Atlas Vector Search — all behind this ABC, selected in config.

ML logic never imports boto3; artifact sync to S3 lives in scripts/, not here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from core.schemas import AttributePredicate, ImageAttributes, RegionRecord


class VectorStore(ABC):
    """Persistence + ANN search over garment vectors, scene vectors and attributes."""

    name: str

    # ── writes ───────────────────────────────────────────────────────────────

    @abstractmethod
    def add_image(self, image_id: str, scene_vec: np.ndarray,
                  garment_vecs: list[tuple[RegionRecord, np.ndarray]],
                  attributes: ImageAttributes) -> None:
        """Idempotent upsert of everything known about one image."""

    # ── vector search ────────────────────────────────────────────────────────

    @abstractmethod
    def search_garments(self, query_vec: np.ndarray, k: int,
                        allowed_ids: Optional[set[str]] = None
                        ) -> list[tuple[str, str, float]]:
        """Top-k garment regions: (image_id, region_id, cosine_sim), best-first.
        If allowed_ids is given, restrict to those image_ids (metadata pre-filter)."""

    @abstractmethod
    def search_scene(self, query_vec: np.ndarray, k: int,
                     allowed_ids: Optional[set[str]] = None
                     ) -> list[tuple[str, float]]:
        """Top-k images by scene vector: (image_id, cosine_sim), best-first."""

    # ── attribute / metadata access ──────────────────────────────────────────

    @abstractmethod
    def find_images(self, required: list[AttributePredicate],
                    excluded: list[AttributePredicate]) -> set[str]:
        """Conjunctive metadata pre-filter.

        An image qualifies iff EVERY predicate in `required` is satisfied by at
        least one of its garment records (per-record type∧color∧material match —
        the structural compositionality guarantee) and NO predicate in `excluded`
        matches any garment record.
        """

    @abstractmethod
    def get_attributes(self, image_id: str) -> Optional[ImageAttributes]: ...

    @abstractmethod
    def get_regions(self, image_id: str) -> list[RegionRecord]: ...

    @abstractmethod
    def iter_attributes(self) -> Iterable[ImageAttributes]:
        """All attribute payloads (admin/stats + soft-filter scoring)."""

    # ── lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def image_ids(self) -> list[str]: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def persist(self, path: str | Path) -> None:
        """Write index artifacts to a local directory (synced to S3 by scripts)."""

    @abstractmethod
    def load(self, path: str | Path) -> None: ...
