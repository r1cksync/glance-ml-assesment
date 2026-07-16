"""Shared fixtures for the retrieval test suite.

Everything here runs WITHOUT model downloads or torch/transformers imports:

- ``FakeEmbedder`` is a deterministic stand-in for FashionCLIP/CLIP. Every
  vocabulary key maps to a stable random unit vector (seeded from a sha256
  hash of the key, so it is identical across processes and platforms).
  ``embed_texts`` sums the vectors of all vocab keys found in the text
  (longest-first, consuming matched spans), which makes phrase similarity
  exact by construction: "a photo of a person wearing red shirt" lands on
  the "red shirt" anchor and nothing else.

- ``fake_store`` is a tiny synthetic corpus that encodes the compositional
  decoys the engine must tell apart structurally:
    img_a  red shirt + blue pants          (street)
    img_b  blue shirt + red pants          (street, mirror of img_a)
    img_c  red tie + white shirt, business (office)
    img_d  white tie + red shirt           (street, bag-of-attributes decoy)
    img_e  casual gray t-shirt             (park)
    img_f  yellow coat                     (street)

- ``engine`` wires the real SearchEngine + RuleParser over the fakes with
  ``prefilter_min_candidates=1`` so the tiny corpus still hard-filters.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.schemas import GarmentAttribute, ImageAttributes, RegionRecord  # noqa: E402
from indexer.models.base import EmbeddingModel  # noqa: E402
from indexer.storage.faiss_store import FaissStore  # noqa: E402
from retriever.parsing.rule_parser import RuleParser  # noqa: E402
from retriever.search.engine import SearchEngine  # noqa: E402

DIM = 64


class FakeEmbedder(EmbeddingModel):
    """Deterministic vocabulary-anchored embedder (text-only, no ML deps)."""

    name = "fake"
    dim = DIM

    #: every color+type combination the tests need, plus bare types/colors
    #: and scene phrases. Longest keys are matched first and consume their
    #: span so "red shirt" never double-counts as "red" + "shirt".
    VOCAB: tuple[str, ...] = (
        # color + type combinations
        "red shirt", "blue pants", "blue shirt", "red pants",
        "red tie", "white shirt", "yellow coat",
        # bare garment types ("t-shirt" before "shirt" via longest-first)
        "t-shirt", "shirt", "pants", "tie", "coat",
        # bare colors
        "red", "blue", "white", "yellow", "green", "gray",
        # scene phrases
        "park", "office", "street",
    )

    @staticmethod
    def _seed(key: str) -> int:
        """Stable cross-process hash (builtin hash() is salted per process)."""
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")

    @classmethod
    def vector(cls, key: str) -> np.ndarray:
        """The stable L2-normalized anchor vector for one vocabulary key."""
        rng = np.random.default_rng(cls._seed(key))
        v = rng.standard_normal(cls.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    @classmethod
    def _match_keys(cls, text: str) -> list[str]:
        """All vocab keys contained in the text, longest-first, spans consumed."""
        t = " ".join(text.lower().split())
        found: list[str] = []
        for key in sorted(cls.VOCAB, key=lambda k: (-len(k), k)):
            if key in t:
                found.append(key)
                t = t.replace(key, "\x00")  # consume so substrings can't re-match
        return found

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            keys = self._match_keys(text)
            if keys:
                v = np.sum([self.vector(k) for k in keys], axis=0)
            else:
                v = self.vector(text)
            out[i] = v / np.linalg.norm(v)
        return out

    def embed_images(self, images: Sequence[object]) -> np.ndarray:
        raise NotImplementedError("FakeEmbedder is text-only; the engine does "
                                  "not embed images for text search.")


def noisy(vec: np.ndarray, rng: np.random.Generator, scale: float = 0.01) -> np.ndarray:
    """Add tiny gaussian noise and renormalize (region vectors ~ their anchor)."""
    v = vec.astype(np.float32) + scale * rng.standard_normal(vec.shape).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── synthetic corpus ─────────────────────────────────────────────────────────
# (image_id, scene vocab key, scene_type, [(label, region phrase, garment attrs)])
_CORPUS: list[tuple[str, str, str, list[tuple[str, str, dict[str, str]]]]] = [
    ("img_a", "street", "street", [
        ("shirt", "red shirt", {"type": "shirt", "color": "red"}),
        ("pants", "blue pants", {"type": "pants", "color": "blue"}),
    ]),
    ("img_b", "street", "street", [
        ("shirt", "blue shirt", {"type": "shirt", "color": "blue"}),
        ("pants", "red pants", {"type": "pants", "color": "red"}),
    ]),
    ("img_c", "office", "office", [
        ("tie", "red tie", {"type": "tie", "color": "red", "formality": "business"}),
        ("shirt", "white shirt", {"type": "shirt", "color": "white", "formality": "business"}),
    ]),
    ("img_d", "street", "street", [  # the compositional decoy: same bag of attributes as img_c
        ("tie", "white tie", {"type": "tie", "color": "white"}),
        ("shirt", "red shirt", {"type": "shirt", "color": "red"}),
    ]),
    ("img_e", "park", "park", [
        ("t-shirt", "t-shirt", {"type": "t-shirt", "color": "gray", "formality": "casual"}),
    ]),
    ("img_f", "street", "street", [
        ("coat", "yellow coat", {"type": "coat", "color": "yellow"}),
    ]),
]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def fake_store(fake_embedder: FakeEmbedder) -> FaissStore:
    """FaissStore(64, 64) populated with the synthetic compositional corpus."""
    store = FaissStore(DIM, DIM)
    rng = np.random.default_rng(123)
    for image_id, scene_key, scene_type, regions in _CORPUS:
        garment_vecs: list[tuple[RegionRecord, np.ndarray]] = []
        garments: list[GarmentAttribute] = []
        for i, (label, phrase, attrs) in enumerate(regions):
            record = RegionRecord(
                image_id=image_id,
                region_id=f"{image_id}#r{i}",
                bbox=(50.0 * i, 10.0, 50.0 * i + 40.0, 90.0),
                label=label,
                det_score=0.9,
            )
            garment_vecs.append((record, noisy(fake_embedder.embed_text(phrase), rng)))
            garments.append(GarmentAttribute(**attrs))
        attributes = ImageAttributes(
            image_id=image_id,
            garments=garments,
            scene=scene_key,
            scene_type=scene_type,
        )
        store.add_image(image_id, FakeEmbedder.vector(scene_key), garment_vecs, attributes)
    return store


@pytest.fixture()
def engine(fake_store: FaissStore, fake_embedder: FakeEmbedder) -> SearchEngine:
    """Real SearchEngine over the fakes; min_candidates=1 keeps the hard filter
    active on the 6-image corpus."""
    return SearchEngine(
        store=fake_store,
        garment_embedder=fake_embedder,
        scene_embedder=fake_embedder,
        parser=RuleParser(),
        reranker=None,
        prefilter_min_candidates=1,
    )
