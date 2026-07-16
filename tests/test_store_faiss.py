"""FaissStore contract tests: exact self-match, allowed_ids restriction,
conjunctive/exclusion metadata filtering, persist/load roundtrip and
idempotent re-adds. Deterministic (seeded rng), no ML model imports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.schemas import (
    AttributePredicate,
    GarmentAttribute,
    ImageAttributes,
    RegionRecord,
)
from indexer.storage.faiss_store import FaissStore

DIM = 32

#: (image_id, scene_vec, [(RegionRecord, vec)], ImageAttributes)
Payload = tuple[str, np.ndarray, list[tuple[RegionRecord, np.ndarray]], ImageAttributes]

# garments per image: (type, color, material)
_SPECS: dict[str, list[tuple[str, str, str | None]]] = {
    "s1": [("tie", "red", None), ("shirt", "white", None)],
    "s2": [("tie", "white", None), ("shirt", "red", None)],  # swapped binding
    "s3": [("pants", "blue", "denim")],
}
_SCENE_TYPES = {"s1": "office", "s2": "street", "s3": "park"}


def _unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _payloads() -> list[Payload]:
    """Deterministic corpus: fixed rng seed, fixed draw order."""
    rng = np.random.default_rng(7)
    payloads: list[Payload] = []
    for image_id, garments in _SPECS.items():
        scene_vec = _unit(rng)
        gvecs: list[tuple[RegionRecord, np.ndarray]] = []
        for i, (gtype, color, material) in enumerate(garments):
            record = RegionRecord(
                image_id=image_id,
                region_id=f"{image_id}#r{i}",
                bbox=(50.0 * i, 0.0, 50.0 * i + 40.0, 90.0),
                label=gtype,
                det_score=0.9,
            )
            gvecs.append((record, _unit(rng)))
        attrs = ImageAttributes(
            image_id=image_id,
            garments=[GarmentAttribute(type=t, color=c, material=m)
                      for t, c, m in garments],
            scene=f"{_SCENE_TYPES[image_id]} scene",
            scene_type=_SCENE_TYPES[image_id],
        )
        payloads.append((image_id, scene_vec, gvecs, attrs))
    return payloads


def _region_vecs(payloads: list[Payload]) -> dict[str, np.ndarray]:
    return {rec.region_id: vec for _, _, gvecs, _ in payloads for rec, vec in gvecs}


def _scene_vecs(payloads: list[Payload]) -> dict[str, np.ndarray]:
    return {image_id: svec for image_id, svec, _, _ in payloads}


@pytest.fixture()
def payloads() -> list[Payload]:
    return _payloads()


@pytest.fixture()
def store(payloads: list[Payload]) -> FaissStore:
    s = FaissStore(DIM, DIM)
    for image_id, scene_vec, gvecs, attrs in payloads:
        s.add_image(image_id, scene_vec, gvecs, attrs)
    return s


# ── vector roundtrip ─────────────────────────────────────────────────────────

def test_garment_search_self_match_first(store: FaissStore,
                                          payloads: list[Payload]) -> None:
    for _, _, gvecs, _ in payloads:
        for record, vec in gvecs:
            hits = store.search_garments(vec, 3)
            assert hits, "no hits for a stored region vector"
            image_id, region_id, sim = hits[0]
            assert (image_id, region_id) == (record.image_id, record.region_id)
            assert sim == pytest.approx(1.0, abs=1e-4)
            sims = [h[2] for h in hits]
            assert sims == sorted(sims, reverse=True), "hits must be best-first"


def test_scene_search_self_match_first(store: FaissStore,
                                       payloads: list[Payload]) -> None:
    for image_id, scene_vec in _scene_vecs(payloads).items():
        hits = store.search_scene(scene_vec, 3)
        assert hits[0][0] == image_id
        assert hits[0][1] == pytest.approx(1.0, abs=1e-4)


def test_allowed_ids_restricts_garment_search(store: FaissStore,
                                              payloads: list[Payload]) -> None:
    probe = _region_vecs(payloads)["s1#r0"]
    unrestricted = store.search_garments(probe, 10)
    assert unrestricted[0][0] == "s1"

    hits = store.search_garments(probe, 10, allowed_ids={"s2", "s3"})
    assert hits, "restricted search returned nothing"
    assert all(h[0] in {"s2", "s3"} for h in hits)
    assert not any(h[0] == "s1" for h in hits)


def test_allowed_ids_restricts_scene_search(store: FaissStore,
                                            payloads: list[Payload]) -> None:
    probe = _scene_vecs(payloads)["s1"]
    hits = store.search_scene(probe, 10, allowed_ids={"s3"})
    assert hits
    assert all(h[0] == "s3" for h in hits)


# ── metadata prefilter ───────────────────────────────────────────────────────

def test_find_images_conjunctive_two_garment_and(store: FaissStore) -> None:
    """Both predicates must be satisfied by single records on the SAME image:
    s1 (red tie + white shirt) qualifies; s2 has the identical attribute bag
    with swapped bindings and must not."""
    required = [
        AttributePredicate(types=["tie"], colors=["red"]),
        AttributePredicate(types=["shirt"], colors=["white"]),
    ]
    assert store.find_images(required, []) == {"s1"}


def test_find_images_single_predicate(store: FaissStore) -> None:
    assert store.find_images([AttributePredicate(types=["tie"])], []) == {"s1", "s2"}


def test_find_images_exclusions(store: FaissStore) -> None:
    no_red = [AttributePredicate(colors=["red"])]
    assert store.find_images([], no_red) == {"s3"}

    no_denim = [AttributePredicate(materials=["denim"])]
    assert store.find_images([], no_denim) == {"s1", "s2"}

    # required + exclusion together: every tie image carries red somewhere
    assert store.find_images([AttributePredicate(types=["tie"])], no_red) == set()


# ── persistence ──────────────────────────────────────────────────────────────

def test_persist_load_roundtrip(store: FaissStore, payloads: list[Payload],
                                tmp_path: Path) -> None:
    target = tmp_path / "store"
    target.mkdir(parents=True, exist_ok=True)
    store.persist(target)

    fresh = FaissStore(DIM, DIM)
    fresh.load(target)

    assert fresh.stats() == store.stats()
    assert sorted(fresh.image_ids()) == sorted(store.image_ids())
    assert len(list(fresh.iter_attributes())) == len(payloads)

    probe = _region_vecs(payloads)["s2#r0"]
    orig = store.search_garments(probe, 5)
    loaded = fresh.search_garments(probe, 5)
    assert [(h[0], h[1]) for h in orig] == [(h[0], h[1]) for h in loaded]
    for a, b in zip(orig, loaded):
        assert a[2] == pytest.approx(b[2], abs=1e-5)

    sprobe = _scene_vecs(payloads)["s1"]
    orig_s = store.search_scene(sprobe, 3)
    loaded_s = fresh.search_scene(sprobe, 3)
    assert [h[0] for h in orig_s] == [h[0] for h in loaded_s]
    for a, b in zip(orig_s, loaded_s):
        assert a[1] == pytest.approx(b[1], abs=1e-5)

    for image_id in ("s1", "s2", "s3"):
        assert fresh.get_attributes(image_id) == store.get_attributes(image_id)
        by_rid = lambda rs: sorted(rs, key=lambda r: r.region_id)  # noqa: E731
        assert by_rid(fresh.get_regions(image_id)) == by_rid(store.get_regions(image_id))


# ── idempotency ──────────────────────────────────────────────────────────────

def test_readd_same_image_is_idempotent(store: FaissStore,
                                        payloads: list[Payload]) -> None:
    image_id, scene_vec, gvecs, attrs = payloads[0]
    before_stats = store.stats()
    before_ids = sorted(store.image_ids())

    store.add_image(image_id, scene_vec, gvecs, attrs)  # exact re-add

    assert store.stats() == before_stats
    assert sorted(store.image_ids()) == before_ids

    probe = _region_vecs(payloads)["s1#r0"]
    hits = store.search_garments(probe, 10)
    region_ids = [h[1] for h in hits]
    assert len(region_ids) == len(set(region_ids)), "duplicate regions after re-add"
    assert (hits[0][0], hits[0][1]) == ("s1", "s1#r0")

    shits = store.search_scene(_scene_vecs(payloads)["s1"], 10)
    scene_ids = [h[0] for h in shits]
    assert len(scene_ids) == len(set(scene_ids)), "duplicate scene rows after re-add"
    assert shits[0][0] == "s1"
