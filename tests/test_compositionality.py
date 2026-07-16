"""THE compositionality proof.

Vanilla CLIP cannot tell "red shirt with blue pants" from "blue shirt with
red pants". This suite proves the engine handles attribute→garment binding
structurally: mirrored queries return different results, a bag-of-attributes
decoy (white tie + red shirt) is excluded/outranked for "red tie and white
shirt", and negations remove images at the metadata layer. Everything runs
on deterministic fakes — no model downloads, no torch.
"""
from __future__ import annotations

import sys

from core.schemas import SearchResult
from retriever.parsing.rule_parser import RuleParser
from retriever.search.engine import SearchEngine


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.image_id for r in results]


# ── mirrored color binding ───────────────────────────────────────────────────

def test_mirrored_queries_return_different_results(engine: SearchEngine) -> None:
    """'red shirt and blue pants' != 'blue shirt and red pants'."""
    _, res_ab = engine.search("red shirt and blue pants")
    _, res_ba = engine.search("blue shirt and red pants")

    ids_ab, ids_ba = _ids(res_ab), _ids(res_ba)
    assert ids_ab, "mirror query A returned nothing"
    assert ids_ba, "mirror query B returned nothing"

    assert ids_ab[0] == "img_a"
    assert ids_ab[0] != "img_b"
    assert ids_ba[0] == "img_b"
    # the two orderings must differ — this is the assignment's core guarantee
    assert ids_ab != ids_ba


def test_mirrored_query_explanations_bind_terms_to_regions(engine: SearchEngine) -> None:
    """Each query term matches the region that actually carries it."""
    _, res = engine.search("red shirt and blue pants")
    top = res[0]
    assert top.image_id == "img_a"
    term_to_region = {m.query_term: m.region_id for m in top.matches}
    assert term_to_region.get("red shirt") == "img_a#r0"
    assert term_to_region.get("blue pants") == "img_a#r1"


# ── structural binding vs bag-of-attributes ─────────────────────────────────

def test_red_tie_white_shirt_prefilter_excludes_decoy(engine: SearchEngine) -> None:
    """img_d (white tie + red shirt) has the same attribute BAG as img_c
    (red tie + white shirt) but the wrong BINDING — with the hard conjunctive
    prefilter it must be excluded entirely before any vector math."""
    _, res = engine.search("a red tie and a white shirt")
    ids = _ids(res)
    assert ids[0] == "img_c"
    assert "img_d" not in ids


def test_binding_beats_bag_of_attributes_even_without_prefilter(
        fake_store, fake_embedder) -> None:
    """With the prefilter disabled, fusion (garment channel + attribute
    channel) must still rank the correctly-bound image above the decoy."""
    eng = SearchEngine(
        store=fake_store,
        garment_embedder=fake_embedder,
        scene_embedder=fake_embedder,
        parser=RuleParser(),
        reranker=None,
        prefilter_enabled=False,
    )
    _, res = eng.search("a red tie and a white shirt", k=10)
    ids = _ids(res)
    assert "img_c" in ids and "img_d" in ids
    assert ids.index("img_c") < ids.index("img_d")
    assert ids[0] == "img_c"


# ── negation ─────────────────────────────────────────────────────────────────

def test_negation_bound_to_garment_type(engine: SearchEngine) -> None:
    """'shirt, no red shirt' excludes red-SHIRT records only: img_a and img_d
    (red shirts) disappear, while img_b (blue shirt, red pants) and img_c
    (white shirt, red TIE) survive — the negation binds structurally too."""
    _, res = engine.search("shirt, no red shirt")
    ids = _ids(res)
    assert ids, "negated query returned nothing"
    assert "img_a" not in ids
    assert "img_d" not in ids
    assert "img_b" in ids
    assert "img_c" in ids
    for r in res:
        assert r.attributes is not None
        assert not any(g.type == "shirt" and g.color == "red"
                       for g in r.attributes.garments)


def test_bare_color_negation_is_a_hard_exclusion(engine: SearchEngine) -> None:
    """'shirt without red' emits a color-only negation, which the engine
    applies as a hard exclusion of ANY red garment — img_a (red shirt) is out,
    and so is every other image carrying red anywhere (img_b's red pants,
    img_c's red tie, img_d's red shirt). The requirement side softens rather
    than returning nothing."""
    _, res = engine.search("shirt without red")
    ids = _ids(res)
    assert ids, "soft-fallback should keep exclusion-safe images"
    assert "img_a" not in ids
    assert "img_b" not in ids  # red pants — bare-color negation is image-wide
    for r in res:
        assert r.attributes is not None
        assert all(g.color != "red" for g in r.attributes.garments)


# ── hygiene ──────────────────────────────────────────────────────────────────

def test_suite_runs_without_heavy_ml_deps() -> None:
    """The whole retrieval path under test must be importable without torch."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "open_clip" not in sys.modules
