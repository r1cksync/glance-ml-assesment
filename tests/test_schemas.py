"""Schema-level guarantees: GarmentAttribute normalization, the structural
AttributePredicate conjunction that powers compositionality, and ParsedQuery
serialization round-trips (the parse-cache path). Pure pydantic — no ML deps.
"""
from __future__ import annotations

import json

from core.schemas import (
    AttributePredicate,
    GarmentAttribute,
    ImageAttributes,
    Negation,
    ParsedQuery,
    QueryGarment,
)


# ── GarmentAttribute normalization ───────────────────────────────────────────

def test_garment_attribute_normalizes_fields() -> None:
    g = GarmentAttribute(type="  Shirt ", color="RED", color_hex="FF0000",
                         formality="Business", material=" Denim ")
    assert g.type == "shirt"
    assert g.color == "red"
    assert g.color_hex == "#ff0000"
    assert g.formality == "business"
    assert g.material == "denim"


def test_garment_attribute_hex_coercion() -> None:
    assert GarmentAttribute(type="hat", color="black",
                            color_hex="#ABCdef").color_hex == "#abcdef"
    assert GarmentAttribute(type="hat", color="black",
                            color_hex="12345").color_hex is None   # wrong length
    assert GarmentAttribute(type="hat", color="black",
                            color_hex="zzzzzz").color_hex is None  # non-hex
    assert GarmentAttribute(type="hat", color="black",
                            color_hex="").color_hex is None


def test_garment_attribute_formality_coercion() -> None:
    assert GarmentAttribute(type="tie", color="red",
                            formality="Smart-Casual").formality == "smart-casual"
    assert GarmentAttribute(type="tie", color="red",
                            formality="fancy dress").formality == "unknown"
    assert GarmentAttribute(type="tie", color="red",
                            formality="").formality == "unknown"
    assert GarmentAttribute(type="tie", color="red").formality == "unknown"


def test_garment_attribute_material_coercion() -> None:
    assert GarmentAttribute(type="pants", color="blue",
                            material="").material is None
    assert GarmentAttribute(type="pants", color="blue").material is None


# ── structural predicate conjunction ─────────────────────────────────────────

def _image(image_id: str, garments: list[tuple[str, str]]) -> ImageAttributes:
    return ImageAttributes(
        image_id=image_id,
        garments=[GarmentAttribute(type=t, color=c) for t, c in garments],
    )


def test_predicate_matches_per_record_conjunction() -> None:
    """type AND color must hold on a SINGLE garment record — the guarantee
    that separates 'red tie + white shirt' from 'white tie + red shirt'."""
    bound = _image("bound", [("tie", "red"), ("shirt", "white")])
    swapped = _image("swapped", [("tie", "white"), ("shirt", "red")])

    red_tie = AttributePredicate(types=["tie"], colors=["red"])
    white_shirt = AttributePredicate(types=["shirt"], colors=["white"])

    assert red_tie.matches_image(bound)
    assert white_shirt.matches_image(bound)
    # same bag of attributes, wrong binding — must NOT satisfy either predicate
    assert not red_tie.matches_image(swapped)
    assert not white_shirt.matches_image(swapped)


def test_predicate_material_conjunct() -> None:
    denim_pants = GarmentAttribute(type="pants", color="blue", material="denim")
    plain_pants = GarmentAttribute(type="pants", color="blue")
    pred = AttributePredicate(types=["pants"], materials=["denim"])
    assert pred.matches_garment(denim_pants)
    assert not pred.matches_garment(plain_pants)  # missing material fails


def test_predicate_on_empty_image_never_matches() -> None:
    empty = ImageAttributes(image_id="empty", garments=[])
    assert not AttributePredicate().matches_image(empty)
    assert not AttributePredicate(types=["tie"]).matches_image(empty)


# ── ParsedQuery round-trip (parse-cache path) ────────────────────────────────

def test_parsed_query_round_trip() -> None:
    pq = ParsedQuery(
        raw="A red tie and a white shirt in a formal setting, no denim",
        garments=[QueryGarment(type="tie", color="red"),
                  QueryGarment(type="shirt", color="white")],
        scene="modern office",
        style="formal",
        negations=[Negation(term="denim", material="denim")],
    )
    assert ParsedQuery.model_validate(pq.model_dump()) == pq
    assert ParsedQuery.model_validate(json.loads(pq.model_dump_json())) == pq


def test_query_garment_text_rendering() -> None:
    assert QueryGarment(type="tie", color="red").text() == "red tie"
    assert QueryGarment(type="shirt").text() == "shirt"
    assert QueryGarment(color="green").text() == "green"
    assert QueryGarment().text() == ""

    assert ParsedQuery(raw="x", garments=[QueryGarment(color="green")]).has_garment_terms()
    assert not ParsedQuery(raw="x").has_garment_terms()
