"""RuleParser expectations for the five acceptance queries plus the binding,
negation and color-only-refinement behaviors the engine depends on.

The parser is fully deterministic (lexicon + regex), so these assertions pin
its contract exactly — no fuzz, no network.
"""
from __future__ import annotations

import pytest

from core.lexicon import canonical_scene
from core.schemas import QueryGarment
from retriever.parsing.rule_parser import RuleParser


@pytest.fixture(scope="module")
def parser() -> RuleParser:
    return RuleParser()


# ── the five acceptance queries ──────────────────────────────────────────────

def test_q1_attribute_specific(parser: RuleParser) -> None:
    p = parser.parse("A person in a bright yellow raincoat.")
    assert p.garments == [QueryGarment(type="coat", color="yellow")]
    assert p.negations == []
    assert p.scene is None


def test_q2_contextual_place(parser: RuleParser) -> None:
    p = parser.parse("Professional business attire inside a modern office.")
    assert p.garments == []
    assert p.style == "business"
    assert p.scene is not None and "office" in p.scene


def test_q3_complex_semantic(parser: RuleParser) -> None:
    p = parser.parse("Someone wearing a blue shirt sitting on a park bench.")
    assert QueryGarment(type="shirt", color="blue") in p.garments
    assert p.scene is not None and "park" in p.scene
    assert canonical_scene(p.scene) == "park"


def test_q4_style_inference(parser: RuleParser) -> None:
    p = parser.parse("Casual weekend outfit for a city walk.")
    assert p.style == "casual"
    assert p.garments == []
    assert p.scene is not None
    assert canonical_scene(p.scene) == "street"


def test_q5_compositional(parser: RuleParser) -> None:
    p = parser.parse("A red tie and a white shirt in a formal setting.")
    assert QueryGarment(type="tie", color="red") in p.garments
    assert QueryGarment(type="shirt", color="white") in p.garments
    assert p.style in {"formal", "business"}


# ── color→garment binding ────────────────────────────────────────────────────

def test_color_binding_is_positional(parser: RuleParser) -> None:
    """Mirrored queries must parse to swapped bindings, not the same bag."""
    p1 = parser.parse("red shirt with blue pants")
    p2 = parser.parse("blue shirt with red pants")
    assert p1.garments == [QueryGarment(type="shirt", color="red"),
                           QueryGarment(type="pants", color="blue")]
    assert p2.garments == [QueryGarment(type="shirt", color="blue"),
                           QueryGarment(type="pants", color="red")]
    assert p1.garments != p2.garments


# ── negation ─────────────────────────────────────────────────────────────────

def test_negation_no_denim(parser: RuleParser) -> None:
    p = parser.parse("casual outfit, no denim")
    assert p.style == "casual"
    assert len(p.negations) == 1
    neg = p.negations[0]
    assert neg.material == "denim"
    assert neg.term == "denim"


def test_negation_bound_type_and_color(parser: RuleParser) -> None:
    p = parser.parse("shirt, no red shirt")
    assert QueryGarment(type="shirt", color=None) in p.garments
    assert len(p.negations) == 1
    neg = p.negations[0]
    assert neg.type == "shirt"
    assert neg.color == "red"


# ── color-only refinement ("but in green") ───────────────────────────────────

def test_color_only_refinement(parser: RuleParser) -> None:
    p = parser.parse("but in green")
    assert p.garments == [QueryGarment(type=None, color="green")]
    assert p.has_garment_terms()
