"""Deterministic lexicon-based query parser.

Default parser: free, ~1ms, fully testable, no external calls. The Bedrock LLM
parser (retriever/parsing/bedrock_parser.py) is a config-swappable upgrade for
long-tail phrasing; both emit the same ParsedQuery schema.

Binding rule: a color attaches to the NEAREST FOLLOWING garment mention within
the same noun phrase (no separator like 'and', ',', 'in', 'with' in between) —
this is what keeps "red shirt with blue pants" ≠ "blue shirt with red pants".
"""
from __future__ import annotations

import re

from core.lexicon import (
    CANONICAL_COLORS,
    COLOR_ALIASES,
    GARMENT_TYPES,
    MATERIALS,
    SCENE_LEXICON,
    STYLE_LEXICON,
    canonical_color,
    canonical_garment,
    canonical_material,
    canonical_scene,
    canonical_style,
)
from core.schemas import Negation, ParsedQuery, QueryGarment
from retriever.parsing.base import QueryParser

_SEPARATORS = {"and", "with", "in", "on", "at", "inside", "wearing", "under",
               "over", "beside", "plus", ","}
_SCENE_PRE_STOP = {"a", "an", "the", "in", "on", "at", "inside", "into", "of",
                   "for", "to", "near", "by", "some", "this", "that"}


def _alternation(phrases: list[str]) -> str:
    """Regex alternation, longest-first so multi-word aliases win."""
    return "|".join(re.escape(p) for p in sorted(set(phrases), key=len, reverse=True))


_ALL_COLORS = list(CANONICAL_COLORS) + list(COLOR_ALIASES)
_ALL_GARMENTS = [a for aliases in GARMENT_TYPES.values() for a in aliases]
_ALL_MATERIALS = [a for aliases in MATERIALS.values() for a in aliases]
_ALL_SCENES = [a for aliases in SCENE_LEXICON.values() for a in aliases]
_ALL_STYLES = [a for aliases in STYLE_LEXICON.values() for a in aliases]

_COLOR_RE = re.compile(rf"\b({_alternation(_ALL_COLORS)})\b")
_GARMENT_RE = re.compile(rf"\b({_alternation(_ALL_GARMENTS)})\b")
_SCENE_RE = re.compile(rf"\b({_alternation(_ALL_SCENES)})\b")
_STYLE_RE = re.compile(rf"\b({_alternation(_ALL_STYLES)})\b")
_NEGATION_RE = re.compile(
    r"\b(?:no|without|not|except|avoid|excluding)\s+"
    r"((?:[a-z][a-z-]*\s?){1,3}?)(?=\s*(?:,|\.|;|$|\band\b|\bwith\b|\bin\b|\bon\b|\bfor\b))"
)


class RuleParser(QueryParser):
    name = "rule"

    def parse(self, query: str) -> ParsedQuery:
        text = " ".join(query.lower().replace("’", "'").split())
        text = re.sub(r"[!?.]+$", "", text)

        # 1 — negations first (then removed, so 'no denim' can't bind elsewhere)
        negations: list[Negation] = []
        def _consume_negation(m: re.Match) -> str:
            phrase = m.group(1).strip()
            neg = Negation(
                term=phrase,
                type=canonical_garment(self._last_match(_GARMENT_RE, phrase)),
                color=canonical_color(self._last_match(_COLOR_RE, phrase)),
                material=canonical_material(self._last_match(
                    re.compile(rf"\b({_alternation(_ALL_MATERIALS)})\b"), phrase)),
            )
            if neg.type or neg.color or neg.material:
                negations.append(neg)
            return " "
        text_wo_neg = _NEGATION_RE.sub(_consume_negation, text)

        # 2 — garments with nearest-preceding-color binding
        garments: list[QueryGarment] = []
        used_color_spans: list[tuple[int, int]] = []
        color_matches = list(_COLOR_RE.finditer(text_wo_neg))
        for gm in _GARMENT_RE.finditer(text_wo_neg):
            color = None
            for cm in reversed(color_matches):
                if cm.end() > gm.start():
                    continue
                between = text_wo_neg[cm.end():gm.start()]
                tokens = between.replace(",", " , ").split()
                if len(tokens) <= 2 and not (set(tokens) & _SEPARATORS):
                    color = cm.group(1)
                    used_color_spans.append(cm.span())
                    break
            g = QueryGarment(type=canonical_garment(gm.group(1)),
                             color=canonical_color(color) if color else None)
            if g.type and g not in garments:
                garments.append(g)

        # color-only intent: "…but in green" (refinements), "something red"
        if not garments:
            for cm in color_matches:
                if cm.span() not in used_color_spans:
                    g = QueryGarment(type=None, color=canonical_color(cm.group(1)))
                    if g.color and g not in garments:
                        garments.append(g)

        # 3 — scene: matched alias plus one preceding modifier ("modern office")
        scene = None
        sm = _SCENE_RE.search(text_wo_neg)
        if sm:
            scene = sm.group(1)
            before = text_wo_neg[:sm.start()].rstrip()
            prev = before.split()[-1] if before.split() else ""
            if (prev and prev not in _SCENE_PRE_STOP
                    and not _COLOR_RE.fullmatch(prev)
                    and not _GARMENT_RE.fullmatch(prev)
                    and not _STYLE_RE.fullmatch(prev)
                    and prev not in _SEPARATORS):
                scene = f"{prev} {scene}"

        # 4 — style
        style = None
        stm = _STYLE_RE.search(text_wo_neg)
        if stm:
            style = canonical_style(stm.group(1))

        return ParsedQuery(raw=query, garments=garments, scene=scene,
                           style=style, negations=negations)

    @staticmethod
    def _last_match(pattern: re.Pattern, text: str) -> str | None:
        hits = pattern.findall(text)
        return hits[-1] if hits else None
