"""LLM query parser on Amazon Bedrock (Converse API).

Decomposes long-tail natural-language fashion queries into the same ParsedQuery
schema the RuleParser emits, then canonicalizes everything through core.lexicon
so query-side terms join index-side terms exactly. Any failure — client setup,
throttling, malformed JSON, validation — falls back to the deterministic
RuleParser: retrieval must never fail because parsing did.

boto3 is imported lazily inside methods (this is one of the few modules allowed
to touch it at all).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.lexicon import (
    canonical_color,
    canonical_garment,
    canonical_material,
    canonical_scene,
    canonical_style,
)
from core.schemas import Negation, ParsedQuery, QueryGarment
from indexer.extraction.base import AttributeExtractor  # static _extract_json_block reuse
from retriever.parsing.base import QueryParser
from retriever.parsing.rule_parser import RuleParser

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fashion search query parser. Decompose the user's natural-language \
fashion search query into a strict JSON object with EXACTLY this shape:

{
  "garments": [{"type": "<garment type or null>", "color": "<color name or null>"}],
  "scene": "<environment/location phrase or null>",
  "style": "<formal|business|smart-casual|casual|sport or null>",
  "negations": [{"term": "<raw excluded phrase>", "type": "<garment type or null>", "color": "<color name or null>", "material": "<material or null>"}]
}

Rules:
- Bind each color to the garment it modifies: "red tie and white shirt" means red binds to tie and white binds to shirt — never swap or merge bindings.
- Use lowercase. Prefer these garment types when applicable: shirt, t-shirt, sweater, sweatshirt, cardigan, jacket, coat, vest, dress, jumpsuit, pants, shorts, skirt, tie, hat, scarf, glove, belt, glasses, watch, shoe, sock, tights, bag, umbrella.
- Prefer basic color names: black, white, gray, red, orange, yellow, green, blue, navy, purple, pink, brown, beige, cream, gold, silver, teal, maroon, olive, khaki.
- "scene" is a physical environment or location ("office", "beach", "city street"); "style" is a formality/dress-code word. Do not confuse the two.
- Anything the user excludes ("no X", "without X", "not X") goes into negations; set "material" for fabric exclusions (denim, leather, wool, silk, cotton, linen, fur).
- Output ONLY the JSON object — no prose, no markdown fences.

Examples:

Query: "a red tie and a white shirt in a formal setting"
{"garments": [{"type": "tie", "color": "red"}, {"type": "shirt", "color": "white"}], "scene": null, "style": "formal", "negations": []}

Query: "casual outfit, no denim"
{"garments": [], "scene": null, "style": "casual", "negations": [{"term": "denim", "type": null, "color": null, "material": "denim"}]}

Query: "woman in a navy blazer walking down a city street"
{"garments": [{"type": "jacket", "color": "navy"}], "scene": "city street", "style": null, "negations": []}"""


class BedrockParser(QueryParser):
    """Query parser backed by a Bedrock chat model via the Converse API."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str = "us-east-1"):
        self.model_id = model_id
        self.region = region
        self._client: Any = None
        self._fallback = RuleParser()

    # ── public API ───────────────────────────────────────────────────────────

    def parse(self, query: str) -> ParsedQuery:
        """Parse via Bedrock; on ANY failure fall back to the rule parser."""
        try:
            raw_text = self._converse(query)
            data = json.loads(AttributeExtractor._extract_json_block(raw_text))
            parsed = self._validate(query, data)
            return self._canonicalize(parsed)
        except Exception as e:  # noqa: BLE001 — retrieval must never fail on parsing
            log.warning("bedrock parse failed for %r (%s: %s); falling back to rule parser",
                        query, type(e).__name__, e)
            return self._fallback.parse(query)

    # ── Bedrock call ─────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # lazy: only module family allowed to import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _converse(self, query: str) -> str:
        resp = self._get_client().converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": query}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 400},
        )
        blocks = resp["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks)

    # ── validation + canonicalization ────────────────────────────────────────

    @staticmethod
    def _validate(query: str, data: dict) -> ParsedQuery:
        """Sanitize the LLM payload into a ParsedQuery (raw always = query)."""
        negations: list[dict] = []
        for n in data.get("negations") or []:
            if not isinstance(n, dict):
                continue
            term = n.get("term") or n.get("material") or n.get("type") or n.get("color")
            if not term:
                continue
            negations.append({"term": str(term), "type": n.get("type"),
                              "color": n.get("color"), "material": n.get("material")})
        payload = {
            "raw": query,
            "garments": [g for g in (data.get("garments") or []) if isinstance(g, dict)],
            "scene": data.get("scene") or None,
            "style": data.get("style") or None,
            "negations": negations,
        }
        return ParsedQuery.model_validate(payload)

    @staticmethod
    def _canonicalize(parsed: ParsedQuery) -> ParsedQuery:
        """Map every field into the shared lexicon; drop unresolvable garments."""
        garments: list[QueryGarment] = []
        for g in parsed.garments:
            ctype = canonical_garment(g.type)
            ccolor = canonical_color(g.color)
            if ctype is None and ccolor is None:
                continue  # neither type nor color resolved — useless as a term
            garments.append(QueryGarment(type=ctype, color=ccolor))
        parsed.garments = garments

        if parsed.scene:
            # keep the raw phrase when out-of-lexicon: it still feeds the scene
            # embedding channel; the attribute scorer re-canonicalizes anyway
            parsed.scene = canonical_scene(parsed.scene) or parsed.scene
        parsed.style = canonical_style(parsed.style) if parsed.style else None

        parsed.negations = [
            Negation(
                term=n.term,
                type=canonical_garment(n.type) or canonical_garment(n.term),
                color=canonical_color(n.color),
                material=canonical_material(n.material) or canonical_material(n.term),
            )
            for n in parsed.negations
        ]
        return parsed
