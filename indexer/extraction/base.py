"""VLM structured-attribute extraction: strict JSON, pydantic-validated,
with a repair-retry loop. Subclasses implement only `_generate`."""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.lexicon import canonical_color, canonical_garment, canonical_material
from core.schemas import GarmentAttribute, ImageAttributes

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a fashion attribute extractor. Look at the image and output ONLY a JSON object, no prose, matching exactly this schema:

{
  "garments": [
    {"type": "<garment type, e.g. shirt|t-shirt|sweater|sweatshirt|cardigan|jacket|coat|vest|dress|jumpsuit|pants|shorts|skirt|tie|hat|scarf|glasses|belt|shoe|bag>",
     "color": "<dominant color name>",
     "color_hex": "<approx hex like #aabbcc>",
     "formality": "<formal|business|smart-casual|casual|sport>",
     "material": "<denim|leather|wool|silk|cotton|linen|fur or null>"}
  ],
  "scene": "<one short sentence describing the environment>",
  "scene_type": "<office|street|park|home|beach|restaurant|studio|indoor|outdoor|other>",
  "lighting": "<bright|dim|natural|artificial|studio>"
}

List every clearly visible garment worn by the main person (max 6). List each distinct garment exactly ONCE — never repeat a garment. Use lowercase. Output valid JSON only."""

REPAIR_PROMPT = """Your previous output was not valid JSON for the required schema. Error: {error}

Previous output:
{previous}

Output ONLY the corrected JSON object, nothing else."""


class ExtractionError(Exception):
    pass


class AttributeExtractor(ABC):
    """Base class: subclasses provide `_generate(image, prompt) -> str`."""

    name: str

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    @abstractmethod
    def _generate(self, image: "Image", prompt: str) -> str:
        """Run the VLM once and return raw text."""

    def extract(self, image: "Image", image_id: str = "") -> ImageAttributes:
        """Extract attributes with validation + repair-retry. Raises ExtractionError
        if all attempts fail (callers may fall back to detector-derived attributes)."""
        prompt = EXTRACTION_PROMPT
        raw = ""
        last_err: Exception | None = None
        for attempt in range(1 + self.max_retries):
            raw = self._generate(image, prompt)
            try:
                attrs = self._parse(raw)
                attrs.image_id = image_id
                return attrs
            except Exception as e:  # noqa: BLE001 — any parse/validation failure triggers repair
                last_err = e
                log.warning("extraction parse failed (attempt %d) for %s: %s",
                            attempt + 1, image_id, e)
                prompt = EXTRACTION_PROMPT + "\n\n" + REPAIR_PROMPT.format(
                    error=str(e)[:300], previous=raw[:1500])
        raise ExtractionError(f"extraction failed for {image_id}: {last_err}")

    # ── parsing helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Pull the first JSON object out of possibly-noisy model output."""
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            return fence.group(1)
        start = text.find("{")
        if start == -1:
            raise ValueError("no JSON object in output")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        raise ValueError("unbalanced JSON braces in output")

    @staticmethod
    def _salvage_truncated_json(text: str) -> dict | None:
        """Best-effort recovery of a max_new_tokens-truncated JSON object: cut
        back to the last complete value and append closers. Returns None if
        nothing parseable emerges."""
        start = text.find("{")
        if start == -1:
            return None
        body = text[start:]
        for cut in (len(body), body.rfind("}"), body.rfind('"')):
            if cut is None or cut <= 0:
                continue
            base = body[:cut + 1] if body[cut - 1:cut] != "" else body[:cut]
            # strip a trailing comma / partial token, then try closer suffixes
            trimmed = base.rstrip().rstrip(",")
            for suffix in ("", "}", "]}", "}]}", '"}]}', "}}", "]}}"):
                try:
                    obj = json.loads(trimmed + suffix)
                    if isinstance(obj, dict):
                        return obj
                except Exception:  # noqa: BLE001
                    continue
        return None

    def _parse(self, raw: str) -> ImageAttributes:
        try:
            data = json.loads(self._extract_json_block(raw))
        except Exception:
            data = self._salvage_truncated_json(raw)
            if data is None:
                raise
        attrs = ImageAttributes.model_validate(data)
        # canonicalize into the shared lexicon so index-side terms join query-side
        # terms, and dedupe (small VLMs sometimes emit the same garment repeatedly)
        garments: list[GarmentAttribute] = []
        seen: set[tuple] = set()
        for g in attrs.garments:
            ctype = canonical_garment(g.type)
            if ctype is None:
                continue  # not a garment we track — drop rather than pollute the index
            g.type = ctype
            g.color = canonical_color(g.color, g.color_hex) or g.color
            g.material = canonical_material(g.material)
            key = (g.type, g.color, g.formality, g.material)
            if key in seen:
                continue
            seen.add(key)
            garments.append(g)
        attrs.garments = garments
        # scene_type sanity: trust the lexicon over the VLM's coarse guess when
        # the free-text scene names a known place ("on a runway" -> studio)
        from core.lexicon import canonical_scene  # local import: avoids cycle at module load
        derived = canonical_scene(attrs.scene)
        if derived and attrs.scene_type in ("other", "indoor", "outdoor"):
            attrs.scene_type = derived
        return attrs
