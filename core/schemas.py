"""Shared data contracts for the whole system.

Every module (indexer, retriever, eval, api) speaks these types.
Pure pydantic + stdlib — no torch, no boto3, no framework imports.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ── controlled vocabularies ──────────────────────────────────────────────────

FORMALITY_LEVELS = ("formal", "business", "smart-casual", "casual", "sport", "unknown")

SCENE_TYPES = (
    "office", "street", "park", "home", "beach",
    "restaurant", "studio", "indoor", "outdoor", "other",
)


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ── indexing-side records ────────────────────────────────────────────────────

class GarmentAttribute(BaseModel):
    """One garment as extracted by the VLM — the unit of compositional binding."""
    type: str
    color: str
    color_hex: Optional[str] = None
    formality: str = "unknown"
    material: Optional[str] = None          # enables negations like "no denim"

    @field_validator("type", "color", mode="before")
    @classmethod
    def _lower(cls, v: str) -> str:
        return _norm(str(v)) if v is not None else ""

    @field_validator("formality", mode="before")
    @classmethod
    def _formality(cls, v: str) -> str:
        v = _norm(str(v)) if v else "unknown"
        return v if v in FORMALITY_LEVELS else "unknown"

    @field_validator("color_hex", mode="before")
    @classmethod
    def _hex(cls, v):
        if not v:
            return None
        v = str(v).strip().lstrip("#")
        if len(v) == 6 and all(c in "0123456789abcdefABCDEF" for c in v):
            return f"#{v.lower()}"
        return None

    @field_validator("material", mode="before")
    @classmethod
    def _material(cls, v):
        return _norm(str(v)) if v else None


class ImageAttributes(BaseModel):
    """Structured payload for one image (VLM output, pydantic-validated)."""
    image_id: str = ""
    garments: list[GarmentAttribute] = Field(default_factory=list)
    scene: str = ""                         # one-line free-text scene description
    scene_type: str = "other"
    lighting: str = ""

    @field_validator("scene_type", mode="before")
    @classmethod
    def _scene_type(cls, v: str) -> str:
        v = _norm(str(v)) if v else "other"
        return v if v in SCENE_TYPES else "other"


class Detection(BaseModel):
    """One proposed garment region in an image (detector output)."""
    bbox: tuple[float, float, float, float]  # xyxy, absolute pixels
    label: str                               # detector class name (Fashionpedia category)
    score: float


class RegionRecord(BaseModel):
    """A stored garment region: joins a garment vector back to its image + bbox."""
    image_id: str
    region_id: str                           # "{image_id}#r{i}"
    bbox: tuple[float, float, float, float]
    label: str
    det_score: float


# ── query-side structures ────────────────────────────────────────────────────

class QueryGarment(BaseModel):
    """One garment term parsed from the query, e.g. {type: 'tie', color: 'red'}."""
    type: Optional[str] = None
    color: Optional[str] = None

    def text(self) -> str:
        """Natural-language rendering used for embedding the term."""
        parts = [p for p in (self.color, self.type) if p]
        return " ".join(parts)


class Negation(BaseModel):
    """Something the user excluded: 'no denim', 'not red', 'without a hat'."""
    term: str                                # raw excluded phrase
    type: Optional[str] = None               # canonical garment type, if resolvable
    color: Optional[str] = None
    material: Optional[str] = None


class ParsedQuery(BaseModel):
    """Structured intent decomposed from the natural-language query."""
    raw: str
    garments: list[QueryGarment] = Field(default_factory=list)
    scene: Optional[str] = None              # e.g. "modern office"
    style: Optional[str] = None              # e.g. "casual", "formal"
    negations: list[Negation] = Field(default_factory=list)

    def has_garment_terms(self) -> bool:
        return any(g.type or g.color for g in self.garments)


class AttributePredicate(BaseModel):
    """One conjunct for the metadata pre-filter, already synonym-expanded.

    An image satisfies the predicate iff SOME single garment record matches
    (type in types if types) AND (color in colors if colors) AND
    (material in materials if materials) — this per-record conjunction is
    what makes attribute→garment binding structural.
    """
    types: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)

    def matches_garment(self, g: GarmentAttribute) -> bool:
        if self.types and g.type not in self.types:
            return False
        if self.colors and g.color not in self.colors:
            return False
        if self.materials and (g.material or "") not in self.materials:
            return False
        return True

    def matches_image(self, attrs: ImageAttributes) -> bool:
        return any(self.matches_garment(g) for g in attrs.garments)


# ── results / explainability ─────────────────────────────────────────────────

class MatchExplanation(BaseModel):
    """Which query term matched which detected region — powers the explain panel."""
    region_id: str
    bbox: tuple[float, float, float, float]
    label: str                               # detector label of the region
    query_term: str                          # e.g. "red tie"
    similarity: float


class ComponentScores(BaseModel):
    garment: float = 0.0
    scene: float = 0.0
    attribute: float = 0.0
    rerank: Optional[float] = None


class SearchResult(BaseModel):
    image_id: str
    score: float
    components: ComponentScores = Field(default_factory=ComponentScores)
    matches: list[MatchExplanation] = Field(default_factory=list)
    attributes: Optional[ImageAttributes] = None
