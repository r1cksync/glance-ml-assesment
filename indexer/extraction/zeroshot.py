"""CLIP zero-shot attribute inference — the middle tier between the detector
fallback and a full VLM pass.

Uses signals the index already contains (garment region vectors, the scene
vector, region bboxes) plus cheap pixel statistics, so upgrading a whole corpus
takes seconds-per-hundred-images on CPU:

- color      : k-means over the crop's central pixels, saturation-weighted
               cluster pick, mapped to the canonical palette by RGB distance
- scene_type : zero-shot — scene vector vs one text prompt per scene type
- formality  : zero-shot — garment vector vs formality prompts
- material   : zero-shot — garment vector vs material prompts (only kept when
               confidently separated; materials are subtle at CLIP resolution)

No boto3, no VLM. Quality sits between the detector fallback and Qwen2-VL;
the ledger-driven VLM refresh overwrites these progressively.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.lexicon import CANONICAL_COLORS, SCENE_LEXICON, nearest_color
from core.schemas import GarmentAttribute, ImageAttributes, RegionRecord
from indexer.models.base import EmbeddingModel

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

_FORMALITY_PROMPTS: dict[str, str] = {
    "formal": "a photo of formal elegant evening wear",
    "business": "a photo of professional business attire",
    "smart-casual": "a photo of smart casual clothing",
    "casual": "a photo of casual everyday clothing",
    "sport": "a photo of sportswear and athletic clothing",
}

_MATERIAL_PROMPTS: dict[str, str] = {
    "denim": "a photo of a denim garment",
    "leather": "a photo of a leather garment",
    "wool": "a photo of a knitted wool garment",
    "silk": "a photo of a silk garment",
    "cotton": "a photo of a cotton garment",
}

_SCENE_PROMPTS: dict[str, str] = {
    name: f"a photo taken in {aliases[0] if name != 'street' else 'a city street'}"
    for name, aliases in SCENE_LEXICON.items()
}


class ZeroShotAttributeInferencer:
    """Batch attribute upgrade using stored vectors + crop pixel statistics."""

    def __init__(self, garment_embedder: EmbeddingModel,
                 scene_embedder: EmbeddingModel,
                 material_floor: float = 0.20, material_margin: float = 0.015,
                 scene_floor: float = 0.18):
        self.material_floor = material_floor
        self.material_margin = material_margin
        self.scene_floor = scene_floor
        # one-time prompt embedding
        self._formality_keys = list(_FORMALITY_PROMPTS)
        self._formality_mat = garment_embedder.embed_texts(
            [_FORMALITY_PROMPTS[k] for k in self._formality_keys])
        self._material_keys = list(_MATERIAL_PROMPTS)
        self._material_mat = garment_embedder.embed_texts(
            [_MATERIAL_PROMPTS[k] for k in self._material_keys])
        self._scene_keys = list(_SCENE_PROMPTS)
        self._scene_mat = scene_embedder.embed_texts(
            [_SCENE_PROMPTS[k] for k in self._scene_keys])

    # ── per-image inference ──────────────────────────────────────────────────

    def infer(self, image: "Image", regions: list[RegionRecord],
              garment_vecs: dict[str, np.ndarray],
              scene_vec: Optional[np.ndarray]) -> ImageAttributes:
        garments: list[GarmentAttribute] = []
        for region in regions:
            vec = garment_vecs.get(region.region_id)
            garments.append(GarmentAttribute(
                type=region.label,
                color=self._dominant_color(image, region.bbox) or "",
                formality=self._zero_shot(vec, self._formality_mat,
                                          self._formality_keys) or "unknown",
                material=self._material(vec),
            ))
        scene_type = "other"
        if scene_vec is not None:
            hit = self._zero_shot(scene_vec, self._scene_mat, self._scene_keys,
                                  floor=self.scene_floor)
            scene_type = hit or "other"
        return ImageAttributes(
            garments=garments,
            scene=f"a {scene_type} scene" if scene_type != "other" else "",
            scene_type=scene_type,
            lighting="",
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_shot(vec: Optional[np.ndarray], prompt_mat: np.ndarray,
                   keys: list[str], floor: float = 0.0) -> Optional[str]:
        if vec is None:
            return None
        sims = prompt_mat @ np.asarray(vec, dtype=np.float32)
        idx = int(np.argmax(sims))
        return keys[idx] if float(sims[idx]) >= floor else None

    def _material(self, vec: Optional[np.ndarray]) -> Optional[str]:
        if vec is None:
            return None
        sims = self._material_mat @ np.asarray(vec, dtype=np.float32)
        order = np.argsort(sims)[::-1]
        top, second = float(sims[order[0]]), float(sims[order[1]])
        if top >= self.material_floor and (top - second) >= self.material_margin:
            return self._material_keys[int(order[0])]
        return None

    @staticmethod
    def _dominant_color(image: "Image", bbox: tuple[float, float, float, float],
                        k: int = 3) -> Optional[str]:
        """Saturation-weighted k-means dominant color of the crop's center."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        # central 70% of the crop, shrunk — cheap and background-resistant
        mx, my = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        crop = image.crop((x1 + mx, y1 + my, x2 - mx, y2 - my)).resize((32, 32))
        px = np.asarray(crop.convert("RGB"), dtype=np.float32).reshape(-1, 3)

        # tiny k-means (numpy, fixed seed, 8 iters is plenty at 1K points)
        rng = np.random.default_rng(0)
        centers = px[rng.choice(len(px), size=k, replace=False)]
        for _ in range(8):
            d = ((px[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            for j in range(k):
                sel = px[assign == j]
                if len(sel):
                    centers[j] = sel.mean(0)
        counts = np.bincount(assign, minlength=k).astype(np.float32)

        # weight cluster size by chroma so vivid garment color beats
        # large desaturated background/skin regions
        maxc, minc = centers.max(1), centers.min(1)
        sat = (maxc - minc) / np.clip(maxc, 1e-6, None)
        weights = counts * (0.25 + sat)
        c = centers[int(weights.argmax())]
        hexcode = f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
        return nearest_color(hexcode)


__all__ = ["ZeroShotAttributeInferencer", "CANONICAL_COLORS"]
