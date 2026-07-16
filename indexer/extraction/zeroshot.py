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


def _contains(outer: tuple, inner: tuple, min_overlap: float = 0.7) -> bool:
    """True when >= min_overlap of `inner`'s area lies inside `outer` and the
    inner region is meaningfully smaller (nested accent like a tie in a shirt)."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    inter = (max(0.0, min(ox2, ix2) - max(ox1, ix1))
             * max(0.0, min(oy2, iy2) - max(oy1, iy1)))
    inner_area = max(1e-6, (ix2 - ix1) * (iy2 - iy1))
    outer_area = (ox2 - ox1) * (oy2 - oy1)
    return inter / inner_area >= min_overlap and inner_area < 0.8 * outer_area


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
            # mask sibling regions nested inside this bbox (e.g. the tie inside
            # a shirt crop) so their color cannot hijack the k-means pick
            siblings = [r.bbox for r in regions
                        if r.region_id != region.region_id
                        and _contains(region.bbox, r.bbox)]
            garments.append(GarmentAttribute(
                type=region.label,
                color=self._dominant_color(image, region.bbox, siblings) or "",
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
                        mask_boxes: Optional[list] = None,
                        k: int = 3) -> Optional[str]:
        """Chroma-weighted k-means dominant color of the crop's center,
        excluding pixels that belong to nested sibling regions."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        # central 70% of the crop, shrunk — cheap and background-resistant
        mx, my = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        cx1, cy1, cx2, cy2 = x1 + mx, y1 + my, x2 - mx, y2 - my
        crop = image.crop((cx1, cy1, cx2, cy2)).resize((32, 32))
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        keep = np.ones((32, 32), dtype=bool)
        for b in (mask_boxes or []):
            sx = 32.0 / max(cx2 - cx1, 1)
            sy = 32.0 / max(cy2 - cy1, 1)
            bx1 = int(np.clip((b[0] - cx1) * sx, 0, 32))
            by1 = int(np.clip((b[1] - cy1) * sy, 0, 32))
            bx2 = int(np.clip((b[2] - cx1) * sx, 0, 32))
            by2 = int(np.clip((b[3] - cy1) * sy, 0, 32))
            keep[by1:by2, bx1:bx2] = False
        px = arr[keep].reshape(-1, 3)
        if len(px) < 32:  # nearly everything masked — fall back to full crop
            px = arr.reshape(-1, 3)

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

        # weight cluster size by chroma so vivid garment color beats large
        # desaturated background — but not so hard that a genuinely white/black
        # garment loses to a small colorful accent
        maxc, minc = centers.max(1), centers.min(1)
        sat = (maxc - minc) / np.clip(maxc, 1e-6, None)
        weights = counts * (0.45 + 0.6 * sat)
        c = centers[int(weights.argmax())]
        hexcode = f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
        return nearest_color(hexcode)


__all__ = ["ZeroShotAttributeInferencer", "CANONICAL_COLORS"]
