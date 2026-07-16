"""Detector-derived attribute fallback.

Degraded path used when the VLM fails an image entirely (ExtractionError after
all repair retries): synthesize a minimal ImageAttributes payload from the
detector's regions alone — canonical garment type from the Fashionpedia label
plus a crude dominant color sampled from the bbox crop. Formality and scene
information are unknowable here and left at their neutral defaults.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.lexicon import canonical_garment, nearest_color
from core.schemas import Detection, GarmentAttribute, ImageAttributes

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


def attributes_from_detections(image: "Image",
                               detections: list[Detection]) -> ImageAttributes:
    """Build ImageAttributes from detector output only (no VLM).

    Keeps a garment per detection whose label resolves through the shared
    lexicon (FASHIONPEDIA_LABEL_MAP-aware canonical_garment); color is the
    nearest canonical color to the median RGB of the crop's central region.
    """
    garments: list[GarmentAttribute] = []
    for det in detections:
        ctype = canonical_garment(det.label)
        if ctype is None:
            continue
        color, color_hex = _dominant_color(image, det.bbox)
        garments.append(GarmentAttribute(
            type=ctype,
            color=color or "",
            color_hex=color_hex,
            formality="unknown",
            material=None,
        ))
    log.debug("fallback attributes: %d/%d detections resolved to garments",
              len(garments), len(detections))
    return ImageAttributes(garments=garments, scene="", scene_type="other",
                           lighting="")


def _dominant_color(image: "Image",
                    bbox: tuple[float, float, float, float],
                    ) -> tuple[Optional[str], Optional[str]]:
    """(canonical color name, hex) for a bbox crop, or (None, None) if degenerate.

    Method: crop → resize to 24x24 RGB → per-channel median over the middle
    50% region (avoids background bleeding in at the box edges) → nearest
    canonical color anchor.
    """
    w, h = image.size
    x0 = max(0.0, min(float(w), bbox[0]))
    y0 = max(0.0, min(float(h), bbox[1]))
    x1 = max(0.0, min(float(w), bbox[2]))
    y1 = max(0.0, min(float(h), bbox[3]))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None, None
    crop = image.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
    small = crop.resize((24, 24))
    arr = np.asarray(small, dtype=np.float64)
    mid = arr[6:18, 6:18, :]  # central 50% of each side
    r = int(round(float(np.median(mid[:, :, 0]))))
    g = int(round(float(np.median(mid[:, :, 1]))))
    b = int(round(float(np.median(mid[:, :, 2]))))
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    return nearest_color(hex_code), hex_code
