"""Garment region detector interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from core.schemas import Detection

if TYPE_CHECKING:
    from PIL.Image import Image


class Detector(ABC):
    """Proposes garment regions for an image. Labels are canonicalized via
    core.lexicon.FASHIONPEDIA_LABEL_MAP by callers that need canonical types."""

    name: str

    @abstractmethod
    def detect(self, image: "Image") -> list[Detection]:
        """Return garment detections (bbox xyxy in absolute pixels), best-first."""

    def detect_batch(self, images: Sequence["Image"]) -> list[list[Detection]]:
        return [self.detect(im) for im in images]

    @staticmethod
    def filter_detections(dets: list[Detection], *, image_size: tuple[int, int],
                          min_score: float, min_area_frac: float,
                          max_regions: int) -> list[Detection]:
        """Shared post-processing: score floor, tiny-region floor, top-N by score."""
        w, h = image_size
        area = float(w * h) or 1.0
        keep = [
            d for d in dets
            if d.score >= min_score
            and ((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) / area >= min_area_frac
        ]
        keep.sort(key=lambda d: d.score, reverse=True)
        return keep[:max_regions]
