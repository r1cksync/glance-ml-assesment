"""Detector backend factory.

Composition roots call ``build_detector(cfg)`` to obtain the configured region
detector. Implementation modules (torch/transformers) are imported lazily
inside the build function so importing this factory stays cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config
    from indexer.detection.base import Detector

log = logging.getLogger(__name__)


def build_detector(cfg: "Config") -> "Detector":
    """Build the detector configured under ``detection.*``."""
    backend = cfg.get_path("detection.backend")
    if not backend:
        raise ValueError("config missing detection.backend")

    if backend == "yolos-fashionpedia":
        from indexer.detection.yolos import YolosFashionpediaDetector
        log.info("building detector backend=%s model_id=%s",
                 backend, cfg.get_path("detection.model_id"))
        return YolosFashionpediaDetector(
            model_id=str(cfg.get_path("detection.model_id")),
            confidence_threshold=float(
                cfg.get_path("detection.confidence_threshold", 0.35)),
            max_regions=int(cfg.get_path("detection.max_regions", 8)),
            min_area_frac=float(cfg.get_path("detection.min_area_frac", 0.005)),
            device=str(cfg.get_path("detection.device", "auto")),
        )
    raise ValueError(f"unknown detection backend {backend!r}")
