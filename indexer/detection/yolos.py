"""Garment region detection with YOLOS fine-tuned on Fashionpedia.

Wraps ``valentinafeve/yolos-fashionpedia`` (YOLOS-small) via transformers'
AutoImageProcessor + AutoModelForObjectDetection. Labels are emitted exactly
as the checkpoint's ``id2label`` gives them — Fashionpedia category names such
as "shirt, blouse", "pants", plus garment-part labels like "sleeve", "collar",
"pocket". No label filtering happens here: canonicalization to main garment
types is done downstream via ``core.lexicon.FASHIONPEDIA_LABEL_MAP``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from core.schemas import Detection
from indexer.detection.base import Detector
from indexer.models.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

#: raw post-process floor — deliberately low; the real score gate is the
#: constructor's confidence_threshold, applied in Detector.filter_detections.
_RAW_SCORE_FLOOR = 0.05


class YolosFashionpediaDetector(Detector):
    """Proposes garment regions (bbox xyxy in absolute pixels), best-first."""

    name = "yolos-fashionpedia"

    def __init__(self, model_id: str, confidence_threshold: float,
                 max_regions: int, min_area_frac: float,
                 device: str = "auto") -> None:
        self.model_id = model_id
        self.confidence_threshold = float(confidence_threshold)
        self.max_regions = int(max_regions)
        self.min_area_frac = float(min_area_frac)
        self.device = resolve_device(device)

        log.info("loading detector %s on %s", model_id, self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label: dict[int, str] = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }

    # ── Detector API ─────────────────────────────────────────────────────────

    def detect(self, image: "Image") -> list[Detection]:
        img = image if image.mode == "RGB" else image.convert("RGB")
        w, h = img.size

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # target_sizes is (height, width) per image
        target_sizes = torch.tensor([(h, w)], dtype=torch.long)
        raw = self.processor.post_process_object_detection(
            outputs,
            threshold=min(_RAW_SCORE_FLOOR, self.confidence_threshold),
            target_sizes=target_sizes,
        )[0]

        dets: list[Detection] = []
        for score, label_id, box in zip(raw["scores"], raw["labels"], raw["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            x1 = min(max(x1, 0.0), float(w))
            y1 = min(max(y1, 0.0), float(h))
            x2 = min(max(x2, 0.0), float(w))
            y2 = min(max(y2, 0.0), float(h))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self.id2label.get(int(label_id), str(int(label_id)))
            dets.append(Detection(bbox=(x1, y1, x2, y2), label=label,
                                  score=float(score)))

        return self.filter_detections(
            dets,
            image_size=(w, h),
            min_score=self.confidence_threshold,
            min_area_frac=self.min_area_frac,
            max_regions=self.max_regions,
        )
