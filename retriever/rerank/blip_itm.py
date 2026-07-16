"""BLIP image-text-matching (ITM) cross-encoder reranker.

Re-scores the top fusion candidates by running each (query, image) pair through
BLIP's ITM head — full cross-attention between text and image tokens, far more
precise than dual-encoder cosine similarity at the cost of one forward pass per
pair. The output is the softmax probability of the "match" class in [0, 1],
blended into the fused score by the search engine.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from transformers import BlipForImageTextRetrieval, BlipProcessor

from indexer.models.device import resolve_device
from retriever.rerank.base import Reranker

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


class BlipITMReranker(Reranker):
    """Cross-modal reranker over the BLIP ITM (image-text matching) head.

    Images are resized by the processor's checkpoint defaults (384px for the
    COCO ITM checkpoint); no manual resizing is applied. Uses fp16 on CUDA.
    """

    name = "blip-itm"

    def __init__(self, model_id: str = "Salesforce/blip-itm-base-coco",
                 device: str = "auto", batch_size: int = 8):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_id)
        self._dtype = torch.float32
        if self.device == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        self.model.to(self.device)
        self.model.eval()
        log.info("BLIP ITM reranker loaded: %s on %s (%s)",
                 model_id, self.device, self._dtype)

    @torch.no_grad()
    def score(self, query: str, images: Sequence["Image"]) -> np.ndarray:
        """Return float32 [N] ITM match probabilities of `query` against each image."""
        imgs = list(images)
        if not imgs:
            return np.zeros((0,), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(imgs), self.batch_size):
            batch = [im if getattr(im, "mode", "RGB") == "RGB" else im.convert("RGB")
                     for im in imgs[start:start + self.batch_size]]
            inputs = self.processor(images=batch, text=[query] * len(batch),
                                    return_tensors="pt", padding=True, truncation=True)
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self._dtype)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            outputs = self.model(pixel_values=pixel_values, input_ids=input_ids,
                                 attention_mask=attention_mask, use_itm_head=True)
            # transformers versions differ: ModelOutput with .itm_score vs plain tuple
            logits = getattr(outputs, "itm_score", None)
            if logits is None:
                logits = outputs[0]
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            chunks.append(probs.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)
