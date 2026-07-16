"""CLIP-family dual encoders via HuggingFace transformers.

Wraps any standard ``CLIPModel`` checkpoint — both ``openai/clip-vit-base-patch32``
and ``patrickjohncyh/fashion-clip`` (FashionCLIP is a fine-tuned CLIP with the
same architecture and processor). All outputs are L2-normalized float32 numpy
arrays per the EmbeddingModel contract.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from indexer.models.base import EmbeddingModel
from indexer.models.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


def _as_feature_tensor(out: "torch.Tensor | object") -> torch.Tensor:
    """Normalize get_image_features/get_text_features return across transformers
    versions: v4 returns the projected tensor directly, v5 returns a
    BaseModelOutputWithPooling whose ``pooler_output`` holds the projected features."""
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        t = getattr(out, attr, None)
        if torch.is_tensor(t):
            return t
    raise TypeError(f"cannot extract features from {type(out).__name__}")


class CLIPEmbedder(EmbeddingModel):
    """Batched image/text embedder over a HF ``CLIPModel`` checkpoint.

    On CUDA the model runs in fp16 (``.half()``); outputs are always cast back
    to float32 before normalization so downstream stores see a uniform dtype.
    """

    def __init__(self, name: str, model_id: str, dim: int,
                 device: str = "auto", batch_size: int = 16) -> None:
        self.name = name
        self.model_id = model_id
        self.dim = dim
        self.batch_size = max(1, int(batch_size))
        self.device = resolve_device(device)

        log.info("loading CLIP checkpoint %s (dim=%d) on %s",
                 model_id, dim, self.device)
        model = CLIPModel.from_pretrained(model_id)
        if self.device.startswith("cuda"):
            model = model.half()
        self.model = model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    # ── EmbeddingModel API ────────────────────────────────────────────────────

    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = list(images[start:start + self.batch_size])
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    self.device, dtype=self.model.dtype)
                feats = _as_feature_tensor(
                    self.model.get_image_features(pixel_values=pixel_values))
                chunks.append(feats.float().cpu().numpy())
        return self._normalize(np.concatenate(chunks, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                inputs = self.processor(
                    text=batch, padding=True, truncation=True,
                    max_length=77, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = _as_feature_tensor(self.model.get_text_features(**inputs))
                chunks.append(feats.float().cpu().numpy())
        return self._normalize(np.concatenate(chunks, axis=0))
