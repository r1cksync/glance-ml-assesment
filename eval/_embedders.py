"""Private embedding wrappers for the eval package.

The ablation ladder needs query-side text encoders and whole-image baseline
encoders that are guaranteed to exist and behave identically across runs, so
the eval package carries its own minimal implementations of the shared
``indexer.models.base.EmbeddingModel`` ABC rather than depending on the
indexer's (heavier, batching-optimized) wrappers. Outputs are L2-normalized
float32, so cosine similarity == inner product — matching the store contract.

- HFCLIPEmbedder:  any ``transformers`` CLIPModel checkpoint
  (openai/clip-vit-base-patch32, patrickjohncyh/fashion-clip).
- OpenCLIPEmbedder: ``open_clip`` checkpoints published on the HF hub
  (Marqo/marqo-fashionSigLIP).

This is an implementation module: heavy deps (torch, transformers/open_clip)
are imported here so that eval.ablations can import it lazily.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from indexer.models.base import EmbeddingModel
from indexer.models.clip_models import _as_feature_tensor  # transformers v4/v5 shim
from indexer.models.device import resolve_device

if TYPE_CHECKING:  # PIL only needed for type checking here
    from PIL.Image import Image

log = logging.getLogger(__name__)

#: config backend name -> default HF model id (mirrors config/default.yaml)
DEFAULT_MODEL_IDS: dict[str, str] = {
    "clip-vit-b32": "openai/clip-vit-base-patch32",
    "fashion-clip": "patrickjohncyh/fashion-clip",
    "marqo-siglip": "Marqo/marqo-fashionSigLIP",
}


class HFCLIPEmbedder(EmbeddingModel):
    """transformers CLIPModel wrapper (covers OpenAI CLIP and FashionCLIP)."""

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        dim: int | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model_id = model_id
        self.name = name or model_id
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self._model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self.dim = int(dim or self._model.config.projection_dim)
        log.info("loaded CLIP embedder %s (dim=%d, device=%s)",
                 model_id, self.dim, self.device)

    @torch.no_grad()
    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = [im.convert("RGB") for im in images[i:i + self.batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            feats = _as_feature_tensor(self._model.get_image_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._normalize(np.vstack(chunks))

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            inputs = self._processor(
                text=texts[i:i + self.batch_size],
                return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)
            feats = _as_feature_tensor(self._model.get_text_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._normalize(np.vstack(chunks))


class OpenCLIPEmbedder(EmbeddingModel):
    """open_clip wrapper for HF-hub checkpoints (e.g. Marqo/marqo-fashionSigLIP)."""

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        dim: int | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        import open_clip

        self.model_id = model_id
        self.name = name or model_id
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        # HF-hub repos need the "hf-hub:" prefix for open_clip.
        ref = model_id if (model_id.startswith("hf-hub:") or "/" not in model_id) \
            else f"hf-hub:{model_id}"
        model, _, preprocess = open_clip.create_model_and_transforms(ref)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(ref)
        if dim:
            self.dim = int(dim)
        else:
            with torch.no_grad():
                probe = self._model.encode_text(self._tokenizer(["probe"]).to(self.device))
            self.dim = int(probe.shape[-1])
        log.info("loaded open_clip embedder %s (dim=%d, device=%s)",
                 model_id, self.dim, self.device)

    @torch.no_grad()
    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self._preprocess(im.convert("RGB")) for im in images[i:i + self.batch_size]]
            ).to(self.device)
            feats = self._model.encode_image(batch)
            chunks.append(feats.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._normalize(np.vstack(chunks))

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer(texts[i:i + self.batch_size]).to(self.device)
            feats = self._model.encode_text(tokens)
            chunks.append(feats.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._normalize(np.vstack(chunks))


def build_eval_embedder(
    backend: str,
    *,
    model_id: str | None = None,
    dim: int | None = None,
    device: str = "auto",
    batch_size: int = 16,
) -> EmbeddingModel:
    """Build the right wrapper for a config backend name.

    SigLIP checkpoints go through open_clip; everything else is assumed to be
    a transformers CLIPModel checkpoint.
    """
    mid = model_id or DEFAULT_MODEL_IDS.get(backend)
    if not mid:
        raise ValueError(
            f"No model id for embedding backend {backend!r} — add it under "
            "embedding.models in the config or pass model_id explicitly."
        )
    if backend == "marqo-siglip" or "siglip" in mid.lower():
        return OpenCLIPEmbedder(mid, name=backend, dim=dim,
                                device=device, batch_size=batch_size)
    return HFCLIPEmbedder(mid, name=backend, dim=dim,
                          device=device, batch_size=batch_size)
