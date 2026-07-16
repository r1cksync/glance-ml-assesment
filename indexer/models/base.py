"""Embedding model interface. Implementations wrap HF checkpoints; all outputs
are L2-normalized float32 numpy arrays so cosine similarity == inner product."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid importing PIL at type-check only sites
    from PIL.Image import Image


class EmbeddingModel(ABC):
    """A dual-encoder that maps images and texts into one shared space."""

    #: unique config name, e.g. "fashion-clip"
    name: str
    #: embedding dimensionality
    dim: int

    @abstractmethod
    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        """(N images) -> float32 array [N, dim], L2-normalized rows."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """(N strings) -> float32 array [N, dim], L2-normalized rows."""

    def embed_image(self, image: "Image") -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)
