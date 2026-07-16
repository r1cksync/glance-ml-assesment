"""Reranker interface: cross-modal relevance scoring over a short candidate list."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, images: Sequence["Image"]) -> np.ndarray:
        """Return float32 [N] relevance scores in [0, 1] (ITM match probability)."""
