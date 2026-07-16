"""Moondream2 structured-attribute extractor (tiny local VLM).

Wraps vikhyatk/moondream2 (trust_remote_code checkpoint) behind the
AttributeExtractor contract. Prefers the modern `model.query(image, prompt)`
API and falls back to the legacy `encode_image` + `answer_question` pair for
older revisions.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM

from indexer.extraction.base import AttributeExtractor
from indexer.models.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


class MoondreamExtractor(AttributeExtractor):
    """Attribute extraction with Moondream2 — fp16 on CUDA, fp32 on CPU."""

    name = "moondream"

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "auto",
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self.model_id = model_id
        self.revision = revision
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self.device},
        )
        self.model.eval()
        self._tokenizer: Any = None  # lazy, only needed by the oldest legacy API
        log.info("loaded %s@%s on %s", model_id, revision, self.device)

    # ── AttributeExtractor ────────────────────────────────────────────────────

    def _generate(self, image: "Image", prompt: str) -> str:
        img = image if image.mode == "RGB" else image.convert("RGB")
        try:
            result = self.model.query(img, prompt)
        except AttributeError:
            log.debug("moondream .query unavailable; using legacy encode/answer API")
            return self._generate_legacy(img, prompt)
        if isinstance(result, dict):
            return str(result.get("answer", ""))
        return str(result)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _generate_legacy(self, image: "Image", prompt: str) -> str:
        enc = self.model.encode_image(image)
        try:
            return str(self.model.answer_question(enc, prompt))
        except TypeError:
            # oldest revisions require an explicit tokenizer argument
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, revision=self.revision, trust_remote_code=True)
            return str(self.model.answer_question(enc, prompt, self._tokenizer))
