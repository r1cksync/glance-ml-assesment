"""Qwen2-VL structured-attribute extractor (local HF checkpoint).

Wraps Qwen/Qwen2-VL-2B-Instruct behind the AttributeExtractor contract:
this module only produces raw text via `_generate`; JSON repair-retry and
lexicon canonicalization live in the base class.

Loads 4-bit (NF4, fp16 compute) on CUDA when requested to keep the 2B model
inside small-GPU VRAM budgets; otherwise falls back to fp32 on CPU.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from indexer.extraction.base import AttributeExtractor
from indexer.models.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

#: images are downscaled so their longest side is at most this many pixels,
#: bounding both vision-token count and VRAM per generate() call.
_MAX_IMAGE_SIDE = 512

#: hard caps on Qwen2-VL vision tokens (28x28-px patches, merged 2x2).
#: 512*512 max_pixels ≈ 334 visual tokens — 4-5x faster prefill than the
#: processor default on small GPUs, with negligible attribute-quality loss
#: at fashion-photo scales.
_MIN_PIXELS = 128 * 28 * 28
_MAX_PIXELS = 512 * 512


class Qwen2VLExtractor(AttributeExtractor):
    """Attribute extraction with Qwen2-VL (chat-template + vision-info pipeline)."""

    name = "qwen2vl"

    def __init__(
        self,
        model_id: str,
        quant: str = "4bit",
        max_new_tokens: int = 512,
        device: str = "auto",
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        resolved = resolve_device(device)
        use_4bit = (
            quant == "4bit"
            and resolved.startswith("cuda")
            and torch.cuda.is_available()
        )
        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map={"": 0},
            )
            self.device = "cuda"
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).to("cpu")
            self.device = "cpu"
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
        # batched generate() needs left padding (decoder-only model)
        self.processor.tokenizer.padding_side = "left"
        log.info("loaded %s on %s (quant=%s, max_new_tokens=%d)",
                 model_id, self.device, "4bit" if use_4bit else "fp32", max_new_tokens)

    # ── AttributeExtractor ────────────────────────────────────────────────────

    def _generate(self, image: "Image", prompt: str) -> str:
        return self._generate_many([image], prompt)[0]

    def _generate_many(self, images: "list[Image]", prompt: str) -> list[str]:
        """One batched generate() over N images — the throughput win on small
        GPUs where 4-bit decode dominates latency."""
        msgs_all = [
            [{
                "role": "user",
                "content": [
                    # pixel caps in the message dict: process_vision_info does its
                    # own smart_resize and ignores the processor-level caps
                    {"type": "image", "image": self._downscale(im),
                     "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS},
                    {"type": "text", "text": prompt},
                ],
            }]
            for im in images
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True)
            for m in msgs_all
        ]
        image_inputs = []
        for m in msgs_all:
            imgs, _videos = process_vision_info(m)
            image_inputs.extend(imgs)
        inputs = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # standard Qwen2-VL pattern: strip the echoed input tokens before decoding
        trimmed = [
            out[in_ids.shape[0]:]
            for in_ids, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # ── batched public API (used by the attribute-refresh pass) ──────────────

    def extract_batch(self, images: "list[Image]", image_ids: list[str]
                      ) -> "list":
        """Batched extraction: one generate() for the whole chunk, then per-image
        parse; images whose JSON fails fall back to the per-image repair loop.
        Returns a list of ImageAttributes or Exception per input."""
        from indexer.extraction.base import EXTRACTION_PROMPT

        raws = self._generate_many(list(images), EXTRACTION_PROMPT)
        results: list = []
        for raw, img, image_id in zip(raws, images, image_ids):
            try:
                attrs = self._parse(raw)
                attrs.image_id = image_id
                results.append(attrs)
            except Exception:  # noqa: BLE001 — repair individually
                try:
                    results.append(self.extract(img, image_id))
                except Exception as e:  # noqa: BLE001
                    results.append(e)
        return results

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _downscale(image: "Image", max_side: int = _MAX_IMAGE_SIDE) -> "Image":
        """RGB copy with longest side <= max_side (thumbnail preserves aspect)."""
        img = image.convert("RGB") if image.mode != "RGB" else image.copy()
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        return img
