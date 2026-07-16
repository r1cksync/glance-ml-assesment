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
_MAX_IMAGE_SIDE = 768


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
        self.processor = AutoProcessor.from_pretrained(model_id)
        log.info("loaded %s on %s (quant=%s, max_new_tokens=%d)",
                 model_id, self.device, "4bit" if use_4bit else "fp32", max_new_tokens)

    # ── AttributeExtractor ────────────────────────────────────────────────────

    def _generate(self, image: "Image", prompt: str) -> str:
        img = self._downscale(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt")
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
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0]

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _downscale(image: "Image", max_side: int = _MAX_IMAGE_SIDE) -> "Image":
        """RGB copy with longest side <= max_side (thumbnail preserves aspect)."""
        img = image.convert("RGB") if image.mode != "RGB" else image.copy()
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        return img
