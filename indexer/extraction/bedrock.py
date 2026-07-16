"""AWS Bedrock structured-attribute extractor (Converse API).

The only indexer module allowed to touch boto3, and only lazily inside this
class — ML logic elsewhere stays cloud-free. Sends a downscaled PNG plus the
extraction prompt to a Bedrock-hosted VLM (e.g. Claude Haiku) and returns the
raw text; JSON repair-retry and canonicalization live in the base class.
"""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

from indexer.extraction.base import AttributeExtractor

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

#: longest image side sent to Bedrock — bounds payload size and vision tokens.
_MAX_IMAGE_SIDE = 1024


class BedrockExtractor(AttributeExtractor):
    """Attribute extraction via the Bedrock `converse` API."""

    name = "bedrock"

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self.model_id = model_id
        self.region = region
        try:
            import boto3  # lazy: boto3 is an optional dependency of this backend only
            self._client: Any = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:  # noqa: BLE001 — surface a setup-oriented message
            raise RuntimeError(
                "BedrockExtractor could not create a bedrock-runtime client for "
                f"region {region!r}. Ensure boto3 is installed and AWS credentials "
                "are configured (env vars, shared credentials file, or IAM role). "
                f"Underlying error: {e}"
            ) from e
        log.info("bedrock extractor ready (model=%s, region=%s)", model_id, region)

    # ── AttributeExtractor ────────────────────────────────────────────────────

    def _generate(self, image: "Image", prompt: str) -> str:
        png = self._png_bytes(image)
        response = self._client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": png}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 700, "temperature": 0.0},
        )
        parts = response["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _png_bytes(image: "Image", max_side: int = _MAX_IMAGE_SIDE) -> bytes:
        """PNG-encode an RGB copy downscaled so the longest side <= max_side."""
        img = image.convert("RGB") if image.mode != "RGB" else image.copy()
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
