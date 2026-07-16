"""End-to-end indexing pipeline: detect -> crop -> embed -> extract -> store.

For each image: propose garment regions, keep those whose detector label
resolves through the shared lexicon, batch-embed padded crops in garment space,
embed the whole image in scene space, extract structured attributes with the
VLM (falling back to detector-derived attributes on failure), and upsert
everything into the vector store — persisting periodically so long runs are
resumable. One corrupt file never kills the run.

All collaborators are injected (see core.bootstrap.build_pipeline); this module
holds no config reading and no AWS coupling.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from core.lexicon import canonical_garment
from core.schemas import RegionRecord
from indexer.detection.base import Detector
from indexer.extraction.base import AttributeExtractor
from indexer.extraction.fallback import attributes_from_detections
from indexer.models.base import EmbeddingModel
from indexer.storage.base import VectorStore

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_PAD_FRAC = 0.03  # bbox padding on each side, relative to bbox width/height


class IndexingPipeline:
    """Orchestrates detector + embedders + extractor + store over a directory."""

    def __init__(
        self,
        detector: Detector,
        garment_embedder: EmbeddingModel,
        scene_embedder: EmbeddingModel,
        extractor: AttributeExtractor,
        store: VectorStore,
        artifacts_dir: Path,
        persist_every: int = 25,
    ):
        self.detector = detector
        self.garment_embedder = garment_embedder
        self.scene_embedder = scene_embedder
        self.extractor = extractor
        self.store = store
        self.artifacts_dir = Path(artifacts_dir)
        self.index_dir = self.artifacts_dir / "index"
        self.persist_every = int(persist_every)

    # ── public API ───────────────────────────────────────────────────────────

    def index_directory(
        self,
        images_dir: Path,
        resume: bool = True,
        limit: int | None = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Index every *.jpg/*.jpeg/*.png under ``images_dir`` (sorted by name).

        ``on_progress`` (optional) is called after each file as
        ``on_progress(position, total, image_id)``.

        Returns a summary dict:
        {indexed, skipped, failed, failed_extraction, fallback_used,
         regions_total, seconds}.
        """
        images_dir = Path(images_dir)
        files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if limit is not None:
            files = files[:limit]
        existing = set(self.store.image_ids()) if resume else set()

        t0 = time.time()
        indexed = skipped = failed = 0
        failed_extraction = fallback_used = regions_total = 0
        total = len(files)
        log.info("indexing %d images from %s (resume=%s, persist_every=%d)",
                 total, images_dir, resume, self.persist_every)

        for pos, path in enumerate(files, start=1):
            image_id = path.name
            if image_id in existing:
                skipped += 1
            else:
                try:
                    n_regions, used_fallback = self._index_one(path, image_id)
                    indexed += 1
                    regions_total += n_regions
                    if used_fallback:
                        failed_extraction += 1
                        fallback_used += 1
                    if self.persist_every > 0 and indexed % self.persist_every == 0:
                        self._persist()
                except Exception:  # noqa: BLE001 — one corrupt file never kills the run
                    failed += 1
                    log.exception("failed to index %s — continuing", image_id)
            if pos % 10 == 0 or pos == total:
                elapsed = max(time.time() - t0, 1e-9)
                log.info("progress: %d/%d (%.2f img/s; indexed=%d skipped=%d failed=%d)",
                         pos, total, pos / elapsed, indexed, skipped, failed)
            if on_progress is not None:
                try:
                    on_progress(pos, total, image_id)
                except Exception:  # noqa: BLE001 — observer errors must not stop indexing
                    log.warning("on_progress callback failed", exc_info=True)

        if indexed:
            self._persist()
        summary = {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "failed_extraction": failed_extraction,
            "fallback_used": fallback_used,
            "regions_total": regions_total,
            "seconds": round(time.time() - t0, 2),
        }
        log.info("indexing done: %s", summary)
        return summary

    # ── per-image work ───────────────────────────────────────────────────────

    def _index_one(self, path: Path, image_id: str) -> tuple[int, bool]:
        """Index one image; returns (regions stored, whether fallback attrs were used)."""
        with Image.open(path) as im:
            img = im.convert("RGB")

        detections = self.detector.detect(img)
        regions: list[RegionRecord] = []
        crops: list[Image.Image] = []
        for i, det in enumerate(detections):
            canonical = canonical_garment(det.label)
            if canonical is None:
                continue  # garment part / untracked category — not indexed
            regions.append(RegionRecord(
                image_id=image_id,
                region_id=f"{image_id}#r{i}",
                bbox=det.bbox,
                label=canonical,
                det_score=det.score,
            ))
            crops.append(self._crop_with_padding(img, det.bbox))

        garment_vecs: list[tuple[RegionRecord, np.ndarray]] = []
        if crops:
            vecs = self.garment_embedder.embed_images(crops)
            garment_vecs = list(zip(regions, vecs))
        # zero garment regions is fine — the scene vector still gets indexed
        scene_vec = self.scene_embedder.embed_image(img)

        used_fallback = False
        try:
            attrs = self.extractor.extract(img, image_id)
        except Exception as e:  # noqa: BLE001 — ExtractionError or anything else
            used_fallback = True
            log.warning("extraction failed for %s (%s: %s); using detector fallback",
                        image_id, type(e).__name__, e)
            attrs = attributes_from_detections(img, detections)
        attrs.image_id = image_id

        self.store.add_image(image_id, scene_vec, garment_vecs, attrs)
        return len(regions), used_fallback

    @staticmethod
    def _crop_with_padding(img: Image.Image,
                           bbox: tuple[float, float, float, float]) -> Image.Image:
        """Crop bbox with 3% padding per side, clamped to the image bounds."""
        x1, y1, x2, y2 = bbox
        pw = _PAD_FRAC * (x2 - x1)
        ph = _PAD_FRAC * (y2 - y1)
        left = int(max(0.0, x1 - pw))
        top = int(max(0.0, y1 - ph))
        right = int(min(float(img.width), round(x2 + pw)))
        bottom = int(min(float(img.height), round(y2 + ph)))
        if right <= left:  # degenerate box — keep at least one pixel
            right = min(img.width, left + 1)
        if bottom <= top:
            bottom = min(img.height, top + 1)
        return img.crop((left, top, right, bottom))

    # ── persistence ──────────────────────────────────────────────────────────

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.persist(self.index_dir)
        log.info("persisted index to %s", self.index_dir)
