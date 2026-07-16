"""Owns the SearchEngine lifecycle: built ONCE in a background thread at
startup; requests get 503 until it is ready; /admin/reindex rebuilds it while
the previous engine keeps serving (atomic swap on success).

core.bootstrap (and everything heavy behind it — torch, transformers, faiss)
is imported LAZILY inside the build, so importing this module stays cheap for
unit tests.
"""
from __future__ import annotations

import inspect
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from api.settings import Settings

if TYPE_CHECKING:
    from api.repos import ParseCache
    from retriever.search.engine import SearchEngine

log = logging.getLogger(__name__)


class EngineHolder:
    """Thread-safe holder around the (slow-to-build) SearchEngine."""

    def __init__(self, settings: Settings,
                 parse_cache: Optional["ParseCache"] = None) -> None:
        self._settings = settings
        self._parse_cache = parse_cache
        self._engine: Optional["SearchEngine"] = None
        self._error: Optional[str] = None
        self._snapshot: dict[str, Any] = {}
        self._build_lock = threading.Lock()
        self._filenames: Optional[dict[str, str]] = None  # image_id -> filename

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Kick off (re)building in a daemon thread; returns immediately."""
        threading.Thread(target=self.build_now, name="engine-build",
                         daemon=True).start()

    def build_now(self) -> None:
        """Build synchronously in the calling thread. On success the new
        engine is swapped in atomically; on failure the previous engine (if
        any) keeps serving and `error` is set."""
        with self._build_lock:
            try:
                engine, snapshot = self._build()
            except Exception as exc:  # noqa: BLE001 — surfaced via /readyz + logs
                log.exception("engine build failed")
                self._error = f"{type(exc).__name__}: {exc}"
                return
            self._engine = engine
            self._snapshot = snapshot
            self._error = None
            self._filenames = None  # image set may have changed — refresh lazily
            log.info("search engine ready (%d images indexed)",
                     len(engine.store.image_ids()))

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def engine(self) -> Optional["SearchEngine"]:
        return self._engine

    def config_snapshot(self) -> dict[str, Any]:
        """Engine config captured at build time (weights, backends, …)."""
        return dict(self._snapshot)

    # ── build internals ───────────────────────────────────────────────────────

    def _build(self) -> tuple["SearchEngine", dict[str, Any]]:
        from core.config import load_config  # lazy — see module docstring

        overrides = {"paths": {
            "artifacts_dir": self._settings.ARTIFACTS_DIR,
            "images_dir": self._settings.IMAGES_DIR,
        }}
        cfg = load_config(overrides=overrides)

        import core.bootstrap as bootstrap  # lazy — pulls torch/faiss/transformers

        build_engine = bootstrap.build_engine
        params = inspect.signature(build_engine).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())

        kwargs: dict[str, Any] = {}
        if has_var_kw or "with_reranker" in params:
            kwargs["with_reranker"] = self._settings.USE_RERANK
        if "store" in params and hasattr(bootstrap, "load_store"):
            kwargs["store"] = bootstrap.load_store(cfg)
        if self._parse_cache is not None:
            for name in ("parse_cache", "cache"):
                if name in params:
                    kwargs[name] = self._parse_cache
                    break

        engine: "SearchEngine" = build_engine(cfg, **kwargs)

        snapshot: dict[str, Any] = {
            "app": cfg.get_path("app.name"),
            "weights": cfg.get_path("search.weights"),
            "candidates_k": cfg.get_path("search.candidates_k"),
            "prefilter": cfg.get_path("search.prefilter"),
            "backends": {
                "storage": cfg.get_path("storage.backend"),
                "garment_embedding": cfg.get_path("embedding.garment_backend"),
                "scene_embedding": cfg.get_path("embedding.scene_backend"),
                "parsing": cfg.get_path("parsing.backend"),
                "rerank": cfg.get_path("rerank.backend"),
            },
            "rerank_enabled": bool(self._settings.USE_RERANK
                                   and cfg.get_path("rerank.enabled", True)),
            "artifacts_dir": str(self._settings.artifacts_path),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        return engine, snapshot

    # ── image URLs ────────────────────────────────────────────────────────────

    def image_url(self, image_id: str) -> str:
        """Public URL for an image.

        Priority: explicit CDN base -> relative "/images/<id>" when S3-backed
        (the same CloudFront distribution that fronts this API serves /images/*
        from the bucket, so relative URLs resolve in the browser and no
        CloudFront<->EC2 config cycle exists) -> local static mount for dev.
        """
        base = self._settings.IMAGE_BASE_URL
        if base:
            return f"{base.rstrip('/')}/images/{image_id}"
        if self._settings.S3_BUCKET:
            return f"/images/{image_id}"
        return f"/local-images/{self._local_filename(image_id)}"

    def _local_filename(self, image_id: str) -> str:
        """Map an image_id to its on-disk filename (adds the extension when
        the id was stored without one). One lazy directory scan, cached."""
        if Path(image_id).suffix:
            return image_id
        mapping = self._filenames
        if mapping is None:
            mapping = {}
            images_dir = self._settings.images_path
            if images_dir.is_dir():
                for p in images_dir.iterdir():
                    if p.is_file():
                        mapping.setdefault(p.stem, p.name)
            self._filenames = mapping
        return mapping.get(image_id, image_id)
