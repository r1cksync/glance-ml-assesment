"""Admin endpoints (role "admin" required): store stats, attribute
distributions, artifact info, and hot reindexing.

/reindex refreshes artifacts (S3 sync when S3_BUCKET is set, plain disk
reload in local mode) and rebuilds the engine in a background thread — the
previous engine keeps serving until the new one swaps in.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, status

from api.deps import (
    get_engine,
    get_engine_holder,
    get_settings,
    require_role,
)
from api.engine_holder import EngineHolder
from api.settings import Settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"],
                   dependencies=[Depends(require_role("admin"))])

_TOP_N = 20


@router.get("/stats")
async def stats(
    holder: EngineHolder = Depends(get_engine_holder),
    engine=Depends(get_engine),
) -> dict[str, Any]:
    """Store stats + the engine config snapshot (weights, backends)."""
    store_stats = await anyio.to_thread.run_sync(engine.store.stats)
    return {"store": store_stats, "engine": holder.config_snapshot()}


@router.get("/attributes/distribution")
async def attributes_distribution(engine=Depends(get_engine)) -> dict[str, Any]:
    """Aggregate the VLM attribute payloads: garment type / color / scene_type
    / formality counts (top 20 each)."""

    def _aggregate() -> dict[str, Any]:
        types: Counter[str] = Counter()
        colors: Counter[str] = Counter()
        scenes: Counter[str] = Counter()
        formality: Counter[str] = Counter()
        n_images = 0
        n_garments = 0
        for attrs in engine.store.iter_attributes():
            n_images += 1
            scenes[attrs.scene_type] += 1
            for g in attrs.garments:
                n_garments += 1
                if g.type:
                    types[g.type] += 1
                if g.color:
                    colors[g.color] += 1
                formality[g.formality] += 1
        return {
            "images": n_images,
            "garments": n_garments,
            "garment_types": dict(types.most_common(_TOP_N)),
            "colors": dict(colors.most_common(_TOP_N)),
            "scene_types": dict(scenes.most_common(_TOP_N)),
            "formality": dict(formality.most_common(_TOP_N)),
        }

    return await anyio.to_thread.run_sync(_aggregate)


@router.post("/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex(
    settings: Settings = Depends(get_settings),
    holder: EngineHolder = Depends(get_engine_holder),
) -> dict[str, str]:
    """Refresh artifacts and rebuild the engine in the background.

    S3 mode: mirror s3://<bucket>/vectors/index/ down first (size-diff sync);
    local mode: reload straight from disk. Serving continues on the old
    engine until the rebuilt one swaps in.
    """

    def _job() -> None:
        try:
            if settings.S3_BUCKET:
                from api.startup import sync_index  # boto3 stays lazy inside

                n = sync_index(settings)
                log.info("reindex: %d artifact file(s) refreshed from s3://%s",
                         n, settings.S3_BUCKET)
            holder.build_now()
        except Exception:  # noqa: BLE001 — background job must not die silently
            log.exception("reindex failed")

    threading.Thread(target=_job, name="reindex", daemon=True).start()
    return {"status": "reloading"}


@router.get("/index/info")
async def index_info(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Artifact files on disk: sizes + last-modified timestamps."""

    def _scan() -> dict[str, Any]:
        base: Path = settings.artifacts_path / "index"
        files: list[dict[str, Any]] = []
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                st = p.stat()
                files.append({
                    "name": p.relative_to(base).as_posix(),
                    "bytes": st.st_size,
                    "modified": datetime.fromtimestamp(
                        st.st_mtime, tz=timezone.utc).isoformat(),
                })
        return {
            "dir": str(base),
            "exists": base.is_dir(),
            "total_bytes": sum(f["bytes"] for f in files),
            "files": files,
        }

    return await anyio.to_thread.run_sync(_scan)
