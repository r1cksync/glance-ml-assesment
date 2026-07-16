"""Search endpoints (auth-guarded): text search, query-by-image, parse debug.

The SearchEngine is synchronous and CPU-bound (CLIP forward passes, FAISS),
so every engine call runs via anyio.to_thread.run_sync — the event loop stays
free to serve health checks and auth while a query computes.

Request latency metrics are emitted by the EMF middleware in api.main.
"""
from __future__ import annotations

import io
import logging
from functools import partial
from typing import Optional

import anyio.to_thread
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from api.deps import (
    get_current_user,
    get_engine,
    get_engine_holder,
    get_settings,
)
from api.engine_holder import EngineHolder
from api.rate_limit import limiter, search_rate_limit
from api.settings import Settings
from core.schemas import (
    ComponentScores,
    ImageAttributes,
    MatchExplanation,
    ParsedQuery,
    SearchResult,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["search"], dependencies=[Depends(get_current_user)])

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB image upload cap


# ── request/response models ───────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(10, ge=1, le=100)
    use_rerank: Optional[bool] = None      # None => server default (USE_RERANK)


class ParseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class ResultOut(BaseModel):
    image_id: str
    url: str
    score: float
    components: ComponentScores
    matches: list[MatchExplanation]
    attributes: Optional[ImageAttributes] = None


class SearchResponse(BaseModel):
    query: str
    parsed: Optional[ParsedQuery] = None
    results: list[ResultOut]


def _to_response(query: str, parsed: Optional[ParsedQuery],
                 results: list[SearchResult],
                 holder: EngineHolder) -> SearchResponse:
    return SearchResponse(
        query=query,
        parsed=parsed,
        results=[
            ResultOut(
                image_id=r.image_id,
                url=holder.image_url(r.image_id),
                score=r.score,
                components=r.components,
                matches=r.matches,
                attributes=r.attributes,
            )
            for r in results
        ],
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
@limiter.limit(search_rate_limit)
async def search(
    request: Request,
    body: SearchRequest,
    settings: Settings = Depends(get_settings),
    holder: EngineHolder = Depends(get_engine_holder),
    engine=Depends(get_engine),
) -> SearchResponse:
    """Natural-language search → parsed intent + explainable ranked results."""
    use_rerank = (settings.USE_RERANK if body.use_rerank is None
                  else body.use_rerank)
    parsed, results = await anyio.to_thread.run_sync(
        partial(engine.search, body.query, k=body.k, use_rerank=use_rerank))
    return _to_response(body.query, parsed, results, holder)


@router.post("/search/image", response_model=SearchResponse)
@limiter.limit(search_rate_limit)
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    refinement: Optional[str] = Form(None, max_length=500),
    k: int = Form(10, ge=1, le=100),
    use_rerank: bool = Form(False),
    holder: EngineHolder = Depends(get_engine_holder),
    engine=Depends(get_engine),
) -> SearchResponse:
    """Query-by-example with optional text refinement ("but in green")."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "image exceeds the 15 MB upload limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")

    def _run() -> tuple[Optional[ParsedQuery], list[SearchResult]]:
        from PIL import Image, UnidentifiedImageError  # lazy: decode in worker thread

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "uploaded file is not a valid image") from exc
        return engine.search_by_image(image, refinement=refinement, k=k,
                                      use_rerank=use_rerank)

    parsed, results = await anyio.to_thread.run_sync(_run)
    return _to_response(refinement or "", parsed, results, holder)


@router.post("/parse", response_model=ParsedQuery)
@limiter.limit(search_rate_limit)
async def parse(
    request: Request,
    body: ParseRequest,
    engine=Depends(get_engine),
) -> ParsedQuery:
    """Debug view: how the parser decomposes a query (no retrieval)."""
    return await anyio.to_thread.run_sync(engine.parser.parse, body.query)
