"""FastAPI dependencies: settings, auth guards, repositories, and the engine.

get_settings is a process-wide lru_cache singleton — main.py's lifespan
mutates the same instance when it resolves the JWT secret from SSM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.security import TokenPayload, decode_token
from api.settings import Settings

if TYPE_CHECKING:  # heavy/typed-only imports
    from api.engine_holder import EngineHolder
    from api.repos import TokenRepo, UserRepo
    from retriever.search.engine import SearchEngine

_bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache
def get_settings() -> Settings:
    """Singleton Settings, shared by routes, lifespan, and rate limiters."""
    return Settings()


# ── auth guards ───────────────────────────────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> TokenPayload:
    """Validate the Bearer access token and expose its claims."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = decode_token(credentials.credentials, "access", settings=settings)
    return TokenPayload(
        sub=str(claims["sub"]),
        role=str(claims.get("role", "user")),
        jti=str(claims.get("jti", "")),
        typ="access",
    )


def require_role(role: str) -> Callable[..., TokenPayload]:
    """Dependency factory: 403 unless the access token carries `role`."""

    def _dep(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role != role:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "insufficient privileges")
        return user

    return _dep


# ── repositories (built once in main.lifespan, stored on app.state) ──────────

def get_user_repo(request: Request) -> "UserRepo":
    repo = getattr(request.app.state, "user_repo", None)
    if repo is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "service starting up")
    return repo


def get_token_repo(request: Request) -> "TokenRepo":
    repo = getattr(request.app.state, "token_repo", None)
    if repo is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "service starting up")
    return repo


# ── engine ────────────────────────────────────────────────────────────────────

def get_engine_holder(request: Request) -> "EngineHolder":
    holder = getattr(request.app.state, "engine_holder", None)
    if holder is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "engine not initialized")
    return holder


def get_engine(holder: "EngineHolder" = Depends(get_engine_holder)) -> "SearchEngine":
    """The ready SearchEngine — 503 while it is still building (or failed)."""
    engine = holder.engine
    if engine is None:
        if holder.error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "search engine failed to initialize")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "search engine is warming up — try again shortly")
    return engine
