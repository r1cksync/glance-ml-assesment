"""Auth endpoints: register / login / refresh / logout / me.

Token pattern: short-lived access token (15 min, kept in memory client-side)
plus a 7-day ROTATING refresh token in an httpOnly cookie scoped to /auth.
Every refresh revokes the presented jti and issues a new one in the same
family; presenting an already-revoked refresh token is treated as theft and
revokes the entire family (see api.repos for the full write-up).

CPU-bound work (argon2) and repository IO run in the anyio thread pool so the
event loop never blocks. Passwords and tokens are never logged.
"""
from __future__ import annotations

import logging
import time
from uuid import uuid4

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator

from api.deps import get_current_user, get_settings, get_token_repo, get_user_repo
from api.rate_limit import auth_rate_limit, limiter
from api.repos import TokenRepo, UserAlreadyExistsError, UserRecord, UserRepo
from api.security import (
    TokenPayload,
    decode_token,
    hash_password,
    make_access_token,
    make_refresh_token,
    needs_rehash,
    verify_password,
)
from api.settings import Settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"

# Lazily-built dummy hash: keeps login timing indistinguishable between
# "unknown email" and "wrong password".
_dummy_hash: str | None = None


def _get_dummy_hash() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(uuid4().hex)
    return _dummy_hash


# ── request/response models ───────────────────────────────────────────────────

def _normalize_email(v: str) -> str:
    v = v.strip().lower()
    if "@" not in v[1:-1]:
        raise ValueError("invalid email address")
    return v


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)  # password policy: >= 8 chars

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _normalize_email(v)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _normalize_email(v)


class RegisterResponse(BaseModel):
    email: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


class MeResponse(BaseModel):
    email: str
    role: str


# ── cookie helpers ────────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        max_age=settings.REFRESH_TTL_SECONDS,
        path="/auth",                       # only ever sent to /auth/* endpoints
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,  # "none" requires Secure
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=REFRESH_COOKIE,
        path="/auth",
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )


def _issue_session(user: UserRecord, settings: Settings,
                   tokens: TokenRepo) -> tuple[str, str]:
    """New token family: returns (access_token, refresh_token)."""
    family = uuid4().hex
    refresh_jti = uuid4().hex
    tokens.save(refresh_jti, user.email, family,
                int(time.time()) + settings.REFRESH_TTL_SECONDS)
    access = make_access_token(user.email, user.role, uuid4().hex,
                               settings=settings)
    refresh = make_refresh_token(user.email, refresh_jti, family,
                                 settings=settings)
    return access, refresh


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED,
             response_model=RegisterResponse)
@limiter.limit(auth_rate_limit)
async def register(
    request: Request,
    body: RegisterRequest,
    users: UserRepo = Depends(get_user_repo),
) -> RegisterResponse:
    """Create a new account (role "user"). 409 if the email is taken."""

    def _create() -> UserRecord:
        return users.create(body.email, hash_password(body.password), role="user")

    try:
        user = await anyio.to_thread.run_sync(_create)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "an account with this email already exists") from exc
    log.info("registered new user %s", user.email)
    return RegisterResponse(email=user.email, role=user.role)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(auth_rate_limit)
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    users: UserRepo = Depends(get_user_repo),
    tokens: TokenRepo = Depends(get_token_repo),
) -> TokenResponse:
    """Password login → access token in body + rotating refresh cookie."""

    def _login() -> tuple[str, str, str]:
        user = users.get_by_email(body.email)
        if user is None:
            verify_password(_get_dummy_hash(), body.password)  # timing equalizer
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "invalid email or password")
        if not verify_password(user.password_hash, body.password):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "invalid email or password")
        if needs_rehash(user.password_hash):
            try:
                users.update_password_hash(user.email, hash_password(body.password))
            except Exception:  # noqa: BLE001 — opportunistic upgrade only
                log.warning("password rehash failed for %s", user.email)
        access, refresh = _issue_session(user, settings, tokens)
        return access, refresh, user.role

    access, refresh, role = await anyio.to_thread.run_sync(_login)
    _set_refresh_cookie(response, refresh, settings)
    return TokenResponse(access_token=access,
                         expires_in=settings.ACCESS_TTL_SECONDS, role=role)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(auth_rate_limit)
async def refresh(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    users: UserRepo = Depends(get_user_repo),
    tokens: TokenRepo = Depends(get_token_repo),
) -> TokenResponse:
    """Rotate the refresh token and mint a new access token.

    Reuse detection: if the presented jti is not active (already rotated
    away, revoked, or unknown) the whole family is revoked — a replayed
    stolen cookie kills every session descended from that login.
    """
    raw = request.cookies.get(REFRESH_COOKIE)
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "missing refresh token")
    claims = decode_token(raw, "refresh", settings=settings)
    jti = str(claims["jti"])
    family = str(claims.get("family", ""))
    sub = str(claims["sub"])

    def _rotate() -> tuple[str, str, str]:
        if not tokens.is_active(jti):
            tokens.revoke_family(family)
            log.warning("refresh token reuse detected for %s — family revoked", sub)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "invalid refresh token")
        user = users.get_by_email(sub)
        if user is None:
            tokens.revoke_family(family)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "invalid refresh token")
        tokens.revoke(jti)  # rotation: old jti dies, new jti in the SAME family
        new_jti = uuid4().hex
        tokens.save(new_jti, user.email, family,
                    int(time.time()) + settings.REFRESH_TTL_SECONDS)
        access = make_access_token(user.email, user.role, uuid4().hex,
                                   settings=settings)
        new_refresh = make_refresh_token(user.email, new_jti, family,
                                         settings=settings)
        return access, new_refresh, user.role

    access, new_refresh, role = await anyio.to_thread.run_sync(_rotate)
    _set_refresh_cookie(response, new_refresh, settings)
    return TokenResponse(access_token=access,
                         expires_in=settings.ACCESS_TTL_SECONDS, role=role)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    settings: Settings = Depends(get_settings),
    tokens: TokenRepo = Depends(get_token_repo),
) -> Response:
    """Revoke the presented refresh token and clear the cookie. Always 204."""
    raw = request.cookies.get(REFRESH_COOKIE)
    if raw:
        try:
            claims = decode_token(raw, "refresh", settings=settings)
            await anyio.to_thread.run_sync(tokens.revoke, str(claims["jti"]))
        except HTTPException:
            pass  # invalid/expired cookie — still clear it below
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_refresh_cookie(response, settings)
    return response


@router.get("/me", response_model=MeResponse)
async def me(user: TokenPayload = Depends(get_current_user)) -> MeResponse:
    """Identity of the presented access token."""
    return MeResponse(email=user.sub, role=user.role)
