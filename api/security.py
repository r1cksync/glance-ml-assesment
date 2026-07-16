"""Password hashing (argon2id) and JWT issuance/validation (HS256, pyjwt).

Token model:
- access token: 15 min, carried client-side in memory, claims {sub, role, jti,
  typ="access", iss, aud, iat, exp}.
- refresh token: 7 days, httpOnly cookie scoped to /auth, claims {sub, jti,
  family, typ="refresh", iss, aud, iat, exp}. Rotation + reuse detection is
  enforced by api.repos.TokenRepo (see api.auth_routes).

SECURITY: nothing in this module ever logs a token, a password, or a hash.
"""
from __future__ import annotations

import time
from typing import Any, Literal

import jwt
from argon2 import PasswordHasher
from argon2 import exceptions as argon2_exceptions
from fastapi import HTTPException, status
from pydantic import BaseModel

from api.settings import Settings

TokenType = Literal["access", "refresh"]

_ph = PasswordHasher()  # argon2id with library-default (RFC 9106 low-memory) params


class TokenPayload(BaseModel):
    """Validated claims of an access token, as seen by route dependencies."""

    sub: str
    role: str = "user"
    jti: str = ""
    typ: str = "access"
    family: str | None = None


# ── passwords ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Argon2id hash (salted, encoded with its own parameters)."""
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-time-ish verification; False on mismatch or malformed hash."""
    try:
        return bool(_ph.verify(password_hash, password))
    except argon2_exceptions.VerifyMismatchError:
        return False
    except argon2_exceptions.VerificationError:
        return False
    except argon2_exceptions.InvalidHash:  # alias kept across argon2-cffi versions
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash uses weaker-than-current parameters."""
    try:
        return bool(_ph.check_needs_rehash(password_hash))
    except argon2_exceptions.InvalidHash:
        return True


# ── JWTs ──────────────────────────────────────────────────────────────────────

def _unauthorized(detail: str = "invalid or expired credentials") -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def make_access_token(sub: str, role: str, jti: str, *, settings: Settings) -> str:
    """Short-lived bearer token carrying the caller's role."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": sub,
        "role": role,
        "jti": jti,
        "typ": "access",
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "iat": now,
        "exp": now + settings.ACCESS_TTL_SECONDS,
    }
    return jwt.encode(claims, settings.JWT_SECRET, algorithm="HS256")


def make_refresh_token(sub: str, jti: str, family: str, *, settings: Settings) -> str:
    """Rotating refresh token; `family` groups every descendant of one login."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": sub,
        "jti": jti,
        "family": family,
        "typ": "refresh",
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "iat": now,
        "exp": now + settings.REFRESH_TTL_SECONDS,
    }
    return jwt.encode(claims, settings.JWT_SECRET, algorithm="HS256")


def decode_token(token: str, expected_type: TokenType, *,
                 settings: Settings) -> dict[str, Any]:
    """Decode + validate signature, exp, iss, aud and the `typ` claim.

    Raises HTTPException(401) on ANY failure — callers never see raw
    jwt exceptions and no token material leaks into error messages.
    """
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "jti"]},
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc
    if claims.get("typ") != expected_type:
        raise _unauthorized("wrong token type")
    return claims
