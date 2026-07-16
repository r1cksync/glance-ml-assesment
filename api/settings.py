"""Environment-driven runtime settings for the API service.

Every deployment knob is an env var (12-factor). Empty strings mean
"local/dev mode": no S3 sync, in-memory repositories, images served
statically from IMAGES_DIR at /local-images/{id}.

boto3 is imported lazily and only when JWT_SECRET_SSM_PARAM is configured —
this module stays importable without AWS SDKs installed.
"""
from __future__ import annotations

import logging
import secrets
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


def _resolve_dir(value: str) -> Path:
    """Resolve a configured directory: absolute paths win, relative paths are
    interpreted against the repo root (same convention as core.config.resolve)."""
    path = Path(value)
    if path.is_absolute():
        return path
    from core.config import repo_root  # lazy: keeps api.settings dependency-minimal

    return (repo_root() / path).resolve()


class Settings(BaseSettings):
    """All runtime configuration for the API process (env-overridable)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    APP_NAME: str = "fashion-retrieval-api"
    AWS_REGION: str = "us-east-1"

    # ── artifacts & images ────────────────────────────────────────────────────
    S3_BUCKET: str = ""                    # empty => local mode, no S3 artifact sync
    ARTIFACTS_DIR: str = "data/artifacts"  # index shards + attribute payloads
    IMAGES_DIR: str = "data/images"        # raw dataset images
    IMAGE_BASE_URL: str = ""               # empty => serve /local-images/{id} from IMAGES_DIR

    # ── persistence (empty => in-memory repositories) ─────────────────────────
    TABLE_USERS: str = ""
    TABLE_TOKENS: str = ""
    TABLE_CACHE: str = ""

    # ── auth / tokens ─────────────────────────────────────────────────────────
    JWT_SECRET: str = ""                   # direct secret, or:
    JWT_SECRET_SSM_PARAM: str = ""         # SSM SecureString parameter fetched at startup
    JWT_ISSUER: str = "fashion-retrieval"
    JWT_AUDIENCE: str = "fashion-retrieval-web"
    ACCESS_TTL_SECONDS: int = 900          # 15 min in-memory access token
    REFRESH_TTL_SECONDS: int = 604800      # 7 day rotating httpOnly refresh cookie
    COOKIE_SECURE: bool = True             # SameSite=None REQUIRES Secure
    COOKIE_SAMESITE: str = "none"          # none | lax | strict

    # ── http ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = ""                 # comma-separated origin list
    RATE_LIMIT_AUTH: str = "5/minute"
    RATE_LIMIT_SEARCH: str = "30/minute"

    # ── engine ────────────────────────────────────────────────────────────────
    USE_RERANK: bool = True                # default for the BLIP-ITM rerank stage

    # ── optional auto-seeded accounts ─────────────────────────────────────────
    DEMO_USER_EMAIL: str = ""
    DEMO_USER_PASSWORD: str = ""
    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD: str = ""

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def artifacts_path(self) -> Path:
        return _resolve_dir(self.ARTIFACTS_DIR)

    @property
    def images_path(self) -> Path:
        return _resolve_dir(self.IMAGES_DIR)

    def resolve_jwt_secret(self) -> None:
        """Ensure JWT_SECRET is populated. Precedence:

        1. JWT_SECRET env var (already set — nothing to do)
        2. JWT_SECRET_SSM_PARAM: fetched once from SSM Parameter Store (SecureString)
        3. fallback: ephemeral random secret (dev only — tokens die with the process)

        The secret value itself is never logged.
        """
        if self.JWT_SECRET:
            return
        if self.JWT_SECRET_SSM_PARAM:
            import boto3  # lazy: only the AWS-configured path needs the SDK

            ssm = boto3.client("ssm", region_name=self.AWS_REGION)
            resp = ssm.get_parameter(
                Name=self.JWT_SECRET_SSM_PARAM, WithDecryption=True
            )
            self.JWT_SECRET = str(resp["Parameter"]["Value"])
            log.info("JWT secret loaded from SSM parameter %s",
                     self.JWT_SECRET_SSM_PARAM)
            return
        self.JWT_SECRET = secrets.token_urlsafe(48)
        log.warning(
            "JWT_SECRET not configured — generated an ephemeral secret; "
            "all tokens will be invalidated on restart (dev mode only)."
        )
