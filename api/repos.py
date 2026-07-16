"""Repositories: users, refresh-token rotation state, and the parse cache.

Each repo has an in-memory implementation (default; empty TABLE_* settings)
and a DynamoDB implementation (selected when the table name is configured).

Refresh-token rotation & stolen-token defense
---------------------------------------------
Refresh tokens rotate: every successful /auth/refresh revokes the presented
jti and issues a NEW jti in the SAME family. Only the *hash* (SHA-256) of the
jti is ever stored, so a database leak cannot be replayed as a cookie.
If a refresh token arrives whose jti is already revoked (or unknown while its
family is known), it was either replayed by an attacker or the legitimate
client lost a race after theft — either way the WHOLE family is revoked
(`revoke_family`), forcibly logging out every descendant of that login.

DynamoDB table shapes (all on-demand, TTL on `expires_at`):
- TABLE_USERS:  pk "email"           {email, password_hash, role, created_at}
- TABLE_TOKENS: pk "jti_hash"        {jti_hash, sub, family, expires_at, revoked}
                family tombstones stored as jti_hash = "family#<sha256(family)>"
- TABLE_CACHE:  pk "cache_key"       {cache_key, value, expires_at}

boto3 is imported lazily INSIDE the Dynamo classes only — importing this
module never pulls the AWS SDK.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from api.settings import Settings

log = logging.getLogger(__name__)

PARSE_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class UserAlreadyExistsError(Exception):
    """Raised by UserRepo.create on a duplicate email."""


# ── records & protocols ───────────────────────────────────────────────────────

@dataclass
class UserRecord:
    email: str
    password_hash: str
    role: str = "user"
    created_at: int = 0


class UserRepo(Protocol):
    def get_by_email(self, email: str) -> Optional[UserRecord]: ...
    def create(self, email: str, password_hash: str, role: str = "user") -> UserRecord: ...
    def update_password_hash(self, email: str, password_hash: str) -> None: ...


class TokenRepo(Protocol):
    """Refresh-token rotation state, keyed by SHA-256(jti)."""

    def save(self, jti: str, sub: str, family: str, exp: int) -> None: ...
    def is_active(self, jti: str) -> bool: ...
    def revoke(self, jti: str) -> None: ...
    def revoke_family(self, family: str) -> None: ...


class ParseCache(Protocol):
    """Mirror of retriever.parsing.base.ParseCache (string get/put)."""

    def get(self, key: str) -> Optional[str]: ...
    def put(self, key: str, value: str) -> None: ...


# ── in-memory implementations ─────────────────────────────────────────────────

class MemoryUserRepo:
    """Dev/test users store. NOT persistent."""

    def __init__(self) -> None:
        self._users: dict[str, UserRecord] = {}
        self._lock = threading.Lock()

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        return self._users.get(email.strip().lower())

    def create(self, email: str, password_hash: str, role: str = "user") -> UserRecord:
        email = email.strip().lower()
        with self._lock:
            if email in self._users:
                raise UserAlreadyExistsError(email)
            record = UserRecord(email=email, password_hash=password_hash,
                                role=role, created_at=int(time.time()))
            self._users[email] = record
            return record

    def update_password_hash(self, email: str, password_hash: str) -> None:
        with self._lock:
            record = self._users.get(email.strip().lower())
            if record is not None:
                record.password_hash = password_hash


@dataclass
class _TokenRow:
    sub: str
    family: str
    expires_at: int
    revoked: bool = False


class MemoryTokenRepo:
    """Dev/test refresh-token state. NOT persistent."""

    def __init__(self) -> None:
        self._rows: dict[str, _TokenRow] = {}          # sha256(jti) -> row
        self._revoked_families: set[str] = set()       # sha256(family)
        self._lock = threading.Lock()

    def save(self, jti: str, sub: str, family: str, exp: int) -> None:
        with self._lock:
            self._rows[sha256_hex(jti)] = _TokenRow(
                sub=sub, family=sha256_hex(family), expires_at=int(exp))

    def is_active(self, jti: str) -> bool:
        with self._lock:
            row = self._rows.get(sha256_hex(jti))
            if row is None or row.revoked:
                return False
            if row.expires_at <= int(time.time()):
                return False
            return row.family not in self._revoked_families

    def revoke(self, jti: str) -> None:
        with self._lock:
            row = self._rows.get(sha256_hex(jti))
            if row is not None:
                row.revoked = True

    def revoke_family(self, family: str) -> None:
        fh = sha256_hex(family)
        with self._lock:
            self._revoked_families.add(fh)
            for row in self._rows.values():
                if row.family == fh:
                    row.revoked = True


@dataclass
class _CacheRow:
    value: str
    expires_at: float


class MemoryParseCache:
    """TTL'd in-process parse cache (mirrors the DynamoDB semantics)."""

    def __init__(self, ttl_seconds: int = PARSE_CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._rows: dict[str, _CacheRow] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                return None
            if row.expires_at <= time.time():
                del self._rows[key]
                return None
            return row.value

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._rows[key] = _CacheRow(value=value,
                                        expires_at=time.time() + self._ttl)


# ── DynamoDB implementations (boto3 imported lazily, ONLY here) ──────────────

class DynamoUserRepo:
    """Users table, pk "email". Duplicate protection via conditional put."""

    def __init__(self, table_name: str, region: str) -> None:
        import boto3  # lazy: only when the DynamoDB backend is configured

        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        resp = self._table.get_item(Key={"email": email.strip().lower()})
        item = resp.get("Item")
        if not item:
            return None
        return UserRecord(
            email=str(item["email"]),
            password_hash=str(item["password_hash"]),
            role=str(item.get("role", "user")),
            created_at=int(item.get("created_at", 0)),
        )

    def create(self, email: str, password_hash: str, role: str = "user") -> UserRecord:
        from botocore.exceptions import ClientError  # lazy, ships with boto3

        email = email.strip().lower()
        record = UserRecord(email=email, password_hash=password_hash,
                            role=role, created_at=int(time.time()))
        try:
            self._table.put_item(
                Item={
                    "email": record.email,
                    "password_hash": record.password_hash,
                    "role": record.role,
                    "created_at": record.created_at,
                },
                ConditionExpression="attribute_not_exists(email)",
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                raise UserAlreadyExistsError(email) from exc
            raise
        return record

    def update_password_hash(self, email: str, password_hash: str) -> None:
        self._table.update_item(
            Key={"email": email.strip().lower()},
            UpdateExpression="SET password_hash = :h",
            ExpressionAttributeValues={":h": password_hash},
        )


class DynamoTokenRepo:
    """Tokens table, pk "jti_hash", TTL attribute "expires_at".

    Family revocation is O(1): a tombstone item is written under
    "family#<sha256(family)>" and every is_active() check consults it —
    no scans, no GSI needed.
    """

    _FAMILY_PREFIX = "family#"

    def __init__(self, table_name: str, region: str,
                 family_ttl_seconds: int = 8 * 24 * 3600) -> None:
        import boto3  # lazy: only when the DynamoDB backend is configured

        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self._family_ttl = family_ttl_seconds

    def save(self, jti: str, sub: str, family: str, exp: int) -> None:
        self._table.put_item(Item={
            "jti_hash": sha256_hex(jti),
            "sub": sub,
            "family": sha256_hex(family),
            "expires_at": int(exp),
            "revoked": False,
        })

    def is_active(self, jti: str) -> bool:
        resp = self._table.get_item(Key={"jti_hash": sha256_hex(jti)},
                                    ConsistentRead=True)
        item = resp.get("Item")
        if not item or bool(item.get("revoked")):
            return False
        if int(item.get("expires_at", 0)) <= int(time.time()):
            return False  # TTL deletion can lag — enforce expiry ourselves
        fam = self._table.get_item(
            Key={"jti_hash": self._FAMILY_PREFIX + str(item.get("family", ""))},
            ConsistentRead=True,
        )
        return fam.get("Item") is None

    def revoke(self, jti: str) -> None:
        self._table.update_item(
            Key={"jti_hash": sha256_hex(jti)},
            UpdateExpression="SET revoked = :t",
            ExpressionAttributeValues={":t": True},
        )

    def revoke_family(self, family: str) -> None:
        self._table.put_item(Item={
            "jti_hash": self._FAMILY_PREFIX + sha256_hex(family),
            "expires_at": int(time.time()) + self._family_ttl,
            "revoked": True,
        })


class DynamoParseCache:
    """ParseCacheRepo: retriever ParseCache protocol backed by DynamoDB.

    pk "cache_key"; items carry a 7-day TTL via "expires_at". Used by the
    retriever's CachedParser so LLM query parses are shared across replicas.
    """

    def __init__(self, table_name: str, region: str,
                 ttl_seconds: int = PARSE_CACHE_TTL_SECONDS) -> None:
        import boto3  # lazy: only when the DynamoDB backend is configured

        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        try:
            resp = self._table.get_item(Key={"cache_key": key})
        except Exception:  # noqa: BLE001 — cache read failure must not fail a query
            log.warning("parse-cache read failed", exc_info=True)
            return None
        item = resp.get("Item")
        if not item:
            return None
        if int(item.get("expires_at", 0)) <= int(time.time()):
            return None  # TTL deletion lags; treat as miss
        value = item.get("value")
        return str(value) if value is not None else None

    def put(self, key: str, value: str) -> None:
        try:
            self._table.put_item(Item={
                "cache_key": key,
                "value": value,
                "expires_at": int(time.time()) + self._ttl,
            })
        except Exception:  # noqa: BLE001 — cache write failure must not fail a query
            log.warning("parse-cache write failed", exc_info=True)


# ── factories (backend selection from settings) ──────────────────────────────

def build_user_repo(settings: Settings) -> UserRepo:
    if settings.TABLE_USERS:
        log.info("user repo: DynamoDB table %s", settings.TABLE_USERS)
        return DynamoUserRepo(settings.TABLE_USERS, settings.AWS_REGION)
    log.info("user repo: in-memory (TABLE_USERS not set)")
    return MemoryUserRepo()


def build_token_repo(settings: Settings) -> TokenRepo:
    if settings.TABLE_TOKENS:
        log.info("token repo: DynamoDB table %s", settings.TABLE_TOKENS)
        return DynamoTokenRepo(settings.TABLE_TOKENS, settings.AWS_REGION,
                               family_ttl_seconds=settings.REFRESH_TTL_SECONDS + 86400)
    log.info("token repo: in-memory (TABLE_TOKENS not set)")
    return MemoryTokenRepo()


def build_parse_cache(settings: Settings) -> ParseCache:
    if settings.TABLE_CACHE:
        log.info("parse cache: DynamoDB table %s", settings.TABLE_CACHE)
        return DynamoParseCache(settings.TABLE_CACHE, settings.AWS_REGION)
    log.info("parse cache: in-memory (TABLE_CACHE not set)")
    return MemoryParseCache()
