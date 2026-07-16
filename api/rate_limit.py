"""Shared slowapi limiter + settings-driven limit providers.

Lives in its own module so both api.main (handler wiring) and the route
modules (decorators) can import it without circular imports.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from api.deps import get_settings

limiter = Limiter(key_func=get_remote_address)


def auth_rate_limit(*_args: object) -> str:
    """Dynamic limit string for /auth endpoints (RATE_LIMIT_AUTH).

    Accepts *args because slowapi may call limit providers with or without
    the rate-limit key depending on version.
    """
    return get_settings().RATE_LIMIT_AUTH


def search_rate_limit(*_args: object) -> str:
    """Dynamic limit string for search endpoints (RATE_LIMIT_SEARCH)."""
    return get_settings().RATE_LIMIT_SEARCH
