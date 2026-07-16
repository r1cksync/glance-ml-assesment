"""FastAPI serving layer: auth, search, admin and observability endpoints.

Kept import-light — heavy ML imports (core.bootstrap, torch, transformers)
happen lazily inside api.engine_holder so unit tests can import api.* cheaply.
"""
