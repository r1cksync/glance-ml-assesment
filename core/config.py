"""YAML config loading. Composition roots pass explicit params into ML modules —
modules never read config files themselves."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """Dict with attribute access and dotted-path get: cfg.get_path('search.weights.garment')."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None) -> Config:
    """Load YAML config. Precedence: overrides > FASHION_CONFIG env file > default.yaml."""
    cfg_path = Path(path or os.environ.get("FASHION_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if cfg_path != DEFAULT_CONFIG_PATH and DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = _deep_merge(yaml.safe_load(f) or {}, data)
    if overrides:
        data = _deep_merge(data, overrides)

    cfg = Config(data)
    # Keep model downloads off the (nearly full) system drive.
    hf_home = cfg.get_path("paths.hf_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", str((_REPO_ROOT / hf_home).resolve()))
    return cfg


def repo_root() -> Path:
    return _REPO_ROOT


def resolve(cfg: Config, dotted: str) -> Path:
    """Resolve a configured relative path against the repo root."""
    return (_REPO_ROOT / str(cfg.get_path(dotted))).resolve()
