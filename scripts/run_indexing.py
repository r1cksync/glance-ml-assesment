"""Run the offline indexing pipeline (composition root).

Loads config (with CLI overrides applied through ``load_config(overrides=...)``),
builds the pipeline via ``core.bootstrap.build_pipeline``, runs it, then prints
the run summary followed by the vector-store stats.

Usage:
    python scripts/run_indexing.py --limit 50
    python scripts/run_indexing.py --subset-file data/subset_ids.txt --extractor moondream
    python scripts/run_indexing.py --config config/default.yaml --no-resume
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("scripts.run_indexing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config YAML (default: config/default.yaml)")
    p.add_argument("--images-dir", default=None, help="override paths.images_dir")
    p.add_argument("--limit", type=int, default=None, help="index at most N images")
    p.add_argument("--subset-file", type=Path, default=None,
                   help="only index the image ids listed in this file (one per line)")
    p.add_argument("--extractor", default=None,
                   help="override extraction.backend (qwen2vl | moondream | bedrock)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip images already present in the store (default: --resume)")
    return p.parse_args(argv)


def _filter_kwargs(fn: Any, requested: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs ``fn`` actually accepts (pass everything through
    if it takes **kwargs). None values are dropped so pipeline defaults apply."""
    requested = {k: v for k, v in requested.items() if v is not None}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C-implemented — pass as-is
        return requested
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return requested
    accepted = {k: v for k, v in requested.items() if k in params}
    dropped = set(requested) - set(accepted)
    if dropped:
        log.warning("pipeline.run does not accept %s — ignoring", sorted(dropped))
    return accepted


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    from core.bootstrap import build_pipeline
    from core.config import load_config, resolve

    overrides: dict[str, Any] = {}
    if args.extractor:
        overrides.setdefault("extraction", {})["backend"] = args.extractor
    if args.images_dir:
        overrides.setdefault("paths", {})["images_dir"] = args.images_dir
    cfg = load_config(args.config, overrides=overrides or None)

    subset_ids: list[str] | None = None
    if args.subset_file:
        with open(args.subset_file, encoding="utf-8") as f:
            subset_ids = [line.strip() for line in f if line.strip()]
        log.info("restricting to %d ids from %s", len(subset_ids), args.subset_file)

    pipeline = build_pipeline(cfg)

    kwargs = _filter_kwargs(pipeline.run, {
        "images_dir": resolve(cfg, "paths.images_dir"),
        "limit": args.limit,
        "image_ids": subset_ids,
        "resume": args.resume,
    })
    log.info("starting indexing run: %s",
             {k: (len(v) if isinstance(v, list) else str(v)) for k, v in kwargs.items()})
    summary = pipeline.run(**kwargs)

    print(json.dumps(summary if isinstance(summary, dict) else {"summary": str(summary)},
                     indent=2, default=str))
    store = getattr(pipeline, "store", None)
    if store is not None:
        print(json.dumps(store.stats(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
