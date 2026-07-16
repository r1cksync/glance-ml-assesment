"""Pick a deterministic dev subset of image filenames.

Samples N ids (default 50, seeded from config) from ``data/images`` and writes
them to ``data/subset_ids.txt``, one filename per line — the file consumed by
``scripts/run_indexing.py --subset-file``.

Usage:
    python scripts/make_subset.py
    python scripts/make_subset.py --n 100 --seed 7
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402

log = logging.getLogger("scripts.make_subset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=None,
                   help="subset size (default: config dataset.subset_size, i.e. 50)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (default: config dataset.seed)")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="image directory (default: config paths.images_dir)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "subset_ids.txt",
                   help="output file (default: data/subset_ids.txt)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config()

    n = args.n if args.n is not None else int(cfg.get_path("dataset.subset_size", 50))
    seed = args.seed if args.seed is not None else int(cfg.get_path("dataset.seed", 42))
    images_dir = (args.images_dir
                  or REPO_ROOT / str(cfg.get_path("paths.images_dir", "data/images")))
    images_dir = Path(images_dir).resolve()

    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not names:
        log.error("no *.jpg files in %s — run scripts/seed_data.py first", images_dir)
        return 1

    rng = random.Random(seed)
    sample = sorted(rng.sample(names, min(n, len(names))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sample) + "\n")

    log.info("wrote %d ids (seed=%d) to %s", len(sample), seed, args.out)
    print(f"{args.out}: {len(sample)} ids (seed={seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
