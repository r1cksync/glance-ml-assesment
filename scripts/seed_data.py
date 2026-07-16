"""Seed the fashion-retrieval dataset.

Ensures ``data/raw/val_test2020.zip`` exists (downloads it if not), extracts
every ``*.jpg`` flat into ``data/images`` (the archive nests them under a
``test/`` folder), optionally writes a seeded-random subset id list, and
optionally uploads all images to ``s3://<bucket>/images/<filename>``.

Idempotent throughout: an existing zip is reused, already-extracted files are
skipped, and existing S3 objects are detected with ``head_object`` and left
alone. This script is one of the few places allowed to import boto3.

Usage:
    python scripts/seed_data.py                     # download + extract only
    python scripts/seed_data.py --subset 50         # + write data/subset_ids.txt
    python scripts/seed_data.py --upload            # + push images to S3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402

log = logging.getLogger("scripts.seed_data")

DEFAULT_ZIP = REPO_ROOT / "data" / "raw" / "val_test2020.zip"
DEFAULT_SUBSET_FILE = REPO_ROOT / "data" / "subset_ids.txt"
_CHUNK = 1024 * 1024  # 1 MiB download chunks
_PROGRESS_EVERY = 100 * _CHUNK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="glance", help="AWS profile (default: glance; pass '' to use env credentials)")
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument("--bucket", default=None,
                   help="target S3 bucket (default: fashion-retrieval-<account_id>)")
    p.add_argument("--url", default=None, help="dataset zip URL (default: config dataset.source_url)")
    p.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP, help="local zip location")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="extraction target (default: config paths.images_dir)")
    p.add_argument("--subset", type=int, default=None, metavar="N",
                   help="write a seeded random sample of N filenames to data/subset_ids.txt")
    p.add_argument("--upload", action="store_true", help="upload data/images/* to s3://<bucket>/images/")
    p.add_argument("--workers", type=int, default=16, help="upload concurrency (default: 16)")
    return p.parse_args(argv)


def download(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` unless it already exists. Returns True if downloaded."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("zip already present: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % _PROGRESS_EVERY < _CHUNK:
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                log.info("  downloaded %.0f MB%s", done / 1e6, pct)
    tmp.replace(dest)
    log.info("download complete: %.1f MB", dest.stat().st_size / 1e6)
    return True


def extract_flat(zip_path: Path, images_dir: Path) -> tuple[int, int]:
    """Extract all *.jpg members flat into ``images_dir`` (drops the inner
    test/ folder). Returns (extracted, skipped)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".jpg"):
                continue
            target = images_dir / Path(info.filename).name
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    log.info("extracted %d images into %s (%d already present)",
             extracted, images_dir, skipped)
    return extracted, skipped


def write_subset(images_dir: Path, n: int, seed: int, out: Path) -> int:
    """Write a deterministic random sample of image filenames, one per line."""
    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not names:
        log.warning("no images in %s — subset not written", images_dir)
        return 0
    rng = random.Random(seed)
    sample = sorted(rng.sample(names, min(n, len(names))))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sample) + "\n")
    log.info("wrote %d subset ids to %s (seed=%d)", len(sample), out, seed)
    return len(sample)


def resolve_bucket(session: Any, explicit: str | None) -> str:
    """Default bucket name is fashion-retrieval-<account_id> (matches terraform)."""
    if explicit:
        return explicit
    account = session.client("sts").get_caller_identity()["Account"]
    return f"fashion-retrieval-{account}"


def upload_images(session: Any, bucket: str, images_dir: Path,
                  workers: int = 16) -> dict[str, int]:
    """Upload all *.jpg files to s3://bucket/images/<filename>.

    Skips objects that already exist (head_object). Thread-safe: boto3 clients
    are safe to share across threads."""
    from botocore.exceptions import ClientError

    s3 = session.client("s3")
    files = sorted(images_dir.glob("*.jpg"))
    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    def _one(path: Path) -> str:
        key = f"images/{path.name}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return "skipped"
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise
        s3.upload_file(str(path), bucket, key,
                       ExtraArgs={"ContentType": "image/jpeg"})
        return "uploaded"

    log.info("uploading %d images to s3://%s/images/ (%d workers)",
             len(files), bucket, workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, f): f for f in files}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:  # noqa: BLE001 — one bad file shouldn't kill the batch
                counts["failed"] += 1
                log.error("upload failed for %s: %s", futures[fut].name, e)
            if i % 500 == 0:
                log.info("  progress: %d/%d", i, len(files))
    log.info("upload done: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config()

    url = args.url or str(cfg.get_path("dataset.source_url"))
    seed = int(cfg.get_path("dataset.seed", 42))
    images_dir = (args.images_dir
                  or (REPO_ROOT / str(cfg.get_path("paths.images_dir", "data/images"))))
    images_dir = Path(images_dir).resolve()

    summary: dict[str, Any] = {"zip": str(args.zip_path), "images_dir": str(images_dir)}
    summary["downloaded"] = download(url, args.zip_path)
    extracted, skipped = extract_flat(args.zip_path, images_dir)
    summary["extracted"] = extracted
    summary["extract_skipped"] = skipped
    summary["images_total"] = sum(1 for _ in images_dir.glob("*.jpg"))

    if args.subset:
        summary["subset_written"] = write_subset(
            images_dir, args.subset, seed, DEFAULT_SUBSET_FILE)

    if args.upload:
        import boto3  # lazy: scripts are an allowed boto3 import site

        session = boto3.Session(profile_name=args.profile or None,
                                region_name=args.region)
        bucket = resolve_bucket(session, args.bucket)
        summary["bucket"] = bucket
        summary.update({f"upload_{k}": v for k, v in
                        upload_images(session, bucket, images_dir, args.workers).items()})

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
