"""Sync index/ONNX artifacts between the local workspace and S3.

push:  data/artifacts/index/**  ->  s3://<bucket>/vectors/index/**
       data/artifacts/onnx/**   ->  s3://<bucket>/vectors/onnx/**   (if present)
pull:  the reverse.

Files whose size already matches on the other side are skipped, so repeated
syncs are cheap and idempotent. This script is one of the few places allowed
to import boto3.

Usage:
    python scripts/sync_artifacts.py push
    python scripts/sync_artifacts.py pull --profile glance
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("scripts.sync_artifacts")

#: (local directory, S3 key prefix) pairs kept in sync.
SYNC_PAIRS: list[tuple[Path, str]] = [
    (REPO_ROOT / "data" / "artifacts" / "index", "vectors/index"),
    (REPO_ROOT / "data" / "artifacts" / "onnx", "vectors/onnx"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("push", "pull"), help="direction of the sync")
    p.add_argument("--profile", default="glance", help="AWS profile (default: glance; pass '' to use env credentials)")
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument("--bucket", default=None,
                   help="S3 bucket (default: fashion-retrieval-<account_id>)")
    p.add_argument("--workers", type=int, default=8, help="transfer concurrency")
    return p.parse_args(argv)


def resolve_bucket(session: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    account = session.client("sts").get_caller_identity()["Account"]
    return f"fashion-retrieval-{account}"


def _s3_size(s3: Any, bucket: str, key: str) -> int | None:
    """Object size in bytes, or None if the object does not exist."""
    from botocore.exceptions import ClientError

    try:
        return int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def _run_batch(jobs: list[tuple[str, Any]], workers: int) -> dict[str, int]:
    """Run (label, thunk) jobs concurrently, counting outcomes."""
    counts = {"transferred": 0, "skipped": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(thunk): label for label, thunk in jobs}
        for fut in concurrent.futures.as_completed(futures):
            try:
                counts[fut.result()] += 1
            except Exception as e:  # noqa: BLE001 — keep syncing the rest
                counts["failed"] += 1
                log.error("transfer failed for %s: %s", futures[fut], e)
    return counts


def push(s3: Any, bucket: str, workers: int) -> dict[str, int]:
    jobs: list[tuple[str, Any]] = []
    for local_dir, prefix in SYNC_PAIRS:
        if not local_dir.is_dir():
            log.info("skipping %s (not present locally)", local_dir)
            continue
        for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            key = f"{prefix}/{path.relative_to(local_dir).as_posix()}"

            def _job(path: Path = path, key: str = key) -> str:
                if _s3_size(s3, bucket, key) == path.stat().st_size:
                    return "skipped"
                s3.upload_file(str(path), bucket, key)
                log.info("uploaded %s -> s3://%s/%s", path.name, bucket, key)
                return "transferred"

            jobs.append((key, _job))
    return _run_batch(jobs, workers)


def pull(s3: Any, bucket: str, workers: int) -> dict[str, int]:
    jobs: list[tuple[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for local_dir, prefix in SYNC_PAIRS:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                rel = key[len(prefix) + 1:]
                if not rel or key.endswith("/"):
                    continue  # directory marker
                target = local_dir / Path(rel)
                size = int(obj["Size"])

                def _job(target: Path = target, key: str = key, size: int = size) -> str:
                    if target.exists() and target.stat().st_size == size:
                        return "skipped"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(target))
                    log.info("downloaded s3://%s/%s -> %s", bucket, key, target)
                    return "transferred"

                jobs.append((key, _job))
    return _run_batch(jobs, workers)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    import boto3  # lazy: scripts are an allowed boto3 import site

    session = boto3.Session(profile_name=args.profile or None,
                            region_name=args.region)
    bucket = resolve_bucket(session, args.bucket)
    s3 = session.client("s3")

    counts = push(s3, bucket, args.workers) if args.action == "push" \
        else pull(s3, bucket, args.workers)

    print(json.dumps({"action": args.action, "bucket": bucket, **counts}, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
