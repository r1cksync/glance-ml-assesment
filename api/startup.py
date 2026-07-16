"""Pre-serve artifact sync: `python -m api.startup`.

Run by the Docker entrypoint before uvicorn. When S3_BUCKET is set it mirrors
s3://<bucket>/vectors/index/ into <ARTIFACTS_DIR>/index, downloading only
objects that are missing locally or whose size differs. Local mode (empty
S3_BUCKET) is a no-op. Always exits 0 on success so the entrypoint proceeds.

This is one of the few modules allowed to import boto3 (lazily, inside the
sync function) — ML logic never touches AWS SDKs.
"""
from __future__ import annotations

import logging
import sys

from api.settings import Settings

log = logging.getLogger(__name__)

S3_INDEX_PREFIX = "vectors/index/"


def sync_index(settings: Settings) -> int:
    """Mirror the S3 index prefix into ARTIFACTS_DIR/index.

    Returns the number of files downloaded (skips size-identical files).
    """
    import boto3  # lazy: local mode must not require the AWS SDK at import

    s3 = boto3.client("s3", region_name=settings.AWS_REGION)
    dest_root = settings.artifacts_path / "index"
    dest_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET,
                                   Prefix=S3_INDEX_PREFIX):
        for obj in page.get("Contents", []):
            key = str(obj["Key"])
            rel = key[len(S3_INDEX_PREFIX):]
            if not rel or rel.endswith("/"):
                continue  # prefix marker objects
            dest = dest_root / rel
            if dest.exists() and dest.stat().st_size == int(obj["Size"]):
                continue  # unchanged by size — skip
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(settings.S3_BUCKET, key, str(dest))
            downloaded += 1
            log.info("downloaded s3://%s/%s -> %s", settings.S3_BUCKET, key, dest)
    return downloaded


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings()
    if not settings.S3_BUCKET:
        log.info("S3_BUCKET not set — local mode, skipping artifact sync")
        return 0
    n = sync_index(settings)
    log.info("artifact sync complete: %d file(s) downloaded from s3://%s/%s",
             n, settings.S3_BUCKET, S3_INDEX_PREFIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
