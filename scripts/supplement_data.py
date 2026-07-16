"""Supplement thin corpus categories with CC-licensed images from Wikimedia Commons.

Fashionpedia val/test 2020 is runway/editorial-heavy: business ties, raincoats
and office/park environments are nearly absent — but 2 of the 5 acceptance
queries need them. Per the assignment ("if environment diversity is thin,
supplement ... and note what you added"), this script:

1. searches the keyless Wikimedia Commons API per thin category,
2. downloads candidate images (1024px thumbs),
3. **CLIP-filters** them: keeps only images whose FashionCLIP/CLIP similarity
   to the category's validation prompt clears a floor (relevance is enforced
   by the same model family that indexes them),
4. writes them to data/images as ``supp_<category>_<nn>.jpg`` and records
   source page, author and license per image in data/supplement_manifest.json
   (CC attribution).

Run `scripts/run_indexing.py` afterwards — resume mode indexes only new files.
Usage: python scripts/supplement_data.py [--per-category 15] [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(message)s")
log = logging.getLogger("supplement")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "fashion-retrieval-assessment/1.0 (student project)"}

#: category -> (commons search queries, CLIP validation prompt, similarity floor)
CATEGORIES: dict[str, tuple[list[str], str, float]] = {
    "yellow_raincoat": (
        ["person wearing yellow raincoat", "yellow rain jacket person"],
        "a person wearing a bright yellow raincoat", 0.24),
    "office_attire": (
        ["business suit office worker", "businessman office desk suit"],
        "a person in professional business attire inside an office", 0.22),
    "park_casual": (
        ["person sitting park bench", "man shirt park bench sitting"],
        "a person in casual clothes sitting on a park bench", 0.22),
    "city_casual": (
        ["casual outfit street style walking city", "person walking city street casual"],
        "a person in a casual weekend outfit walking in a city", 0.22),
    "red_tie": (
        ["man red tie white shirt", "red necktie white shirt suit"],
        "a man wearing a red tie and a white shirt", 0.24),
}


def commons_search(query: str, limit: int = 40) -> list[dict]:
    """Search Commons files; return [{title, url, descriptionurl, extmetadata}]."""
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 1024,
    }
    r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for p in pages.values():
        infos = p.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if (ii.get("width") or 0) < 400 or (ii.get("height") or 0) < 400:
            continue
        meta = ii.get("extmetadata") or {}
        out.append({
            "title": p.get("title", ""),
            "thumb_url": ii.get("thumburl") or ii.get("url"),
            "page_url": ii.get("descriptionurl", ""),
            "license": (meta.get("LicenseShortName") or {}).get("value", "unknown"),
            "artist": (meta.get("Artist") or {}).get("value", "unknown"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    from core.config import load_config
    from indexer.models.factory import build_embedder

    cfg = load_config()
    embedder = build_embedder(cfg, "garment")

    images_dir = ROOT / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data" / "supplement_manifest.json"
    manifest: dict = (json.loads(manifest_path.read_text(encoding="utf-8"))
                      if manifest_path.exists() else {})

    for cat, (queries, prompt, floor) in CATEGORIES.items():
        existing = [k for k, v in manifest.items() if v.get("category") == cat]
        if len(existing) >= args.per_category:
            log.info("%s: already have %d — skipping", cat, len(existing))
            continue
        candidates: list[dict] = []
        seen = set()
        for q in queries:
            try:
                for c in commons_search(q):
                    if c["thumb_url"] not in seen:
                        seen.add(c["thumb_url"])
                        candidates.append(c)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: search %r failed: %s", cat, q, e)
            time.sleep(1.0)
        log.info("%s: %d candidates", cat, len(candidates))
        if args.dry_run:
            continue

        # download + CLIP-validate
        scored: list[tuple[float, dict, "Image.Image"]] = []
        pvec = embedder.embed_text(prompt)
        for c in candidates:
            try:
                r = requests.get(c["thumb_url"], headers=HEADERS, timeout=30)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            sim = float(embedder.embed_image(img) @ pvec)
            if sim >= floor:
                scored.append((sim, c, img))
            time.sleep(0.3)
        scored.sort(key=lambda t: t[0], reverse=True)
        kept = scored[: args.per_category - len(existing)]
        for i, (sim, c, img) in enumerate(kept):
            name = f"supp_{cat}_{len(existing) + i:02d}.jpg"
            img.save(images_dir / name, quality=90)
            manifest[name] = {
                "category": cat, "clip_sim": round(sim, 4),
                "source_page": c["page_url"], "license": c["license"],
                "artist": c["artist"][:200], "title": c["title"],
            }
            log.info("kept %s (sim %.3f, %s)", name, sim, c["license"])
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    log.info("manifest: %d supplemental images", len(manifest))


if __name__ == "__main__":
    main()
