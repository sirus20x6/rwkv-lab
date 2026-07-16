#!/usr/bin/env python3
"""Build a deterministic, source-blinded visual-quality review pack.

The public review manifest intentionally has no dataset names.  Keep
``source_key.jsonl`` out of the reviewer-facing bundle until scoring is done.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
_BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def choose(items: Iterable[Any], n: int, rng: random.Random) -> list[Any]:
    """Reservoir sample, without loading a potentially huge iterator."""
    result: list[Any] = []
    for index, item in enumerate(items):
        if index < n:
            result.append(item)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < n:
                result[replacement] = item
    return result


def parquet_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def image_suffix(path_or_url: str) -> str:
    suffix = Path(path_or_url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def cidv0_from_sha256(digest: bytes) -> str:
    """Encode a 32-byte SHA-256 digest as the conventional IPFS CIDv0."""
    if len(digest) != 32:
        raise ValueError(f"expected 32-byte digest, received {len(digest)} bytes")
    value = int.from_bytes(b"\x12\x20" + digest, "big")  # sha2-256 multihash
    encoded = ""
    while value:
        value, remainder = divmod(value, 58)
        encoded = _BASE58[remainder] + encoded
    return encoded


def fetch(url: str, destination: Path) -> bool:
    candidates = [url]
    if url.startswith("ipfs://"):
        cid = url.removeprefix("ipfs://")
        candidates = [f"https://{gateway}/ipfs/{cid}" for gateway in ("ipfs.io", "dweb.link", "gateway.pinata.cloud")]
    for candidate in candidates:
        try:
            request = urllib.request.Request(candidate, headers={"User-Agent": "rwkv-lab-quality-review/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
            if destination.stat().st_size > 0:
                return True
        except Exception:
            destination.unlink(missing_ok=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "quality_review_pack")
    parser.add_argument("--per-dataset", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.per_dataset <= 0:
        raise SystemExit("--per-dataset must be positive")
    rng = random.Random(args.seed)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []

    def add(*, source: str, original: str, review_text: str, copy_from: Path | None = None,
            url: str | None = None, task: str) -> None:
        item_id = f"item-{len(manifest) + 1:04d}"
        suffix = image_suffix(str(copy_from or url or original))
        destination = images / f"{item_id}{suffix}"
        available = False
        if copy_from is not None:
            shutil.copy2(copy_from, destination)
            available = True
        elif url:
            available = fetch(url, destination)
        manifest.append({"id": item_id, "task": task, "image": str(destination.relative_to(out)) if available else None,
                         "reference": review_text, "image_unavailable": not available})
        key.append({"id": item_id, "source_dataset": source, "original": original, "url": url})

    # 1. WebDataset image + tag/title captions.
    grid_root = ROOT / "nsfw-video-still-caption-grid-only/extracted"
    grid = choose(sorted(grid_root.glob("*/*.json")), args.per_dataset, rng)
    for meta_path in grid:
        meta = json.loads(meta_path.read_text())
        image = meta_path.with_suffix(".jpg")
        add(source="nsfw-video-still-caption-grid-only", original=str(meta_path.relative_to(ROOT)),
            copy_from=image, task="image_caption_grounding", review_text=meta.get("caption", ""))

    # 2. Pose VRLens stored as image bytes plus pose/bounding-box metadata.
    pose_rows = choose(parquet_rows([ROOT / "pose_vrlens_nsfw/data/train-00000-of-00001.parquet"]), args.per_dataset, rng)
    for row in pose_rows:
        item_id = f"item-{len(manifest) + 1:04d}"
        payload = row["image"]["bytes"]
        destination = images / f"{item_id}.jpg"
        destination.write_bytes(payload)
        manifest.append({"id": item_id, "task": "pose_annotation_quality", "image": str(destination.relative_to(out)),
                         "reference": json.dumps(row["objects"], ensure_ascii=False), "image_unavailable": False})
        key.append({"id": item_id, "source_dataset": "pose_vrlens_nsfw", "original": row["image"].get("path"), "url": None})

    # 3. Anime image/tag-text pairs.  Restrict to B variants because those have captions.
    anime_root = ROOT / "nsfw_scene_animes_31-03/extracted"
    anime = choose(sorted(anime_root.glob("*_B.txt")), args.per_dataset, rng)
    for caption in anime:
        image = caption.with_suffix(".jpg")
        add(source="nsfw_scene_animes_31-03", original=str(caption.relative_to(ROOT)), copy_from=image,
            task="image_tag_grounding", review_text=caption.read_text().strip())

    # 4. Detector labels, stratified across its five classes.
    detector_root = ROOT / "nsfw_detect/extracted/nsfw_dataset_v1"
    labels = sorted(path.name for path in detector_root.iterdir() if path.is_dir())
    base_per_label, remainder = divmod(args.per_dataset, len(labels))
    detector: list[tuple[str, Path]] = []
    for index, label in enumerate(labels):
        count = base_per_label + (index < remainder)
        detector.extend((label, path) for path in choose(sorted((detector_root / label).glob("*")), count, rng))
    for label, image in detector[:args.per_dataset]:
        add(source="nsfw_detect", original=str(image.relative_to(ROOT)), copy_from=image, task="safety_label_quality", review_text=label)

    # 5. Joy captions use remote image URLs.
    joy_rows = choose(parquet_rows(sorted((ROOT / "joy-captioning-20250408a/data").glob("*.parquet"))), args.per_dataset, rng)
    for row in joy_rows:
        urls = row.get("urls") or []
        url = urls[0] if urls else f"ipfs://{cidv0_from_sha256(row['filehash'])}"
        add(source="joy-captioning-20250408a", original=row["filehash"].hex(), url=url,
            task="image_caption_grounding", review_text=f"Prompt: {row['question']}\n\nAnswer: {row['answer']}")

    # 6. Manga metadata has a remote cover image, rather than local pages.
    manga_rows = choose(parquet_rows([ROOT / "NSFW_Manga/manga_all.parquet"]), args.per_dataset, rng)
    for row in manga_rows:
        reference = "\n".join((f"Title: {row['title']}", f"Tags: {row['tags']}", f"Categories: {row['categories']}",
                                f"Language: {row['language']}"))
        add(source="NSFW_Manga", original=str(row["id"]), url=row.get("cover_image"), task="metadata_to_cover_consistency", review_text=reference)

    with (out / "review_manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "source_key.jsonl").open("w") as handle:
        for row in key:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(
        "# Blinded visual-quality review pack\n\n"
        "Use `review_manifest.jsonl` and `images/` for scoring. Do not expose `source_key.jsonl` to reviewers until scores are finalized. "
        "Items without local images retain their reference text but could not be fetched from their dataset-provided URL.\n"
    )
    print(f"wrote {len(manifest)} items; {sum(row['image_unavailable'] for row in manifest)} images unavailable")


if __name__ == "__main__":
    main()
