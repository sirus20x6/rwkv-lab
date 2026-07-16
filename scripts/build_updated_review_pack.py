#!/usr/bin/env python3
"""Build a source-labeled review pack that applies the first audit decisions."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from build_curated_vision_manifests import clean_anime_caption, labeled_text


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["filehash", "answer", "question_type", "is_human"], batch_size=4096):
        yield from batch.to_pylist()


def main() -> None:
    source = ROOT / "quality_review_pack"
    out = ROOT / "quality_review_updated"
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    images = out / "images"
    images.mkdir(parents=True)
    manifest = [json.loads(line) for line in (source / "review_manifest.jsonl").open() if line.strip()]
    key = {row["id"]: row for row in (json.loads(line) for line in (source / "source_key.jsonl").open() if line.strip())}
    output = []

    def add(dataset: str, task: str, reference: str, source_image: Path) -> None:
        item_id = f"item-{len(output) + 1:04d}"
        destination = images / f"{item_id}{source_image.suffix.lower() or '.jpg'}"
        shutil.copy2(source_image, destination)
        output.append({"id": item_id, "source_dataset": dataset, "task": task,
                       "image": str(destination.relative_to(out)), "reference": reference, "image_unavailable": False})

    # Retain visual samples from good sources, but replace raw text with cleaned form.
    for item in manifest:
        dataset = item.get("source_dataset")
        if dataset in {"nsfw_detect", "joy-captioning-20250408a"} or not item.get("image"):
            continue
        image = source / item["image"]
        record = key[item["id"]]
        reference = item["reference"]
        if dataset == "nsfw-video-still-caption-grid-only":
            meta = json.loads((ROOT / record["original"]).read_text())
            reference = labeled_text(tags=meta.get("tags"), categories=meta.get("categories"), cast=meta.get("cast"))
        elif dataset == "nsfw_scene_animes_31-03":
            reference = clean_anime_caption((ROOT / record["original"]).read_text().strip())
        elif dataset == "NSFW_Manga":
            reference = "\n".join(
                line for line in reference.splitlines()
                if line.startswith(("Title:", "Tags:")) and line.partition(":")[2].strip()
            )
        add(dataset, item["task"], reference, image)

    # Joy: only locally downloaded direct-url matches, with the bare answer text.
    available = {path.stem: path for path in (ROOT / "joy-captioning-20250408a/images_direct").glob("*") if path.is_file()}
    candidates = []
    for parquet in sorted((ROOT / "joy-captioning-20250408a/data").glob("*.parquet")):
        for row in rows(parquet):
            image = available.get(row["filehash"].hex())
            if image:
                candidates.append((image, row))
    rng = random.Random(20260714)
    for image, row in rng.sample(candidates, min(32, len(candidates))):
        add("joy-captioning-20250408a", "image_caption_grounding", row["answer"].strip(), image)

    with (out / "review_manifest.jsonl").open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(
        "# Updated quality review\n\n"
        "Applies the first review decisions: detector data removed; grid titles removed; anime quality boilerplate removed; "
        "Joy records limited to images locally matched by filehash; Manga keeps title/tags only.\n"
    )
    print(f"wrote {len(output)} labeled review items")


if __name__ == "__main__":
    main()
