#!/usr/bin/env python3
"""Create non-destructive, cleaned manifests from the downloaded vision sets."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
QUALITY_TOKENS = {"masterpiece", "best quality", "amazing", "highres", "absurdres", "very aesthetic", "aesthetic"}


def clean_anime_caption(text: str) -> str:
    """Remove prompt-quality boilerplate while retaining visual tags."""
    tags = [tag.strip() for tag in text.replace("\n", " ").split(",")]
    return ", ".join(tag for tag in tags if tag and tag.casefold() not in QUALITY_TOKENS)


def labeled_text(**fields: object) -> str:
    """Render only populated source fields; never emit placeholders like ``Cast:``."""
    def present(value: object) -> bool:
        return value is not None and str(value).strip().casefold() not in ("", "null", "none")
    return "\n".join(f"{label.title()}: {value}" for label, value in fields.items() if present(value))


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def grid_rows():
    root = ROOT / "nsfw-video-still-caption-grid-only/extracted"
    for metadata_path in sorted(root.glob("*/*.json")):
        metadata = json.loads(metadata_path.read_text())
        image = metadata_path.with_suffix(".jpg")
        if not image.exists():
            continue
        # Titles are intentionally excluded: user audit found them unreliable.
        yield {
            "image": str(image.relative_to(ROOT)),
            "tags": metadata.get("tags"),
            "categories": metadata.get("categories"),
            "cast": metadata.get("cast"),
            "text": labeled_text(tags=metadata.get("tags"), categories=metadata.get("categories"), cast=metadata.get("cast")),
        }


def anime_rows():
    root = ROOT / "nsfw_scene_animes_31-03/extracted"
    for caption in sorted(root.glob("*_B.txt")):
        image = caption.with_suffix(".jpg")
        if image.exists():
            yield {"image": str(image.relative_to(ROOT)), "text": clean_anime_caption(caption.read_text().strip())}


def joy_rows():
    source = ROOT / "joy-captioning-20250408a/data"
    images = {path.stem: path for path in (ROOT / "joy-captioning-20250408a/images_direct").glob("*") if path.is_file()}
    for parquet in sorted(source.glob("*.parquet")):
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["filehash", "answer", "question_type", "is_human", "urls"], batch_size=4096):
            for row in batch.to_pylist():
                image = images.get(row["filehash"].hex())
                if image is None:
                    continue  # Explicitly exclude unpaired descriptions.
                yield {
                    "image": str(image.relative_to(ROOT)),
                    "text": row["answer"].strip(),  # no Prompt/Answer wrapper
                    "question_type": row["question_type"],
                    "is_human": row["is_human"],
                    "source_urls": row["urls"],
                }


def manga_rows():
    source = ROOT / "NSFW_Manga/manga_all.parquet"
    for batch in pq.ParquetFile(source).iter_batches(columns=["id", "title", "tags", "cover_image"], batch_size=4096):
        for row in batch.to_pylist():
            yield {"id": row["id"], "title": row["title"], "tags": row["tags"], "cover_image": row["cover_image"]}


def main() -> None:
    out = ROOT / "curated_vision"
    out.mkdir(exist_ok=True)
    counts = {
        "grid": write_jsonl(out / "grid_caption_no_titles.jsonl", grid_rows()),
        "anime": write_jsonl(out / "anime_b_cleaned.jsonl", anime_rows()),
        "joy_matched": write_jsonl(out / "joy_matched_cleaned.jsonl", joy_rows()),
        "manga_metadata": write_jsonl(out / "manga_title_tags.jsonl", manga_rows()),
    }
    (out / "README.md").write_text(
        "# Curated vision manifests\n\n"
        "Derived files only; raw datasets are unchanged. `grid_caption_no_titles.jsonl` deliberately omits titles. "
        "`anime_b_cleaned.jsonl` removes quality-boilerplate tags. `joy_matched_cleaned.jsonl` contains only rows with "
        "an image presently stored in `joy-captioning-20250408a/images_direct/`; its text is the raw answer with no prompt/header. "
        "`manga_title_tags.jsonl` omits categories and language, and does not claim image availability.\n"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
