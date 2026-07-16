#!/usr/bin/env python3
"""Overlay normalized pose-dataset bounding boxes onto updated review images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#00e5ff", "#ff3d71", "#a3ff12", "#ffb000", "#c77dff")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=ROOT / "quality_review_updated")
    args = parser.parse_args()
    pack = args.pack.resolve()
    manifest_path = pack / "review_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.open() if line.strip()]
    font = ImageFont.load_default()
    changed = 0
    for row in rows:
        if row.get("source_dataset") != "pose_vrlens_nsfw":
            continue
        image_path = pack / row["image"]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        objects = json.loads(row["reference"])
        for category, bbox in zip(objects["categories"], objects["bbox"]):
            x, y, box_width, box_height = (float(value) for value in bbox)
            left, top = x * width, y * height
            right, bottom = (x + box_width) * width, (y + box_height) * height
            color = COLORS[int(category) % len(COLORS)]
            line_width = max(2, round(min(width, height) / 300))
            draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
            label = f"category {category}"
            label_box = draw.textbbox((left, top), label, font=font)
            draw.rectangle(label_box, fill=color)
            draw.text((left, top), label, fill="black", font=font)
        overlay = image_path.with_name(image_path.stem + "_boxes.jpg")
        image.save(overlay, quality=95)
        row["image"] = str(overlay.relative_to(pack))
        row["annotation_overlay"] = "bounding_boxes"
        changed += 1
    with manifest_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"annotated {changed} pose review images")


if __name__ == "__main__":
    main()
