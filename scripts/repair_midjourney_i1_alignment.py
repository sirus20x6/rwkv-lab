#!/usr/bin/env python3
"""Repair Midjourney/i1 sibling suffixes without changing physical image paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from midjourney_alignment import (ALIGNMENT_SCHEMA, I1_CAPTION_COLUMNS,
                                  SOURCE_CAPTION_COLUMNS, align_group)


GENERATIONISM = re.compile(
    r"(?ix)(?:\bmasterpiece\b|\bbest\s+quality\b|\bamazing\s+quality\b|"
    r"\bhighest\s+quality\b|\babsurd\s*res\b|\bhigh\s*res\b|\bhi-res\b|"
    r"\bvery\s+aesthetic\b|\baward[- ]winning\b|\buhd\b|"
    r"\b(?:high|very\s+high|low|poor)\s+quality\b|\bhighly\s+detailed\b|"
    r"\b(?:4k|8k)\s*(?:resolution)?\b|<\s*(?:lora|lyco|embedding):|\blora:|"
    r"\b(?:positive|negative)\s+prompt\s*:|<segment:|\f)"
)
LOW_CONFIDENCE_MARGIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path,
                        default=ROOT / "i1-captions/midjourneyv6")
    parser.add_argument("--initial-shard", type=Path, default=ROOT /
                        "datasets/i1_source_shards/midjourneyv6/train_000.parquet")
    parser.add_argument("--selection", type=Path, default=ROOT /
                        "datasets/i1_eight_hour_work/midjourneyv6_selection.jsonl")
    parser.add_argument("--selection-contract", type=Path, default=ROOT /
                        "datasets/i1_eight_hour_work/midjourneyv6_selection.json")
    parser.add_argument("--receipt", type=Path, default=ROOT /
                        "curated_vision/midjourney_alignment_repair.json")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".alignment.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".alignment.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_groups(shards: list[Path]) -> tuple[
        list[tuple[Path, str, list[tuple[int, list[str]]]]], set[str]]:
    groups = []
    wanted = set()
    for shard_number, path in enumerate(shards, 1):
        parquet = pq.ParquetFile(path)
        grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
        row_index = 0
        for batch in parquet.iter_batches(
                columns=["id", *SOURCE_CAPTION_COLUMNS], batch_size=16_384):
            for row in batch.to_pylist():
                numeric_id = str(int(row["id"]))
                grouped[numeric_id].append((
                    row_index,
                    [str(row.get(name) or "") for name in SOURCE_CAPTION_COLUMNS],
                ))
                row_index += 1
        malformed = [key for key, rows in grouped.items() if len(rows) != 4]
        if malformed:
            raise RuntimeError(f"{path} has {len(malformed)} non-four-image groups")
        for numeric_id, rows in grouped.items():
            groups.append((path.resolve(), numeric_id, rows))
            wanted.update(f"{numeric_id}_{suffix}" for suffix in range(4))
        print(f"source captions {shard_number}/{len(shards)}: {path.name} "
              f"({row_index:,} rows)", flush=True)
    return groups, wanted


def load_i1_captions(directory: Path, wanted: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    options = pc.SetLookupOptions(value_set=pa.array(sorted(wanted), type=pa.string()))
    for path in sorted(directory.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        available = [name for name in I1_CAPTION_COLUMNS
                     if name in parquet.schema_arrow.names]
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["key", *available])
            table = table.filter(pc.is_in(table["key"], options=options))
            for row in table.to_pylist():
                result[str(row["key"])] = [
                    str(row.get(name) or "").strip() for name in I1_CAPTION_COLUMNS]
        print(f"i1 captions: {len(result):,}/{len(wanted):,} after {path.name}",
              flush=True)
    missing = wanted - result.keys()
    if missing:
        print(f"i1 captions: inferring {len(missing)} absent suffixes from their "
              "three captioned siblings", flush=True)
        result.update({key: [] for key in missing})
    return result


def build_mapping(groups, captions):
    by_source_row = {}
    by_physical_key = {}
    source_caption_fallback_rows = 0
    margins = []
    row_scores = []
    for path, numeric_id, rows in groups:
        aligned = align_group(
            [values for _, values in rows],
            [captions[f"{numeric_id}_{suffix}"] for suffix in range(4)],
        )
        margins.append(aligned.margin)
        row_scores.extend(aligned.row_scores)
        for source_offset, (_, source_values) in enumerate(rows):
            canonical_key = f"{numeric_id}_{aligned.row_to_suffix[source_offset]}"
            if not captions[canonical_key]:
                # Preserve an already-cached physical image even when i1 omitted
                # its row. These are exact captions from that source image.
                captions[canonical_key] = [
                    text.strip() for text in source_values if text.strip()]
                source_caption_fallback_rows += 1
        for source_offset, (row_index, _) in enumerate(rows):
            physical_key = f"{numeric_id}_{source_offset}"
            canonical_key = f"{numeric_id}_{aligned.row_to_suffix[source_offset]}"
            item = {
                "physical_key": physical_key,
                "canonical_key": canonical_key,
                "source_file": str(path),
                "row_index": row_index,
                "alignment_margin": aligned.margin,
                "alignment_score": aligned.row_scores[source_offset],
                "alignment_low_confidence": aligned.margin < LOW_CONFIDENCE_MARGIN,
                "source_captions": list(rows[source_offset][1]),
            }
            by_source_row[(str(path), row_index)] = item
            if physical_key in by_physical_key:
                raise RuntimeError(f"duplicate physical Midjourney key {physical_key}")
            by_physical_key[physical_key] = item
    stats = {
        "groups": len(groups),
        "physical_rows": len(by_physical_key),
        "identity_rows": sum(item["physical_key"] == item["canonical_key"]
                             for item in by_physical_key.values()),
        "changed_rows": sum(item["physical_key"] != item["canonical_key"]
                            for item in by_physical_key.values()),
        "minimum_assignment_margin": min(margins),
        "mean_assignment_margin": sum(margins) / len(margins),
        "zero_margin_groups": sum(margin <= 1e-12 for margin in margins),
        "margin_below_0_01_groups": sum(
            margin < LOW_CONFIDENCE_MARGIN for margin in margins),
        "minimum_matched_row_score": min(row_scores),
        "mean_matched_row_score": sum(row_scores) / len(row_scores),
        "source_caption_fallback_rows": source_caption_fallback_rows,
    }
    return by_source_row, by_physical_key, stats


def strip_generationisms(text: str) -> str:
    """Remove prompt-quality boilerplate without discarding visual facts."""
    text = GENERATIONISM.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(
        r"\b(This|That|It|The|A|An)\s*(?:,|\band\b)\s+",
        lambda match: match.group(1) + " ", text)
    text = re.sub(r"^\s*[,;:]\s*|\s*[,;:]\s*$", "", text)
    return text.strip()


def clean_caption_options(values: list[str]) -> list[tuple[int, str]]:
    result = []
    for index, text in enumerate(values):
        cleaned = strip_generationisms(text)
        if cleaned:
            result.append((index, cleaned))
    return result


def stable_index(seed: int, key: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def selected_caption(values: list[str], key: str, seed: int) -> tuple[str, int]:
    clean = clean_caption_options(values)
    if not clean:
        raise RuntimeError(f"canonical key {key} has no clean captions")
    index = stable_index(seed, key, len(clean))
    column, text = clean[index]
    return text, column + 1


def aligned_caption(item: dict, captions: dict[str, list[str]],
                    seed: int) -> tuple[str, int, str]:
    if item["alignment_low_confidence"]:
        text, variant = selected_caption(
            item["source_captions"], item["physical_key"], seed)
        return text, variant, "source_exact_low_alignment_confidence"
    text, variant = selected_caption(
        captions[item["canonical_key"]], item["canonical_key"], seed)
    return text, variant, "i1_aligned"


def corresponding_caption(old_text: str, old_values: list[str],
                          new_values: list[str], key: str) -> tuple[str, int]:
    old_text = old_text.strip()
    old_index = next((index for index, text in enumerate(old_values)
                      if text.strip() == old_text), None)
    if old_index is not None and new_values[old_index]:
        cleaned = strip_generationisms(new_values[old_index])
        if cleaned:
            return cleaned, old_index + 1
    return selected_caption(new_values, key, 20260714)


def physical_key(row: dict) -> str:
    if row.get("source_row_key"):
        return str(row["source_row_key"])
    return Path(str(row["image"])).stem


def is_midjourney(row: dict) -> bool:
    return (row.get("i1_subset") == "midjourneyv6"
            or "midjourney" in str(row.get("stage1_source", "")))


def repair_row(row: dict, mapping: dict[str, dict], captions: dict[str, list[str]],
               eight_hour_selection: dict[str, dict]) -> tuple[dict, bool]:
    if not is_midjourney(row):
        return row, False
    source_key = physical_key(row)
    item = mapping.get(source_key)
    if item is None:
        raise RuntimeError(f"no alignment mapping for {source_key} in {row.get('image')}")
    canonical_key = item["canonical_key"]
    if str(row.get("stage1_source", "")).startswith("eight_hour_"):
        selected = eight_hour_selection[source_key]
        text, variant = selected["text"], int(selected["caption_variant"])
        caption_provenance = selected["caption_provenance"]
    elif item["alignment_low_confidence"]:
        text, variant, caption_provenance = aligned_caption(
            item, captions, 20260714)
    elif str(row.get("stage1_source", "")) == "i1_midjourneyv6":
        text, variant = selected_caption(captions[canonical_key], canonical_key, 20260714)
        caption_provenance = "i1_aligned"
    else:
        text, variant = corresponding_caption(
            str(row.get("text", "")), captions[source_key],
            captions[canonical_key], canonical_key)
        caption_provenance = "i1_aligned"
    repaired = dict(row)
    repaired.update({
        "i1_key": canonical_key,
        "text": text,
        "source_row_key": source_key,
        "alignment_schema": ALIGNMENT_SCHEMA,
        "alignment_margin": item["alignment_margin"],
        "alignment_low_confidence": item["alignment_low_confidence"],
        "caption_provenance": caption_provenance,
    })
    if "caption_variant" in repaired:
        repaired["caption_variant"] = variant
    changed = (row.get("i1_key") != canonical_key or row.get("text") != text
               or row.get("alignment_schema") != ALIGNMENT_SCHEMA)
    return repaired, changed


def update_summary_hashes() -> None:
    specs = {
        ROOT / "curated_vision/vision_next_i1_25pct.summary.json": {
            "output_sha256": ROOT / "curated_vision/vision_next_i1_25pct.jsonl"},
        ROOT / "curated_vision/vision_next_i1_civitai.summary.json": {
            "train_sha256": ROOT / "curated_vision/vision_next_i1_civitai.jsonl",
            "eval_sha256": ROOT / "curated_vision/vision_next_i1_civitai_eval.jsonl"},
        ROOT / "curated_vision/vision_finish_grounded.summary.json": {
            "train_sha256": ROOT / "curated_vision/vision_finish_grounded.jsonl",
            "eval_sha256": ROOT / "curated_vision/vision_finish_grounded_eval.jsonl"},
        ROOT / "curated_vision/vision_eight_hour.summary.json": {
            "train_sha256": ROOT / "curated_vision/vision_eight_hour.jsonl",
            "eval_sha256": ROOT / "curated_vision/vision_eight_hour_eval.jsonl"},
    }
    for summary_path, fields in specs.items():
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for field, manifest in fields.items():
            if manifest.is_file():
                summary[field] = sha256(manifest)
        summary["midjourney_alignment_schema"] = ALIGNMENT_SCHEMA
        atomic_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    contract = json.loads(args.selection_contract.read_text())
    selected_shards = [Path(path) for path in contract["shards"]]
    shards = [args.initial_shard, *selected_shards]
    groups, wanted = source_groups(shards)
    captions = load_i1_captions(args.captions, wanted)
    by_source_row, mapping, stats = build_mapping(groups, captions)
    print(json.dumps(stats, indent=2), flush=True)

    selection = read_jsonl(args.selection)
    repaired_selection = []
    selection_by_physical = {}
    for row in selection:
        lookup = (str(Path(row["source_file"]).resolve()), int(row["row_index"]))
        item = by_source_row.get(lookup)
        if item is None:
            raise RuntimeError(f"selection source row is absent from alignment map: {lookup}")
        canonical_key = item["canonical_key"]
        text, variant, caption_provenance = aligned_caption(
            item, captions, 20260715)
        repaired = dict(row)
        repaired.update({
            "i1_key": canonical_key,
            "text": text,
            "caption_variant": variant,
            "source_row_key": item["physical_key"],
            "alignment_schema": ALIGNMENT_SCHEMA,
            "alignment_margin": item["alignment_margin"],
            "alignment_low_confidence": item["alignment_low_confidence"],
            "caption_provenance": caption_provenance,
        })
        repaired_selection.append(repaired)
        selection_by_physical[item["physical_key"]] = repaired
    if len(selection_by_physical) != len(selection):
        raise RuntimeError("repaired selection contains duplicate physical keys")

    manifests = sorted((ROOT / "curated_vision").glob("*.jsonl"))
    plans = []
    for manifest in manifests:
        rows = read_jsonl(manifest)
        affected = sum(is_midjourney(row) for row in rows)
        if not affected:
            continue
        repaired_rows = []
        changed = 0
        for row in rows:
            repaired, did_change = repair_row(
                row, mapping, captions, selection_by_physical)
            repaired_rows.append(repaired)
            changed += did_change
        plans.append((manifest, repaired_rows, affected, changed, sha256(manifest)))
        print(f"repair plan {manifest.name}: affected={affected:,} changed={changed:,}",
              flush=True)

    receipt = {
        "schema": 1,
        "alignment_schema": ALIGNMENT_SCHEMA,
        "applied": bool(args.apply),
        "mapping": stats,
        "selection_rows": len(repaired_selection),
        "manifests": [{
            "path": str(path.relative_to(ROOT)),
            "affected_rows": affected,
            "changed_rows": changed,
            "before_sha256": before,
        } for path, _, affected, changed, before in plans],
    }
    if not args.apply:
        print("dry run complete; pass --apply to publish repairs", flush=True)
        atomic_json(args.receipt.with_suffix(".dry_run.json"), receipt)
        return

    atomic_jsonl(args.selection, repaired_selection)
    contract.update({"schema": 2, "alignment_schema": ALIGNMENT_SCHEMA})
    atomic_json(args.selection_contract, contract)
    for path, rows, _, _, _ in plans:
        atomic_jsonl(path, rows)
    update_summary_hashes()
    for manifest in receipt["manifests"]:
        path = ROOT / manifest["path"]
        manifest["after_sha256"] = sha256(path)
    receipt["selection_sha256"] = sha256(args.selection)
    atomic_json(args.receipt, receipt)
    print(f"published repair receipt: {args.receipt}", flush=True)


if __name__ == "__main__":
    main()
