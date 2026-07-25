#!/usr/bin/env python3
"""Materialize training manifests after the Qwen3.6 recaption queue is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_recaption_civitai_joy_qwen36 import SOURCE_NAMES, stable_job_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(
    "/thearray/downloads/cache/moe-mla/qwen36_civitai_joy_recaption/recaptions.jsonl")


def load_results(path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("caption"):
                results[str(row["job_id"])] = row
    return results


def materialize(source: Path, target: Path, results: dict[str, dict]) -> tuple[int, int]:
    temporary = target.with_suffix(target.suffix + ".tmp")
    selected = replaced = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as input_handle, \
            temporary.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            row = json.loads(line)
            stage1_source = str(row.get("stage1_source", ""))
            if stage1_source in SOURCE_NAMES:
                selected += 1
                image = Path(str(row["image"])).expanduser()
                if not image.is_absolute():
                    image = ROOT / image
                job_id = stable_job_id(stage1_source, image, str(row.get("text", "")).strip())
                result = results.get(job_id)
                if result is None:
                    temporary.unlink(missing_ok=True)
                    raise SystemExit(
                        f"missing successful Qwen3.6 caption for {source}:{line_number} "
                        f"({stage1_source}, {image})")
                row["original_text"] = row["text"]
                row["text"] = str(result["caption"]).strip()
                row["caption_policy"] = (
                    "Qwen3.6-35B-A3B-heretic visual recaption with existing text as untrusted hint")
                row["caption_teacher"] = result.get("model")
                row["caption_prompt_version"] = result.get("prompt_version")
                row["recaption_job_id"] = job_id
                replaced += 1
            output_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(target)
    return selected, replaced


def missing_results(source: Path, results: dict[str, dict]) -> list[tuple[int, str, Path]]:
    missing: list[tuple[int, str, Path]] = []
    with source.open(encoding="utf-8") as input_handle:
        for line_number, line in enumerate(input_handle, 1):
            row = json.loads(line)
            stage1_source = str(row.get("stage1_source", ""))
            if stage1_source not in SOURCE_NAMES:
                continue
            image = Path(str(row["image"])).expanduser()
            if not image.is_absolute():
                image = ROOT / image
            job_id = stable_job_id(stage1_source, image, str(row.get("text", "")).strip())
            if job_id not in results:
                missing.append((line_number, stage1_source, image))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--train-in", type=Path,
        default=ROOT / "curated_vision/captioning_first_civitai_joy_train.jsonl")
    parser.add_argument(
        "--eval-in", type=Path,
        default=ROOT / "curated_vision/captioning_first_civitai_joy_eval.jsonl")
    parser.add_argument(
        "--train-out", type=Path,
        default=ROOT / "curated_vision/captioning_first_civitai_joy_qwen36_train.jsonl")
    parser.add_argument(
        "--eval-out", type=Path,
        default=ROOT / "curated_vision/captioning_first_civitai_joy_qwen36_eval.jsonl")
    args = parser.parse_args()
    if not args.results.is_file():
        raise SystemExit(f"missing recaption journal: {args.results}")
    results = load_results(args.results)
    missing = []
    for source in (args.train_in, args.eval_in):
        missing.extend((source, *entry) for entry in missing_results(source, results))
    if missing:
        preview = "\n".join(
            f"  {source}:{line_number} ({stage1_source}, {image})"
            for source, line_number, stage1_source, image in missing[:10])
        raise SystemExit(
            f"refusing to materialize an incomplete teacher set: "
            f"{len(missing)} captions are missing\n{preview}")
    summaries = {}
    for split, source, target in (
        ("train", args.train_in, args.train_out),
        ("eval", args.eval_in, args.eval_out),
    ):
        selected, replaced = materialize(source, target, results)
        summaries[split] = {
            "input": str(source), "output": str(target),
            "selected": selected, "replaced": replaced,
        }
    print(json.dumps({"results": str(args.results), "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()
