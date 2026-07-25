#!/usr/bin/env python3
"""Build a grounded, captioning-first i1 mixture from locally available images.

Unlike i1's text-to-image recipe, this builder treats multiple synthetic
captions as noisy observations of the pixels.  It chooses the least speculative
variant, removes aesthetic/narrative tail sentences, uses ImageNet's full-image
``no_center_crop`` target, and keeps OCR under an explicit transcription prompt.

The default mix contains 100k unique caption/OCR images plus 5k COCO and 5k
LVIS open-vocabulary instance-segmentation tasks. These tasks teach RWKV to
decode the SAM 3 knowledge already distilled into C-RADIOv4; they do not add a
second image encoder. The mix is sized for roughly one eight-hour RADIO run
at the measured trainer throughput. Every source is materialized to normal
files and committed independently, so long archive scans are resumable at
source boundaries.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from build_vision_eight_hour_mix import (  # noqa: E402
    CAPTION_COLUMNS,
    GENERATIONISM,
    cleaned_caption_text,
    materialize_image,
    strip_generationisms,
)


CAPTION_PROMPT = "Describe this image accurately and only state visible details:\n"
OCR_PROMPT = ("Transcribe all visible text in this document image. "
              "Preserve reading order and line breaks:\n")
CONCEPT_SAM_PROMPT = (
    'Segment every visible instance matching the concept "{concept}". '
    "Return one line per instance with normalized box [x1,y1,x2,y2] and "
    "16x16 mask row spans. Return `none` when the concept is absent:\n")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
INFERENCE = re.compile(
    r"(?i)\b(?:likely|possibly|perhaps|seems?|suggest(?:s|ing)?|"
    r"may be|could be|appears? to be|reminiscent of)\b")
SUBJECTIVE = re.compile(
    r"(?i)\b(?:beautiful(?:ly)?|stunning|striking|vibrant|serene|cozy|"
    r"dramatic|elegant(?:ly)?|captivating|timeless beauty|visually appealing|"
    r"inviting|delectable)\b")
AESTHETIC_TAIL = re.compile(
    r"(?i)(?:\bthe overall (?:composition|image|scene|effect|mood)\b|"
    r"\b(?:creates?|conveys?|evokes?|adds?|gives?|imbues?) (?:the image |the scene )?"
    r"(?:with )?(?:a |an )?(?:sense|feeling|mood|atmosphere)\b|"
    r"\bdraws? the (?:viewer|eye)\b|\binvites? the viewer\b|"
    r"\bmaking (?:the image|the scene|it) (?:feel|appear)\b|"
    r"\bthe image exudes\b|\bthe scene exudes\b)"
)
HIDDEN_RENDER_METADATA = re.compile(
    r"(?i)\b(?:\d{2,4}\s*[x×]\s*\d{2,4}\s*pixels?|font[_ ]size|"
    r"font[_ ]type|RGB\s*[:\[]|rotation[_ ]degree)\b")
WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "with",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--pexels", type=int, default=25_000)
    ap.add_argument("--imagenet22k", type=int, default=30_000)
    ap.add_argument("--places365", type=int, default=20_000)
    ap.add_argument("--inaturalist", type=int, default=12_000)
    ap.add_argument("--doclingmatix", type=int, default=4_445)
    ap.add_argument("--fluxreason", type=int, default=5_000)
    ap.add_argument("--midjourneyv6", type=int, default=3_555)
    ap.add_argument("--coco-sam", type=int, default=5_000)
    ap.add_argument("--lvis-sam", type=int, default=5_000)
    ap.add_argument("--eval-per-source", type=int, default=64)
    ap.add_argument("--flux-shards", type=int, default=32)
    ap.add_argument("--captions", type=Path, default=ROOT / "i1-captions")
    ap.add_argument("--source-root", type=Path,
                    default=ROOT / "datasets/i1_full_sources")
    ap.add_argument("--legacy-selection", type=Path,
                    default=ROOT / "datasets/i1_eight_hour_work")
    ap.add_argument("--legacy-eval", type=Path,
                    default=ROOT / "curated_vision/vision_eight_hour_eval.jsonl")
    ap.add_argument("--ocr-train", type=Path,
                    default=ROOT / "curated_vision/vision_next_ocr10_train.jsonl")
    ap.add_argument("--ocr-eval", type=Path,
                    default=ROOT / "curated_vision/vision_next_ocr10_eval.jsonl")
    ap.add_argument("--coco-root", type=Path,
                    default=ROOT / "datasets/coco2017")
    ap.add_argument("--image-dir", type=Path,
                    default=ROOT / "datasets/captioning_first_images")
    ap.add_argument("--work-dir", type=Path,
                    default=ROOT / "datasets/captioning_first_work")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "curated_vision/captioning_first_train.jsonl")
    ap.add_argument("--eval-output", type=Path,
                    default=ROOT / "curated_vision/captioning_first_eval.jsonl")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def stable_rank(seed: int, namespace: str, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{key}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def select_smallest(records: Iterable[dict], count: int, *, seed: int,
                    namespace: str, key_field: str = "i1_key") -> list[dict]:
    """Select the globally smallest deterministic hashes in bounded memory."""
    if count < 1:
        return []
    heap: list[tuple[int, str, dict]] = []
    seen: set[str] = set()
    for row in records:
        key = str(row[key_field])
        if key in seen:
            continue
        seen.add(key)
        rank = stable_rank(seed, namespace, key)
        entry = (-rank, key, row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(
        heap, key=lambda entry: (-entry[0], entry[1]))]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()]


def clean_grounded_caption(text: str) -> str:
    text = cleaned_caption_text(strip_generationisms(str(text)))
    kept = []
    for sentence in _sentences(text):
        if AESTHETIC_TAIL.search(sentence):
            continue
        if HIDDEN_RENDER_METADATA.search(sentence):
            continue
        kept.append(sentence)
    cleaned = " ".join(kept).strip()
    return cleaned or text.strip()


def _content_tokens(text: str) -> set[str]:
    return {word.casefold() for word in WORD.findall(text)
            if word.casefold() not in STOPWORDS and len(word) > 2}


def choose_grounded_caption(values: list[tuple[str, str]]) -> tuple[str, str]:
    """Choose the least speculative caption with strongest variant consensus."""
    candidates = [(name, clean_grounded_caption(value))
                  for name, value in values if str(value).strip()]
    if not candidates:
        raise ValueError("no non-empty caption variants")
    token_sets = [_content_tokens(text) for _, text in candidates]
    frequency = Counter(token for tokens in token_sets for token in tokens)
    consensus = {token for token, occurrences in frequency.items()
                 if occurrences >= min(2, len(candidates))}

    def score(item: tuple[int, tuple[str, str]]) -> tuple[float, int, str]:
        index, (name, text) = item
        tokens = token_sets[index]
        coverage = len(tokens & consensus) / max(1, len(tokens))
        speculation = len(INFERENCE.findall(text))
        subjective = len(SUBJECTIVE.findall(text))
        words = len(text.split())
        value = coverage - 0.025 * speculation - 0.012 * subjective
        value -= abs(words - 145) / 10_000
        return value, words, name

    _score, _words, name = max(map(score, enumerate(candidates)))
    text = next(text for candidate_name, text in candidates
                if candidate_name == name)
    return text, name


def load_matching_captions(directory: Path, wanted: set[str],
                           columns: tuple[str, ...]) -> dict[str, dict]:
    matches: dict[str, dict] = {}
    lookup = pc.SetLookupOptions(
        value_set=pa.array(sorted(wanted), type=pa.string()))
    for path in sorted(directory.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        available = tuple(name for name in columns
                          if name in parquet.schema_arrow.names)
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["key", *available])
            table = table.filter(pc.is_in(table["key"], options=lookup))
            for row in table.to_pylist():
                matches[str(row["key"])] = row
        print({"phase": "captions", "source": directory.name,
               "matched": len(matches), "wanted": len(wanted),
               "file": path.name}, flush=True)
        if len(matches) == len(wanted):
            break
    return matches


def select_from_caption_table(directory: Path, count: int, *, seed: int,
                              namespace: str, column: str = "caption1") -> list[dict]:
    def records() -> Iterator[dict]:
        done = 0
        for path in sorted(directory.glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            if column not in parquet.schema_arrow.names:
                raise RuntimeError(f"{path} has no {column}")
            for batch in parquet.iter_batches(columns=["key", column],
                                              batch_size=16_384):
                for row in batch.to_pylist():
                    if row.get(column):
                        yield {"i1_key": str(row["key"]), column: row[column]}
                done += len(batch)
                if done % 500_000 < len(batch):
                    print({"phase": "select_captions", "source": directory.name,
                           "done": done}, flush=True)
    return select_smallest(records(), count, seed=seed, namespace=namespace)


def caption_for(source: str, row: dict) -> tuple[str, str]:
    if source == "imagenet22k":
        value = str(row.get("no_center_crop") or "").strip()
        if not value:
            raise ValueError("ImageNet row has no full-image caption")
        return clean_grounded_caption(value), "no_center_crop"
    values = [(name, str(row.get(name) or "")) for name in CAPTION_COLUMNS]
    return choose_grounded_caption(values)


def final_row(source: str, key: str, payload: bytes, caption: str,
              variant: str, image_dir: Path) -> dict:
    image, digest, width, height = materialize_image(
        payload, image_dir, source, key)
    return {
        "image": image,
        "text": caption,
        "prompt": CAPTION_PROMPT,
        "stage1_source": f"captioning_i1_{source}",
        "source": source,
        "task": "caption",
        "i1_subset": source,
        "i1_key": key,
        "caption_variant": variant,
        "caption_policy": "grounded_consensus_selection_v1",
        "image_sha256": digest,
        "width": width,
        "height": height,
    }


def valid_part(path: Path, expected: int) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if len(rows) != expected:
        return None
    if any(not rooted(row["image"]).is_file() for row in rows):
        return None
    digests = [str(row.get("image_sha256") or "") for row in rows]
    if any(not digest for digest in digests) or len(set(digests)) != len(digests):
        return None
    return rows


def _selected_receipt(path: Path, count: int, *, seed: int,
                      namespace: str) -> list[dict]:
    return select_smallest(
        (json.loads(line) for line in path.open() if line.strip()), count,
        seed=seed, namespace=namespace)


def build_pexels(args: argparse.Namespace, count: int) -> list[dict]:
    part = args.work_dir / "pexels.jsonl"
    if (rows := valid_part(part, count)) is not None:
        return rows
    extra = max(512, count // 50)
    reserve = count + extra
    candidates = _selected_receipt(
        args.legacy_selection / "pexels_selection.jsonl", reserve,
        seed=args.seed, namespace="pexels")
    captions = load_matching_captions(
        args.captions / "pexels", {str(row["i1_key"]) for row in candidates},
        CAPTION_COLUMNS)
    selected = []
    for candidate in candidates:
        key = str(candidate["i1_key"])
        if key not in captions:
            continue
        text, variant = caption_for("pexels", captions[key])
        selected.append({**candidate, "caption": text, "variant": variant})
        if len(selected) == reserve:
            break
    grouped: dict[Path, dict[str, dict]] = defaultdict(dict)
    for row in selected:
        grouped[Path(row["source_file"])][str(row["member"])] = row
    output = []
    for index, (source_file, wanted) in enumerate(sorted(grouped.items()), 1):
        with tarfile.open(source_file) as archive:
            for member in archive:
                row = wanted.get(member.name)
                if row is None:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    output.append(final_row(
                        "pexels", str(row["i1_key"]), handle.read(),
                        row["caption"], row["variant"], args.image_dir))
        print({"phase": "materialize", "source": "pexels", "shard": index,
               "shards": len(grouped), "done": len(output)}, flush=True)
    output.sort(key=lambda row: stable_rank(
        args.seed, "pexels", str(row["i1_key"])))
    output = unique_content_rows(output, count)
    if len(output) != count:
        raise RuntimeError(f"Pexels materialized {len(output)}/{count} unique images")
    write_jsonl(part, output)
    return output


def build_midjourney(args: argparse.Namespace, count: int) -> list[dict]:
    part = args.work_dir / "midjourneyv6.jsonl"
    if (rows := valid_part(part, count)) is not None:
        return rows
    extra = max(256, count // 20)
    reserve = count + extra
    candidates = _selected_receipt(
        args.legacy_selection / "midjourneyv6_selection.jsonl", reserve,
        seed=args.seed, namespace="midjourneyv6")
    captions = load_matching_captions(
        args.captions / "midjourneyv6",
        {str(row["i1_key"]) for row in candidates}, CAPTION_COLUMNS)
    selected = []
    for candidate in candidates:
        key = str(candidate["i1_key"])
        if key not in captions:
            continue
        text, variant = caption_for("midjourneyv6", captions[key])
        selected.append({**candidate, "caption": text, "variant": variant})
        if len(selected) == reserve:
            break
    grouped: dict[Path, dict[int, dict]] = defaultdict(dict)
    for row in selected:
        grouped[Path(row["source_file"])][int(row["row_index"])] = row
    output = []
    for index, (source_file, wanted) in enumerate(sorted(grouped.items()), 1):
        offset = 0
        for batch in pq.ParquetFile(source_file).iter_batches(
                columns=["image"], batch_size=1_024):
            for local, value in enumerate(batch.column(0).to_pylist()):
                row = wanted.get(offset + local)
                if row is not None and value and value.get("bytes"):
                    output.append(final_row(
                        "midjourneyv6", str(row["i1_key"]), value["bytes"],
                        row["caption"], row["variant"], args.image_dir))
            offset += len(batch)
        print({"phase": "materialize", "source": "midjourneyv6",
               "shard": index, "shards": len(grouped), "done": len(output)},
              flush=True)
    output.sort(key=lambda row: stable_rank(
        args.seed, "midjourneyv6", str(row["i1_key"])))
    output = unique_content_rows(output, count)
    if len(output) != count:
        raise RuntimeError(
            f"Midjourney materialized {len(output)}/{count} unique images")
    write_jsonl(part, output)
    return output


def tar_candidates(files: list[Path], *, key_fn: Callable[[str], str | None]
                   ) -> Iterator[dict]:
    for index, source_file in enumerate(files, 1):
        count = 0
        with tarfile.open(source_file) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key = key_fn(member.name)
                if key:
                    count += 1
                    yield {"i1_key": key, "source_file": str(source_file),
                           "member": member.name}
        print({"phase": "index_tar", "file": source_file.name,
               "index": index, "files": len(files), "images": count}, flush=True)


def stem_key(name: str) -> str | None:
    path = Path(name)
    return path.stem if path.suffix.lower() in IMAGE_SUFFIXES else None


def unique_content_rows(rows: Iterable[dict], count: int) -> list[dict]:
    output = []
    seen: set[str] = set()
    for row in rows:
        digest = str(row.get("image_sha256") or "")
        if not digest:
            raise RuntimeError("materialized row is missing image_sha256")
        if digest in seen:
            continue
        seen.add(digest)
        output.append(row)
        if len(output) == count:
            break
    return output


@contextlib.contextmanager
def open_archive_stream(archive: Path) -> Iterator[tarfile.TarFile]:
    """Stream large gzip tars through rapidgzip when it is available.

    Python's tarfile gzip reader is single-threaded.  The iNaturalist archive
    is hundreds of gigabytes, so parallel DEFLATE block discovery is worth a
    dedicated subprocess.  Streaming mode also keeps memory bounded and does
    not write the decompressed tar to disk.
    """
    rapidgzip = shutil.which("rapidgzip")
    if archive.name.endswith((".tar.gz", ".tgz")) and rapidgzip:
        workers = max(1, min(24, (os.cpu_count() or 1) // 2))
        process = subprocess.Popen(
            [rapidgzip, "-d", "-P", str(workers), "--no-verify", "-c",
             str(archive)],
            stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        stream = tarfile.open(fileobj=process.stdout, mode="r|")
        completed = False
        try:
            yield stream
            completed = True
        finally:
            stream.close()
            process.stdout.close()
            if process.poll() is None:
                if completed:
                    process.wait()
                else:
                    process.terminate()
                    process.wait()
            # Closing a stream after the final requested member intentionally
            # gives the producer SIGPIPE; that is a successful early exit.
            if completed and process.returncode not in (0, -13, 141):
                raise RuntimeError(
                    f"rapidgzip failed for {archive} with status "
                    f"{process.returncode}")
        return
    with tarfile.open(archive) as stream:
        yield stream


def build_imagenet(args: argparse.Namespace, count: int,
                   eval_count: int) -> tuple[list[dict], list[dict]]:
    total = count + eval_count
    part = args.work_dir / "imagenet22k.jsonl"
    if (rows := valid_part(part, total)) is None:
        files = sorted((args.source_root / "imagenet22k").glob("*.tar"))
        reserve = 2_000
        candidates = select_smallest(
            tar_candidates(files, key_fn=stem_key), total + reserve,
            seed=args.seed, namespace="imagenet22k")
        captions = load_matching_captions(
            args.captions / "imagenet22k",
            {str(row["i1_key"]) for row in candidates}, ("no_center_crop",))
        selected = []
        for candidate in candidates:
            key = str(candidate["i1_key"])
            if key not in captions or not captions[key].get("no_center_crop"):
                continue
            text, variant = caption_for("imagenet22k", captions[key])
            selected.append({**candidate, "caption": text, "variant": variant})
            if len(selected) == total + reserve:
                break
        grouped: dict[Path, dict[str, dict]] = defaultdict(dict)
        for row in selected:
            grouped[Path(row["source_file"])][str(row["member"])] = row
        rows = []
        for index, (source_file, wanted) in enumerate(sorted(grouped.items()), 1):
            with tarfile.open(source_file) as archive:
                for member in archive:
                    row = wanted.get(member.name)
                    if row is None:
                        continue
                    handle = archive.extractfile(member)
                    if handle is not None:
                        rows.append(final_row(
                            "imagenet22k", str(row["i1_key"]), handle.read(),
                            row["caption"], row["variant"], args.image_dir))
            if index % 20 == 0 or index == len(grouped):
                print({"phase": "materialize", "source": "imagenet22k",
                       "shard": index, "shards": len(grouped),
                       "done": len(rows)}, flush=True)
        rows.sort(key=lambda row: stable_rank(
            args.seed, "imagenet22k", str(row["i1_key"])))
        rows = unique_content_rows(rows, total)
        if len(rows) != total:
            raise RuntimeError(
                f"ImageNet materialized only {len(rows)}/{total} unique images")
        write_jsonl(part, rows)
    return split_eval(rows, count, eval_count, args.seed, "imagenet22k")


def places_key(name: str) -> str | None:
    path = Path(name)
    if path.suffix.lower() not in IMAGE_SUFFIXES or len(path.parts) < 2:
        return None
    return f"{path.parent.name}_{path.stem}"


def build_archive_source(args: argparse.Namespace, source: str, count: int,
                         eval_count: int, archive: Path,
                         key_fn: Callable[[str], str | None]) -> tuple[list[dict], list[dict]]:
    total = count + eval_count
    part = args.work_dir / f"{source}.jsonl"
    if (rows := valid_part(part, total)) is None:
        reserve = max(256, total // 50)
        selected_captions = select_from_caption_table(
            args.captions / ("places365-challenge2016"
                             if source == "places365" else source),
            total + reserve, seed=args.seed, namespace=source)
        by_key = {str(row["i1_key"]): row for row in selected_captions}
        rows = []
        scanned = 0
        with open_archive_stream(archive) as handle:
            for member in handle:
                if not member.isfile():
                    continue
                scanned += 1
                key = key_fn(member.name)
                caption_row = by_key.get(str(key)) if key else None
                if caption_row is not None:
                    extracted = handle.extractfile(member)
                    if extracted is not None:
                        text, variant = caption_for(source, caption_row)
                        rows.append(final_row(
                            source, str(key), extracted.read(), text, variant,
                            args.image_dir))
                        del by_key[str(key)]
                        if not by_key:
                            break
                if scanned % 500_000 == 0:
                    print({"phase": "scan_archive", "source": source,
                           "members": scanned, "done": len(rows),
                           "wanted": total}, flush=True)
        rows.sort(key=lambda row: stable_rank(
            args.seed, source, str(row["i1_key"])))
        rows = unique_content_rows(rows, total)
        if len(rows) != total:
            raise RuntimeError(
                f"{source} materialized {len(rows)}/{total} unique images; "
                f"missing keys include {sorted(by_key)[:5]}")
        write_jsonl(part, rows)
    return split_eval(rows, count, eval_count, args.seed, source)


def build_flux(args: argparse.Namespace, count: int,
               eval_count: int) -> tuple[list[dict], list[dict]]:
    total = count + eval_count
    part = args.work_dir / "fluxreason.jsonl"
    if (rows := valid_part(part, total)) is None:
        all_files = sorted((args.source_root / "fluxreason").rglob("*.parquet"))
        files = sorted(all_files, key=lambda path: stable_rank(
            args.seed, "flux_shard", str(path.relative_to(args.source_root))))[
                :args.flux_shards]

        def candidates() -> Iterator[dict]:
            for source_file in files:
                offset = 0
                parquet = pq.ParquetFile(source_file)
                columns = ["id", "score_image_clarity", "score_image_structure"]
                for batch in parquet.iter_batches(columns=columns,
                                                  batch_size=4_096):
                    for local, row in enumerate(batch.to_pylist()):
                        if (int(row.get("score_image_clarity") or 0) >= 8
                                and int(row.get("score_image_structure") or 0) >= 8):
                            yield {"i1_key": str(row["id"]),
                                   "source_file": str(source_file),
                                   "row_index": offset + local}
                    offset += len(batch)
        selected = select_smallest(
            candidates(), total + 512, seed=args.seed, namespace="fluxreason")
        captions = load_matching_captions(
            args.captions / "fluxreason",
            {str(row["i1_key"]) for row in selected}, CAPTION_COLUMNS)
        chosen = []
        for candidate in selected:
            key = str(candidate["i1_key"])
            if key not in captions:
                continue
            text, variant = caption_for("fluxreason", captions[key])
            chosen.append({**candidate, "caption": text, "variant": variant})
            if len(chosen) == total + 512:
                break
        grouped: dict[Path, dict[int, dict]] = defaultdict(dict)
        for row in chosen:
            grouped[Path(row["source_file"])][int(row["row_index"])] = row
        rows = []
        for index, (source_file, wanted) in enumerate(sorted(grouped.items()), 1):
            offset = 0
            for batch in pq.ParquetFile(source_file).iter_batches(
                    columns=["image"], batch_size=512):
                for local, value in enumerate(batch.column(0).to_pylist()):
                    row = wanted.get(offset + local)
                    if row is not None and value and value.get("bytes"):
                        rows.append(final_row(
                            "fluxreason", str(row["i1_key"]), value["bytes"],
                            row["caption"], row["variant"], args.image_dir))
                offset += len(batch)
            print({"phase": "materialize", "source": "fluxreason",
                   "shard": index, "shards": len(grouped), "done": len(rows)},
                  flush=True)
        rows.sort(key=lambda row: stable_rank(
            args.seed, "fluxreason", str(row["i1_key"])))
        rows = unique_content_rows(rows, total)
        if len(rows) != total:
            raise RuntimeError(
                f"FluxReason materialized {len(rows)}/{total} unique images")
        write_jsonl(part, rows)
    return split_eval(rows, count, eval_count, args.seed, "fluxreason")


def _normalized_box(annotation: dict, width: int, height: int) -> list[int]:
    x, y, box_width, box_height = map(float, annotation["bbox"])
    values = (x / width, y / height,
              (x + box_width) / width, (y + box_height) / height)
    return [max(0, min(999, round(value * 999))) for value in values]


def _mask_row_spans(annotation: dict, width: int, height: int,
                    grid: int = 16) -> str:
    canvas = Image.new("1", (grid, grid), 0)
    draw = ImageDraw.Draw(canvas)
    polygons = annotation.get("segmentation")
    if not isinstance(polygons, list):
        return ""
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = [(float(polygon[index]) * grid / width,
                   float(polygon[index + 1]) * grid / height)
                  for index in range(0, len(polygon) - 1, 2)]
        draw.polygon(points, fill=1)
    spans = []
    pixels = canvas.load()
    for y in range(grid):
        x = 0
        while x < grid:
            while x < grid and not pixels[x, y]:
                x += 1
            if x == grid:
                break
            start = x
            while x + 1 < grid and pixels[x + 1, y]:
                x += 1
            spans.append(f"{y}:{start}-{x}")
            x += 1
    return "|".join(spans)


def _coco_sam_target(annotations: list[dict], categories: dict[int, str],
                     width: int, height: int) -> tuple[str, list[int]]:
    lines = []
    annotation_ids = []
    for annotation in sorted(
            annotations, key=lambda item: float(item.get("area") or 0),
            reverse=True):
        spans = _mask_row_spans(annotation, width, height)
        if not spans:
            continue
        label = categories[int(annotation["category_id"])].replace(";", " ")
        box = ",".join(f"{value:03d}" for value in _normalized_box(
            annotation, width, height))
        lines.append(f"{label}; box=[{box}]; mask16={spans}")
        annotation_ids.append(int(annotation["id"]))
        if len(lines) == 3:
            break
    if not lines:
        raise ValueError("COCO image has no polygon instance masks")
    return "\n".join(lines), annotation_ids


def coco_sam_prompt(target: str) -> str:
    """Turn a COCO target into a fully specified, inference-usable request.

    COCO rows retain only the three largest usable annotations. Naming their
    categories and requested counts tells the model exactly which visible
    objects to return; the old phrase "annotated objects" exposed a hidden
    dataset concept that an inference caller could not know.
    """
    counts: Counter[str] = Counter()
    for line in target.splitlines():
        label, separator, _ = line.partition(";")
        label = label.strip()
        if separator and label:
            counts[label] += 1
    if not counts:
        raise ValueError("COCO SAM target has no category-labelled instances")
    requested = ", ".join(
        f"{label}: {count} largest" for label, count in counts.items())
    return (
        f"Segment these requested visible object instances ({requested}). "
        "Return exactly one line per requested instance. For each line, give "
        "its category, normalized box [x1,y1,x2,y2], and 16x16 mask row spans:\n"
    )


def build_coco_sam(args: argparse.Namespace, count: int,
                   eval_count: int) -> tuple[list[dict], list[dict]]:
    total = count + eval_count
    part = args.work_dir / "coco_sam.jsonl"
    if (rows := valid_part(part, total)) is None:
        annotations_zip = args.coco_root / "annotations_trainval2017.zip"
        images_zip = args.coco_root / "train2017.zip"
        with zipfile.ZipFile(annotations_zip) as archive:
            payload = json.loads(archive.read(
                "annotations/instances_train2017.json"))
        categories = {int(row["id"]): str(row["name"])
                      for row in payload["categories"]}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            segmentation = annotation.get("segmentation")
            if (not annotation.get("iscrowd") and isinstance(segmentation, list)
                    and segmentation and float(annotation.get("area") or 0) > 0):
                annotations_by_image[int(annotation["image_id"])].append(annotation)
        candidates = []
        for image in payload["images"]:
            image_id = int(image["id"])
            annotations = annotations_by_image.get(image_id)
            if not annotations:
                continue
            # Exclude images whose only masks are tiny annotation artifacts.
            image_area = int(image["width"]) * int(image["height"])
            if max(float(row.get("area") or 0) for row in annotations) < image_area * 0.002:
                continue
            candidates.append({"i1_key": str(image_id), **image})
        selected = select_smallest(
            candidates, total + 256, seed=args.seed, namespace="coco_sam")
        selected_by_name = {f"train2017/{row['file_name']}": row
                            for row in selected}
        rows = []
        with zipfile.ZipFile(images_zip) as archive:
            for member in archive.infolist():
                image = selected_by_name.get(member.filename)
                if image is None:
                    continue
                image_id = int(image["id"])
                target, annotation_ids = _coco_sam_target(
                    annotations_by_image[image_id], categories,
                    int(image["width"]), int(image["height"]))
                row = final_row(
                    "coco_sam", str(image_id), archive.read(member), target,
                    "coco_instance_polygon_mask16", args.image_dir)
                row.update({
                    "prompt": coco_sam_prompt(target),
                    "stage1_source": "captioning_coco_sam_mask16",
                    "task": "sam_mask",
                    "caption_policy": "coco_polygon_mask16_v2_explicit_request",
                    "coco_image_id": image_id,
                    "coco_annotation_ids": annotation_ids,
                    "mask_grid": 16,
                })
                rows.append(row)
        rows.sort(key=lambda row: stable_rank(
            args.seed, "coco_sam", str(row["i1_key"])))
        rows = unique_content_rows(rows, total)
        if len(rows) != total:
            raise RuntimeError(
                f"COCO SAM materialized {len(rows)}/{total} unique images")
        write_jsonl(part, rows)
    else:
        # Upgrade resumable v1 work shards instead of silently preserving the
        # ambiguous prompt forever after valid_part accepts them.
        changed = False
        for row in rows:
            prompt = coco_sam_prompt(str(row["text"]))
            if row.get("prompt") != prompt or row.get("caption_policy") != (
                    "coco_polygon_mask16_v2_explicit_request"):
                row["prompt"] = prompt
                row["caption_policy"] = "coco_polygon_mask16_v2_explicit_request"
                changed = True
        if changed:
            write_jsonl(part, rows)
    return split_eval(rows, count, eval_count, args.seed, "coco_sam")


def _concept_sam_target(annotations: list[dict], width: int,
                        height: int) -> tuple[str, list[int]]:
    """Serialize every usable instance of one prompted concept."""
    lines = []
    annotation_ids = []
    for index, annotation in enumerate(sorted(
            annotations, key=lambda item: float(item.get("area") or 0),
            reverse=True), 1):
        spans = _mask_row_spans(annotation, width, height)
        if not spans:
            continue
        box = ",".join(f"{value:03d}" for value in _normalized_box(
            annotation, width, height))
        lines.append(f"instance {index}; box=[{box}]; mask16={spans}")
        annotation_ids.append(int(annotation["id"]))
        if len(lines) == 6:
            break
    if not lines:
        raise ValueError("concept has no polygon instance masks")
    return "\n".join(lines), annotation_ids


def build_lvis_sam(args: argparse.Namespace, count: int, eval_count: int,
                   excluded_image_ids: set[int]) -> tuple[list[dict], list[dict]]:
    """Build concept-conditioned segmentation tasks from LVIS v1 train.

    LVIS shares COCO train2017 pixels but expands supervision from 80 to about
    1,200 named concepts.  Each physical image appears once. Explicit LVIS
    negative-category annotations supply trustworthy absent-concept examples;
    categories marked non-exhaustive are never selected.
    """
    total = count + eval_count
    part = args.work_dir / "lvis_sam_v1.jsonl"
    if (rows := valid_part(part, total)) is None:
        annotations_zip = args.coco_root / "lvis_v1_train.json.zip"
        images_zip = args.coco_root / "train2017.zip"
        with zipfile.ZipFile(annotations_zip) as archive:
            names = [name for name in archive.namelist()
                     if name.endswith("lvis_v1_train.json")]
            if len(names) != 1:
                raise RuntimeError("LVIS archive has no unique train annotation file")
            payload = json.loads(archive.read(names[0]))
        categories = {
            int(row["id"]): str(row["name"]).replace("_", " ")
            for row in payload["categories"]}
        grouped: dict[int, dict[int, list[dict]]] = defaultdict(
            lambda: defaultdict(list))
        for annotation in payload["annotations"]:
            segmentation = annotation.get("segmentation")
            if (not annotation.get("iscrowd") and isinstance(segmentation, list)
                    and segmentation and float(annotation.get("area") or 0) > 0):
                grouped[int(annotation["image_id"])][
                    int(annotation["category_id"])].append(annotation)

        candidates = []
        for image in payload["images"]:
            image_id = int(image["id"])
            if image_id in excluded_image_ids:
                continue
            by_category = grouped.get(image_id)
            if not by_category:
                continue
            not_exhaustive = {int(value) for value in
                              image.get("not_exhaustive_category_ids", [])}
            positive = [category_id for category_id in by_category
                        if category_id not in not_exhaustive]
            negative = [int(value) for value in image.get("neg_category_ids", [])
                        if int(value) in categories]
            # About 10% of rows teach calibrated absence, but only using LVIS's
            # explicit negatives. Never infer a negative from missing masks.
            choose_negative = bool(negative and stable_rank(
                args.seed, "lvis-negative", str(image_id)) % 10 == 0)
            choices = negative if choose_negative else positive
            if not choices:
                continue
            category_id = min(choices, key=lambda value: stable_rank(
                args.seed, f"lvis-concept:{image_id}", str(value)))
            file_name = Path(str(image.get("coco_url") or
                                 image.get("file_name") or "")).name
            if not file_name:
                continue
            candidates.append({
                "i1_key": str(image_id), "image_id": image_id,
                "file_name": file_name, "width": int(image["width"]),
                "height": int(image["height"]), "category_id": category_id,
                "negative": choose_negative,
            })

        selected = select_smallest(
            candidates, total + 512, seed=args.seed, namespace="lvis_sam_v1")
        selected_by_name = {f"train2017/{row['file_name']}": row
                            for row in selected}
        rows = []
        with zipfile.ZipFile(images_zip) as archive:
            for member in archive.infolist():
                selected_row = selected_by_name.get(member.filename)
                if selected_row is None:
                    continue
                category_id = int(selected_row["category_id"])
                image_id = int(selected_row["image_id"])
                if selected_row["negative"]:
                    target, annotation_ids = "none", []
                else:
                    target, annotation_ids = _concept_sam_target(
                        grouped[image_id][category_id],
                        int(selected_row["width"]), int(selected_row["height"]))
                concept = categories[category_id]
                row = final_row(
                    "lvis_sam", str(image_id), archive.read(member), target,
                    "lvis_v1_concept_polygon_mask16", args.image_dir)
                row.update({
                    "prompt": CONCEPT_SAM_PROMPT.format(concept=concept),
                    "stage1_source": "captioning_lvis_sam3_concept_mask16",
                    "task": "sam_mask",
                    "caption_policy": "lvis_v1_concept_polygon_mask16_v1",
                    "lvis_image_id": image_id,
                    "lvis_category_id": category_id,
                    "lvis_concept": concept,
                    "lvis_negative": bool(selected_row["negative"]),
                    "lvis_annotation_ids": annotation_ids,
                    "mask_grid": 16,
                })
                rows.append(row)
        rows.sort(key=lambda row: stable_rank(
            args.seed, "lvis_sam_v1", str(row["i1_key"])))
        rows = unique_content_rows(rows, total)
        if len(rows) != total:
            raise RuntimeError(
                f"LVIS SAM materialized {len(rows)}/{total} unique images")
        write_jsonl(part, rows)
    return split_eval(rows, count, eval_count, args.seed, "lvis_sam")


def split_eval(rows: list[dict], train_count: int, eval_count: int,
               seed: int, source: str) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: stable_rank(
        seed, f"eval:{source}", str(row.get("i1_key") or row["image"])))
    eval_rows = ordered[:eval_count]
    train_rows = ordered[eval_count:eval_count + train_count]
    if len(train_rows) != train_count or len(eval_rows) != eval_count:
        raise RuntimeError(f"invalid {source} train/eval split")
    return train_rows, eval_rows


def legacy_eval(args: argparse.Namespace, source: str,
                count: int) -> list[dict]:
    stage = f"eval_i1_{source}"
    candidates = [row for row in read_jsonl(args.legacy_eval)
                  if row.get("stage1_source") == stage
                  and rooted(row["image"]).is_file()]
    selected = select_smallest(
        ({**row, "selection_key": str(row.get("i1_key") or row["image"])}
         for row in candidates), count, seed=args.seed,
        namespace=f"legacy_eval:{source}", key_field="selection_key")
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)}/{count} legacy eval rows for {source}")
    output = []
    for row in selected:
        item = dict(row)
        item.pop("selection_key", None)
        item["text"] = clean_grounded_caption(str(item["text"]))
        item["prompt"] = CAPTION_PROMPT
        item["source"] = source
        item["stage1_source"] = f"captioning_eval_{source}"
        item["caption_policy"] = "grounded_consensus_selection_v1"
        output.append(item)
    return output


def validate(args: argparse.Namespace, train: list[dict],
             eval_rows: list[dict]) -> dict:
    train_paths = [str(rooted(row["image"]).resolve()) for row in train]
    eval_paths = [str(rooted(row["image"]).resolve()) for row in eval_rows]
    missing = [path for path in train_paths + eval_paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} materialized images are missing")
    if len(set(train_paths)) != len(train_paths):
        raise RuntimeError("training manifest repeats physical image paths")
    if len(set(eval_paths)) != len(eval_paths):
        raise RuntimeError("evaluation manifest repeats physical image paths")
    overlap = set(train_paths) & set(eval_paths)
    if overlap:
        raise RuntimeError(f"train/eval image leakage: {len(overlap)}")
    blank = [row for row in train + eval_rows if not str(row.get("text", "")).strip()]
    if blank:
        raise RuntimeError(f"{len(blank)} blank targets")
    hidden = [row for row in train + eval_rows
              if row.get("task") == "caption"
              and HIDDEN_RENDER_METADATA.search(str(row["text"]))]
    if hidden:
        raise RuntimeError(f"{len(hidden)} caption targets retain render metadata")
    generationisms = [row for row in train + eval_rows
                      if row.get("task") == "caption"
                      and GENERATIONISM.search(str(row["text"]))]
    if generationisms:
        raise RuntimeError(
            f"{len(generationisms)} caption targets retain generation prompt terms")
    train_digests = [str(row.get("image_sha256") or "") for row in train]
    if any(not digest for digest in train_digests):
        raise RuntimeError("training rows must carry materialized image hashes")
    if len(set(train_digests)) != len(train_digests):
        raise RuntimeError("training manifest contains duplicate image content")
    eval_digests = []
    for row, path in zip(eval_rows, eval_paths, strict=True):
        digest = str(row.get("image_sha256") or "")
        if not digest:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            row["image_sha256"] = digest
        eval_digests.append(digest)
    if len(set(eval_digests)) != len(eval_digests):
        raise RuntimeError("evaluation manifest contains duplicate image content")
    content_overlap = set(train_digests) & set(eval_digests)
    if content_overlap:
        raise RuntimeError(
            f"train/eval content leakage: {len(content_overlap)} hashes")
    source_counts = Counter(str(row["source"]) for row in train)
    eval_counts = Counter(str(row["source"]) for row in eval_rows)
    return {
        "schema": 1,
        "seed": args.seed,
        "rows": len(train),
        "eval_rows": len(eval_rows),
        "unique_train_images": len(set(train_paths)),
        "unique_eval_images": len(set(eval_paths)),
        "unique_train_content_hashes": len(set(train_digests)),
        "unique_eval_content_hashes": len(set(eval_digests)),
        "train_eval_image_overlap": 0,
        "train_eval_content_overlap": 0,
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {source: count / len(train)
                          for source, count in sorted(source_counts.items())},
        "eval_source_counts": dict(sorted(eval_counts.items())),
        "task_counts": dict(sorted(Counter(
            str(row["task"]) for row in train).items())),
        "caption_policy": (
            "full-image grounded selection across i1 variants; aesthetic and "
            "hidden generation metadata removed; explicit OCR transcription; "
            "COCO and LVIS concept-conditioned mask supervision for RADIO's "
            "distilled SAM 3 features"),
        "intentionally_deferred_sources": [
            "yfcc", "redcaps", "megalith10m", "rendered_text", "textatlas",
            "gptedit"],
    }


def main() -> None:
    args = parse_args()
    targets = {
        "pexels": args.pexels,
        "imagenet22k": args.imagenet22k,
        "places365": args.places365,
        "inaturalist": args.inaturalist,
        "doclingmatix": args.doclingmatix,
        "fluxreason": args.fluxreason,
        "midjourneyv6": args.midjourneyv6,
        "coco_sam": args.coco_sam,
        "lvis_sam": args.lvis_sam,
    }
    if min(targets.values()) < 1:
        raise SystemExit("every source target must be positive")
    required = [
        args.captions, args.source_root / "pexels",
        args.source_root / "imagenet22k",
        args.source_root / "places365/train_large_places365challenge.tar",
        args.source_root / "inaturalist/train.tar.gz",
        args.source_root / "fluxreason", args.source_root / "midjourneyv6",
        args.legacy_selection / "pexels_selection.jsonl",
        args.legacy_selection / "midjourneyv6_selection.jsonl",
        args.legacy_eval, args.ocr_train, args.ocr_eval,
        args.coco_root / "annotations_trainval2017.zip",
        args.coco_root / "lvis_v1_train.json.zip",
        args.coco_root / "train2017.zip",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    train: list[dict] = []
    eval_rows: list[dict] = []

    train.extend(build_pexels(args, args.pexels))
    eval_rows.extend(legacy_eval(args, "pexels", args.eval_per_source))

    rows, heldout = build_imagenet(
        args, args.imagenet22k, args.eval_per_source)
    train.extend(rows); eval_rows.extend(heldout)

    rows, heldout = build_archive_source(
        args, "places365", args.places365, args.eval_per_source,
        args.source_root / "places365/train_large_places365challenge.tar",
        places_key)
    train.extend(rows); eval_rows.extend(heldout)

    rows, heldout = build_archive_source(
        args, "inaturalist", args.inaturalist, args.eval_per_source,
        args.source_root / "inaturalist/train.tar.gz", stem_key)
    train.extend(rows); eval_rows.extend(heldout)

    ocr_train = read_jsonl(args.ocr_train)
    if len(ocr_train) < args.doclingmatix:
        raise RuntimeError(
            f"only {len(ocr_train)} DoclingMatix train rows for {args.doclingmatix}")
    ocr_train = select_smallest(
        ({**row, "selection_key": str(row.get("docling_key") or row["image"])}
         for row in ocr_train), args.doclingmatix, seed=args.seed,
        namespace="doclingmatix", key_field="selection_key")
    for row in ocr_train:
        row.pop("selection_key", None)
        row["source"] = "doclingmatix"
        row["prompt"] = OCR_PROMPT
        row["task"] = "ocr"
    train.extend(ocr_train)
    ocr_eval = read_jsonl(args.ocr_eval)
    if len(ocr_eval) < args.eval_per_source:
        raise RuntimeError("DoclingMatix eval is smaller than eval-per-source")
    for row in ocr_eval[:args.eval_per_source]:
        row["source"] = "doclingmatix"
        row["prompt"] = OCR_PROMPT
        row["task"] = "ocr"
    eval_rows.extend(ocr_eval[:args.eval_per_source])

    rows, heldout = build_flux(args, args.fluxreason, args.eval_per_source)
    train.extend(rows); eval_rows.extend(heldout)

    train.extend(build_midjourney(args, args.midjourneyv6))
    eval_rows.extend(legacy_eval(
        args, "midjourneyv6", args.eval_per_source))

    rows, heldout = build_coco_sam(
        args, args.coco_sam, args.eval_per_source)
    train.extend(rows); eval_rows.extend(heldout)

    coco_image_ids = {
        int(row["coco_image_id"]) for row in rows + heldout}
    rows, heldout = build_lvis_sam(
        args, args.lvis_sam, args.eval_per_source, coco_image_ids)
    train.extend(rows); eval_rows.extend(heldout)

    random.Random(args.seed).shuffle(train)
    eval_rows = sorted(eval_rows, key=lambda row: (
        str(row.get("source", "")), str(row.get("i1_key") or row["image"])))
    expected = sum(targets.values())
    if len(train) != expected:
        raise RuntimeError(f"built {len(train)}/{expected} training rows")
    receipt = validate(args, train, eval_rows)
    write_jsonl(args.output, train)
    write_jsonl(args.eval_output, eval_rows)
    receipt.update({
        "train_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
    })
    write_json(args.output.with_suffix(".summary.json"), receipt)
    print(json.dumps({"output": str(args.output),
                      "eval_output": str(args.eval_output), **receipt},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
