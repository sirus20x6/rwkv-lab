"""Reproducible preparation for AO3 continued-pretraining corpora.

The source selection is treated as immutable.  This module writes a separate,
versioned corpus and records rejection reason codes without copying rejected
titles or text into logs.  It deliberately handles both singular and plural
AO3 metadata keys; historical exports use both forms.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterator


SCHEMA = "rwkv-lab.ao3-cpt.v1"

_SPACE_RE = re.compile(r"\s+")
_DIRECT_MINOR_RE = re.compile(
    r"\b(?:"
    r"underage\s+sex|"
    r"sex(?:ual(?:ly)?)?\s+(?:with|involving)\s+(?:a\s+)?minor|"
    r"sexual\s+(?:abuse|assault|exploitation)\s+of\s+(?:a\s+)?(?:minor|child)|"
    r"(?:child|minor)\s+(?:sexual\s+)?(?:abuse|assault|exploitation|prostitution)|"
    r"pedophil(?:e|ia|ic)"
    r")\b",
    re.IGNORECASE,
)
_MINOR_TAG_RE = re.compile(
    r"\b(?:"
    r"underage(?:\s+(?:sex|rape|prostitution))?|"
    r"extremely\s+underage|"
    r"consensual\s+underage\s+sex|"
    r"minor\s*/\s*adult\s+(?:relationship|sex)|"
    r"(?:child|teen)\s+prostitution|"
    r"sexual\s+abuse\s+of\s+(?:a\s+)?minor"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_MINOR_TAG_RE = re.compile(
    r"\b(?:not\s+underage|no\s+underage\s+sex|aged[- ]up\s+characters?)\b",
    re.IGNORECASE,
)
_NUMERIC_MINOR_RE = re.compile(
    r"\b(?:"
    r"(?:a|an|the|he|she|they|boy|girl|teen|teenager|student|character)\s+"
    r"(?:is|was|aged?|age(?:d)?\s+)?(?:[1-9]|1[0-7])\b|"
    r"(?:[1-9]|1[0-7])[- ]year[- ]old\b|"
    r"age(?:d)?\s+(?:[1-9]|1[0-7])\b"
    r")",
    re.IGNORECASE,
)
_SEXUAL_CONTEXT_RE = re.compile(
    r"\b(?:sex|sexual|intercourse|nude|naked|orgasm|aroused|genitals?|rape)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AO3FilterPolicy:
    min_text_chars: int = 256
    require_english: bool = True
    reject_minor_content: bool = True
    age_context_chars: int = 512
    eval_fraction: float = 0.001
    split_seed: int = 20260727
    output_shard_records: int = 10_000

    def validate(self) -> None:
        if self.min_text_chars < 1:
            raise ValueError("min_text_chars must be positive")
        if self.age_context_chars < 0:
            raise ValueError("age_context_chars cannot be negative")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in [0, 1)")
        if self.output_shard_records < 1:
            raise ValueError("output_shard_records must be positive")


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reason: str


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalized_metadata(metadata: Any) -> dict[str, tuple[str, ...]]:
    """Return case/punctuation-insensitive metadata while retaining all aliases."""
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, list[str]] = {}
    for raw_key, raw_value in metadata.items():
        values = raw_value if isinstance(raw_value, (list, tuple, set)) else (raw_value,)
        bucket = result.setdefault(_key(raw_key), [])
        for value in values:
            text = _SPACE_RE.sub(" ", str(value or "")).strip()
            if text:
                bucket.append(text)
    return {name: tuple(values) for name, values in result.items()}


def metadata_values(metadata: Any, *names: str) -> tuple[str, ...]:
    normalized = normalized_metadata(metadata)
    values: list[str] = []
    for name in names:
        values.extend(normalized.get(_key(name), ()))
    return tuple(values)


def _metadata_minor_reason(metadata: Any) -> str | None:
    warnings = metadata_values(metadata, "Archive Warning", "Archive Warnings")
    if any(re.search(r"\bunderage\s+sex\b", warning, re.IGNORECASE) for warning in warnings):
        return "minor_archive_warning"

    candidate_fields = (
        "Additional Tag",
        "Additional Tags",
        "Freeform",
        "Freeforms",
        "Relationship",
        "Relationships",
        "Category",
        "Categories",
    )
    for value in metadata_values(metadata, *candidate_fields):
        scrubbed = _NEGATED_MINOR_TAG_RE.sub("", value)
        if _MINOR_TAG_RE.search(scrubbed) or _DIRECT_MINOR_RE.search(scrubbed):
            return "minor_metadata_tag"
    return None


def _text_minor_reason(text: str, *, context_chars: int) -> str | None:
    if _DIRECT_MINOR_RE.search(text):
        return "minor_text_direct"
    for match in _NUMERIC_MINOR_RE.finditer(text):
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        if _SEXUAL_CONTEXT_RE.search(text[start:end]):
            return "minor_text_age_context"
    return None


def filter_record(record: Any, policy: AO3FilterPolicy = AO3FilterPolicy()) -> FilterDecision:
    policy.validate()
    if not isinstance(record, dict):
        return FilterDecision(False, "invalid_record")
    if not str(record.get("id") or "").strip():
        return FilterDecision(False, "missing_id")
    text = str(record.get("text") or "").replace("\x00", "").strip()
    if len(text) < policy.min_text_chars:
        return FilterDecision(False, "text_too_short")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return FilterDecision(False, "invalid_metadata")
    if policy.require_english:
        languages = metadata_values(metadata, "Language", "Languages")
        if not languages or not all(value.casefold() == "english" for value in languages):
            return FilterDecision(False, "not_english")
    if policy.reject_minor_content:
        reason = _metadata_minor_reason(metadata)
        if reason:
            return FilterDecision(False, reason)
        reason = _text_minor_reason(text, context_chars=policy.age_context_chars)
        if reason:
            return FilterDecision(False, reason)
    return FilterDecision(True, "accepted")


def render_document(record: dict[str, Any]) -> str:
    """Render prose for causal pretraining without leaking tag-list syntax into it."""
    title = _SPACE_RE.sub(" ", str(record.get("title") or "")).strip()
    text = str(record.get("text") or "").replace("\x00", "").strip()
    return f"{title}\n\n{text}" if title else text


def deterministic_split(record_id: Any, policy: AO3FilterPolicy = AO3FilterPolicy()) -> str:
    if policy.eval_fraction <= 0:
        return "train"
    payload = f"{policy.split_seed}\0{record_id}".encode("utf-8", "surrogatepass")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return "eval" if value < policy.eval_fraction else "train"


def hashed_id(record_id: Any) -> str:
    return hashlib.sha256(str(record_id).encode("utf-8", "surrogatepass")).hexdigest()


def _source_files(source: Path) -> list[Path]:
    selected = source / "selected"
    root = selected if selected.is_dir() else source
    files = sorted(root.glob("*.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"no .jsonl.zst shards under {root}")
    return files


def iter_source_records(source: Path) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("AO3 preparation requires the zstandard package") from exc

    for shard in _source_files(source):
        with shard.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for line_number, line in enumerate(text, 1):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        value = {"_malformed": True}
                    yield shard, line_number, value


class _CompressedShardWriter:
    def __init__(self, root: Path, split: str, records_per_shard: int):
        import zstandard as zstd

        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.records_per_shard = records_per_shard
        self.zstd = zstd
        self.total = 0
        self.shard_records = 0
        self.shard_index = -1
        self.raw = None
        self.stream = None
        self.text = None
        self.files: list[str] = []

    def _open_next(self) -> None:
        self.close_current()
        self.shard_index += 1
        path = self.root / f"part-{self.shard_index:05d}.jsonl.zst"
        self.raw = path.open("wb")
        self.stream = self.zstd.ZstdCompressor(level=6, threads=-1).stream_writer(
            self.raw, closefd=False
        )
        self.text = io.TextIOWrapper(self.stream, encoding="utf-8")
        self.shard_records = 0
        self.files.append(str(path))

    def write(self, value: dict[str, Any]) -> None:
        if self.text is None or self.shard_records >= self.records_per_shard:
            self._open_next()
        assert self.text is not None
        self.text.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.shard_records += 1
        self.total += 1

    def close_current(self) -> None:
        if self.text is not None:
            self.text.flush()
            self.text.detach()
            self.stream.close()
            self.raw.close()
        self.raw = self.stream = self.text = None

    def close(self) -> None:
        self.close_current()


def _source_fingerprint(source: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    config = source / "run_config.json"
    if config.is_file():
        digest.update(config.read_bytes())
    for path in files:
        stat = path.stat()
        digest.update(path.name.encode() + b"\0")
        digest.update(str(stat.st_size).encode() + b"\0")
    return digest.hexdigest()


def prepare_ao3_corpus(
    source: str | Path,
    output: str | Path,
    *,
    policy: AO3FilterPolicy = AO3FilterPolicy(),
    max_records: int = 0,
) -> dict[str, Any]:
    """Filter and materialize a deterministic pretraining corpus.

    ``max_records`` limits source rows and is intended for smoke tests.  Existing
    output directories are never overwritten.
    """
    policy.validate()
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    files = _source_files(source)
    temporary = output.with_name(f".{output.name}.tmp-{time.time_ns()}")
    temporary.mkdir(parents=True)
    writers = {
        split: _CompressedShardWriter(temporary / "documents", split, policy.output_shard_records)
        for split in ("train", "eval")
    }
    rejection_writer = _CompressedShardWriter(
        temporary, "rejections", policy.output_shard_records
    )
    counts: Counter[str] = Counter()
    split_chars: Counter[str] = Counter()
    split_words: Counter[str] = Counter()
    scanned = 0
    try:
        for shard, line_number, record in iter_source_records(source):
            if max_records and scanned >= max_records:
                break
            scanned += 1
            if record.get("_malformed"):
                decision = FilterDecision(False, "malformed_json")
            else:
                decision = filter_record(record, policy)
            counts[decision.reason] += 1
            if not decision.accepted:
                rejection_writer.write(
                    {
                        "schema": SCHEMA,
                        "id_sha256": hashed_id(record.get("id", f"{shard.name}:{line_number}")),
                        "source_shard": shard.name,
                        "source_line": line_number,
                        "reason": decision.reason,
                    }
                )
                continue

            rendered = render_document(record)
            split = deterministic_split(record["id"], policy)
            source_hash = hashlib.sha256(
                str(record.get("text") or "").encode("utf-8", "surrogatepass")
            ).hexdigest()
            writers[split].write(
                {
                    "schema": SCHEMA,
                    "id": str(record["id"]),
                    "split": split,
                    "kind": "pretrain",
                    "text": rendered,
                    "metadata": {
                        "source_shard": shard.name,
                        "source_line": line_number,
                        "source_text_sha256": source_hash,
                    },
                }
            )
            split_chars[split] += len(rendered)
            split_words[split] += len(rendered.split())
        for writer in (*writers.values(), rejection_writer):
            writer.close()

        manifest = {
            "schema": SCHEMA,
            "source": str(source),
            "source_fingerprint": _source_fingerprint(source, files),
            "source_shards": len(files),
            "policy": asdict(policy),
            "max_records": int(max_records),
            "counts": dict(sorted(counts.items())),
            "scanned": scanned,
            "accepted": sum(writer.total for writer in writers.values()),
            "rejected": rejection_writer.total,
            "splits": {
                split: {
                    "records": writer.total,
                    "characters": split_chars[split],
                    "whitespace_words": split_words[split],
                    "files": [
                        str(Path(path).relative_to(temporary)) for path in writer.files
                    ],
                }
                for split, writer in writers.items()
            },
            "rejection_files": [
                str(Path(path).relative_to(temporary)) for path in rejection_writer.files
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        return manifest
    except Exception:
        for writer in (*writers.values(), rejection_writer):
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare immutable AO3 source shards for continued pretraining"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/thearray/downloads/completed/ao3/ao3_filthiest_top5pct"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/thearray/downloads/completed/ao3/ao3_filthiest_top5pct_cpt"),
    )
    parser.add_argument("--min-text-chars", type=int, default=256)
    parser.add_argument("--age-context-chars", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.001)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--output-shard-records", type=int, default=10_000)
    parser.add_argument("--max-records", type=int, default=0, help="smoke-test source-row limit")
    parser.add_argument("--allow-non-english", action="store_true")
    parser.add_argument(
        "--allow-minor-content",
        action="store_true",
        help="unsafe opt-out; recorded in the output policy",
    )
    args = parser.parse_args()
    policy = AO3FilterPolicy(
        min_text_chars=args.min_text_chars,
        require_english=not args.allow_non_english,
        reject_minor_content=not args.allow_minor_content,
        age_context_chars=args.age_context_chars,
        eval_fraction=args.eval_fraction,
        split_seed=args.split_seed,
        output_shard_records=args.output_shard_records,
    )
    print(
        json.dumps(
            prepare_ao3_corpus(
                args.source,
                args.output,
                policy=policy,
                max_records=args.max_records,
            ),
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "AO3FilterPolicy",
    "FilterDecision",
    "SCHEMA",
    "deterministic_split",
    "filter_record",
    "hashed_id",
    "iter_source_records",
    "metadata_values",
    "main",
    "normalized_metadata",
    "prepare_ao3_corpus",
    "render_document",
]


if __name__ == "__main__":
    main()
