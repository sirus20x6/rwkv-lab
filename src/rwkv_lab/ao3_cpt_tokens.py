"""Tokenize and exactly-once pack prepared AO3 continued-pretraining data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterator

import numpy as np


SCHEMA = "rwkv-lab.ao3-cpt-tokens.v1"
UINT32_LIMIT = np.iinfo(np.uint32).max


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepared_files(prepared: Path, split: str) -> list[Path]:
    files = sorted((prepared / "documents" / split).glob("*.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"no prepared {split} shards under {prepared}")
    return files


def iter_documents(prepared: Path, split: str) -> Iterator[dict[str, Any]]:
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("AO3 tokenization requires the zstandard package") from exc
    for path in _prepared_files(prepared, split):
        with path.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                for line in text:
                    value = json.loads(line)
                    if value.get("kind") != "pretrain" or value.get("split") != split:
                        raise ValueError(f"{path}: row violates prepared pretrain/{split} contract")
                    if not str(value.get("text") or "").strip():
                        raise ValueError(f"{path}: empty prepared document")
                    yield value


def _tokenizer_fingerprint(tokenizer_dir: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "config.json")
    result = {}
    for name in names:
        path = tokenizer_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = _sha256(path)
    return result


def _document_separator(tokenizer_dir: Path, tokenizer: Any) -> tuple[str, int, str]:
    """Choose the raw-document EOS from the model config.

    Chat tokenizers often advertise ``<|im_end|>`` as their API-level EOS,
    while the causal model config retains ``<|endoftext|>`` as its raw
    pretraining boundary. Continued pretraining should follow the latter.
    """
    model_config = json.loads((tokenizer_dir / "config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    eos_token_id = text_config.get("eos_token_id")
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[-1] if eos_token_id else None
    if isinstance(eos_token_id, int):
        eos_token = tokenizer.id_to_token(eos_token_id)
        if eos_token is None:
            raise ValueError(f"model EOS id {eos_token_id} is absent from tokenizer")
        return eos_token, eos_token_id, "model_config.text_config.eos_token_id"

    tokenizer_config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    eos_token_id = tokenizer.token_to_id(str(eos_token or ""))
    if eos_token_id is None:
        raise ValueError(f"tokenizer does not define eos token {eos_token!r}")
    return str(eos_token), eos_token_id, "tokenizer_config.eos_token"


def _tokenize_split(
    prepared: Path,
    split: str,
    tokenizer: Any,
    eos_token_id: int,
    output: Path,
    *,
    batch_char_budget: int,
    max_documents: int,
) -> dict[str, Any]:
    token_path = output / f"{split}.tokens.bin"
    offsets: list[int] = []
    lengths: list[int] = []
    id_digest = hashlib.sha256()
    token_count = 0
    document_count = 0
    batch: list[dict[str, Any]] = []
    batch_chars = 0

    with token_path.open("wb") as token_file:
        def flush() -> None:
            nonlocal token_count, document_count, batch_chars
            if not batch:
                return
            encodings = tokenizer.encode_batch(
                [str(row["text"]) for row in batch], add_special_tokens=False
            )
            for row, encoding in zip(batch, encodings):
                ids = list(encoding.ids)
                ids.append(eos_token_id)
                if ids and max(ids) > UINT32_LIMIT:
                    raise OverflowError("token id does not fit uint32")
                offsets.append(token_count)
                lengths.append(len(ids))
                np.asarray(ids, dtype=np.uint32).tofile(token_file)
                token_count += len(ids)
                document_count += 1
                id_digest.update(str(row["id"]).encode("utf-8", "surrogatepass") + b"\0")
            batch.clear()
            batch_chars = 0

        for row in iter_documents(prepared, split):
            if max_documents and document_count + len(batch) >= max_documents:
                break
            text_length = len(str(row["text"]))
            if batch and batch_chars + text_length > batch_char_budget:
                flush()
            batch.append(row)
            batch_chars += text_length
        flush()

    np.save(output / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(output / f"{split}.lengths.npy", np.asarray(lengths, dtype=np.uint64))
    return {
        "documents": document_count,
        "tokens": token_count,
        "tokens_file": token_path.name,
        "offsets_file": f"{split}.offsets.npy",
        "lengths_file": f"{split}.lengths.npy",
        "ordered_document_ids_sha256": id_digest.hexdigest(),
    }


def tokenize_prepared_corpus(
    prepared: str | Path,
    tokenizer_dir: str | Path,
    output: str | Path,
    *,
    batch_char_budget: int = 2_000_000,
    max_documents_per_split: int = 0,
) -> dict[str, Any]:
    """Create uint32 token streams plus document offsets without truncation."""
    if batch_char_budget < 1:
        raise ValueError("batch_char_budget must be positive")
    prepared = Path(prepared).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    prepared_manifest = prepared / "manifest.json"
    if not prepared_manifest.is_file():
        raise FileNotFoundError(prepared_manifest)

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("AO3 tokenization requires Hugging Face tokenizers") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    eos_token, eos_token_id, eos_source = _document_separator(
        tokenizer_dir, tokenizer
    )

    temporary = output.with_name(f".{output.name}.tmp-{time.time_ns()}")
    temporary.mkdir(parents=True)
    try:
        splits = {
            split: _tokenize_split(
                prepared,
                split,
                tokenizer,
                eos_token_id,
                temporary,
                batch_char_budget=batch_char_budget,
                max_documents=max_documents_per_split,
            )
            for split in ("train", "eval")
        }
        manifest = {
            "schema": SCHEMA,
            "prepared": str(prepared),
            "prepared_manifest_sha256": _sha256(prepared_manifest),
            "tokenizer": str(tokenizer_dir),
            "tokenizer_files_sha256": _tokenizer_fingerprint(tokenizer_dir),
            "eos_token": eos_token,
            "eos_token_id": eos_token_id,
            "eos_source": eos_source,
            "dtype": "uint32",
            "batch_char_budget": batch_char_budget,
            "max_documents_per_split": max_documents_per_split,
            "splits": splits,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def rewrite_token_cache_separator(
    token_cache: str | Path,
    tokenizer_dir: str | Path,
) -> dict[str, Any]:
    """Migrate only per-document separator IDs in an existing token cache."""
    token_cache = Path(token_cache).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    manifest_path = token_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("token cache schema mismatch")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    new_token, new_id, new_source = _document_separator(tokenizer_dir, tokenizer)
    old_id = int(manifest["eos_token_id"])
    if old_id == new_id:
        manifest["eos_source"] = new_source
        return manifest

    migrated: dict[str, int] = {}
    for split, split_manifest in manifest["splits"].items():
        tokens = np.memmap(
            token_cache / split_manifest["tokens_file"],
            mode="r+",
            dtype=np.uint32,
        )
        offsets = np.load(
            token_cache / split_manifest["offsets_file"], mmap_mode="r"
        )
        lengths = np.load(
            token_cache / split_manifest["lengths_file"], mmap_mode="r"
        )
        positions = np.asarray(offsets + lengths - 1, dtype=np.uint64)
        for start in range(0, len(positions), 100_000):
            selected = positions[start : start + 100_000]
            current = np.asarray(tokens[selected])
            if not np.all(current == old_id):
                bad = int(np.count_nonzero(current != old_id))
                raise ValueError(
                    f"{split}: {bad} document tails do not contain old EOS {old_id}"
                )
            tokens[selected] = new_id
        tokens.flush()
        migrated[split] = len(positions)

    manifest["eos_token"] = new_token
    manifest["eos_token_id"] = new_id
    manifest["eos_source"] = new_source
    manifest["separator_migration"] = {
        "old_eos_token_id": old_id,
        "new_eos_token_id": new_id,
        "documents": migrated,
    }
    temporary = manifest_path.with_name(f".manifest.json.tmp-{time.time_ns()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest


def pack_token_stream(
    token_cache: str | Path,
    split: str,
    output: str | Path,
    *,
    context_length: int = 8192,
    seed: int = 20260727,
    shuffle_documents: bool = True,
) -> dict[str, Any]:
    """Pack each source token exactly once (except a sub-context final tail).

    Documents are separated by the EOS token already present in each cached
    document.  Rows are full, so no padding token or loss mask is needed.
    """
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if split not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    token_cache = Path(token_cache).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    cache_manifest_path = token_cache / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    split_manifest = cache_manifest["splits"][split]
    tokens = np.memmap(
        token_cache / split_manifest["tokens_file"], mode="r", dtype=np.uint32
    )
    offsets = np.load(token_cache / split_manifest["offsets_file"], mmap_mode="r")
    lengths = np.load(token_cache / split_manifest["lengths_file"], mmap_mode="r")
    if len(offsets) != len(lengths):
        raise ValueError("token offsets/lengths have different sizes")
    if int(lengths.sum()) != len(tokens):
        raise ValueError("document lengths do not cover the token stream exactly")

    order = list(range(len(offsets)))
    if shuffle_documents:
        random.Random(seed).shuffle(order)
    rows = len(tokens) // context_length
    if rows < 1:
        raise ValueError("token stream is shorter than one context")
    temporary = output.with_name(f".{output.name}.tmp-{time.time_ns()}")
    temporary.mkdir(parents=True)
    packed_path = temporary / f"{split}.ctx{context_length}.bin"
    packed = np.memmap(
        packed_path, mode="w+", dtype=np.uint32, shape=(rows, context_length)
    )
    capacity = rows * context_length
    written = 0
    for index in order:
        if written >= capacity:
            break
        start = int(offsets[index])
        remaining = min(int(lengths[index]), capacity - written)
        source_position = start
        while remaining:
            row = written // context_length
            column = written % context_length
            amount = min(remaining, context_length - column)
            packed[row, column : column + amount] = tokens[
                source_position : source_position + amount
            ]
            written += amount
            source_position += amount
            remaining -= amount
    packed.flush()
    del packed
    manifest = {
        "schema": SCHEMA,
        "token_cache": str(token_cache),
        "token_cache_manifest_sha256": _sha256(cache_manifest_path),
        "split": split,
        "context_length": context_length,
        "seed": seed,
        "shuffle_documents": shuffle_documents,
        "rows": rows,
        "packed_tokens": written,
        "dropped_tail_tokens": len(tokens) - written,
        "dtype": "uint32",
        "packed_file": packed_path.name,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    tokenize = subparsers.add_parser("tokenize")
    tokenize.add_argument("--prepared", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)
    tokenize.add_argument("--batch-char-budget", type=int, default=2_000_000)
    tokenize.add_argument("--max-documents-per-split", type=int, default=0)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--token-cache", type=Path, required=True)
    pack.add_argument("--split", choices=("train", "eval"), required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--context-length", type=int, default=8192)
    pack.add_argument("--seed", type=int, default=20260727)
    pack.add_argument("--no-shuffle-documents", action="store_true")
    rewrite = subparsers.add_parser("rewrite-eos")
    rewrite.add_argument("--token-cache", type=Path, required=True)
    rewrite.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "tokenize":
        result = tokenize_prepared_corpus(
            args.prepared,
            args.tokenizer,
            args.output,
            batch_char_budget=args.batch_char_budget,
            max_documents_per_split=args.max_documents_per_split,
        )
    elif args.action == "pack":
        result = pack_token_stream(
            args.token_cache,
            args.split,
            args.output,
            context_length=args.context_length,
            seed=args.seed,
            shuffle_documents=not args.no_shuffle_documents,
        )
    else:
        result = rewrite_token_cache_separator(
            args.token_cache,
            args.tokenizer,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
