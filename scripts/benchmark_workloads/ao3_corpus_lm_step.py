#!/usr/bin/env python3
"""One portable language-model step with real AO3 input work, measured.

Unlike the synthetic benchmark workloads, every step reads complete source
documents from the AO3 corpus, decodes their UTF-8 bytes, tokenizes all of the
decoded text with the RWKV World tokenizer, and only then buckets token IDs for
the small CPU model. Source documents stay within a bounded 128--256 KiB band
so a warm page-cache read cannot turn this into a tiny-file loader benchmark
and batch size, rather than an outlier document, controls input work.

Run as a subprocess by scripts/run_benchmark_fixture.py, one process per
phase, and emit a single JSON report on stdout. Nothing here decides whether a
result qualifies, and there is deliberately no synthetic fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import resource
import sqlite3
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

DEFAULT_CORPUS_INDEX = pathlib.Path(
    "/thearray/data/AO3_final_location/ao3_current.sqlite3"
)
DEFAULT_CORPUS_ROOT = pathlib.Path(
    "/thearray/data/AO3_final_location/selected"
)
DEFAULT_TOKENIZER_VOCAB = pathlib.Path(
    "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt"
)
MINIMUM_DOCUMENT_BYTES = 128 * 1024
MAXIMUM_DOCUMENT_BYTES = 256 * 1024
INPUT_PIPELINE = "ao3_raw_utf8_decode_ztok"
RAW_TEXT_SUFFIXES = {".htm", ".html", ".txt"}


@dataclass(frozen=True)
class CorpusDocument:
    rowid: int
    relative_path: str
    path: pathlib.Path
    size_bytes: int


class _VisibleText(HTMLParser):
    """Extract visible text while retaining deterministic document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(
        self, tag: str, _attributes: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(part for part in self.parts if part.strip())


def parse_bucket(bucket: str) -> tuple[int, int]:
    """'seq512xbatch8' -> (512, 8). Shapes come from the fixture."""
    try:
        sequence, batch = bucket.split("x")
        sequence_length = int(sequence.removeprefix("seq"))
        batch_size = int(batch.removeprefix("batch"))
    except ValueError as error:
        raise SystemExit(f"unsupported shape bucket {bucket!r}") from error
    if sequence_length < 1 or batch_size < 1:
        raise SystemExit(f"unsupported shape bucket {bucket!r}")
    return sequence_length, batch_size


def _configuration_path(environment_name: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ.get(environment_name, str(default)))


def corpus_configuration() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return host paths, allowing tests/operators to point at another copy."""
    return (
        _configuration_path("MOE_BENCHMARK_AO3_INDEX", DEFAULT_CORPUS_INDEX),
        _configuration_path("MOE_BENCHMARK_AO3_ROOT", DEFAULT_CORPUS_ROOT),
        _configuration_path(
            "MOE_BENCHMARK_RWKV_VOCAB", DEFAULT_TOKENIZER_VOCAB),
    )


def _require_file(path: pathlib.Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} does not exist: {path}; "
            "no synthetic fallback is available"
        )


def select_documents(
    corpus_index: pathlib.Path,
    corpus_root: pathlib.Path,
    *,
    seed: int,
    count: int,
    minimum_document_bytes: int = MINIMUM_DOCUMENT_BYTES,
    maximum_document_bytes: int = MAXIMUM_DOCUMENT_BYTES,
    maximum_windows: int = 64,
) -> list[CorpusDocument]:
    """Select the same large, resolvable documents for the same seed.

    The indexed legacy AO3 sources are raw UTF-8 text/HTML, avoiding the
    current-source EPUB paths that are known to include stale entries.  A
    seeded rowid window keeps selection deterministic without materializing
    millions of paths or using SQLite's nondeterministic ``random()``.
    """
    _require_file(corpus_index, "AO3 corpus index")
    if not corpus_root.is_dir():
        raise FileNotFoundError(
            f"AO3 corpus root does not exist: {corpus_root}; "
            "no synthetic fallback is available"
        )
    if count < 1:
        raise ValueError("document count must be positive")
    if minimum_document_bytes < 1:
        raise ValueError("minimum document size must be positive")
    if maximum_document_bytes < minimum_document_bytes:
        raise ValueError("maximum document size must not be smaller than minimum")

    index_uri = corpus_index.resolve().as_uri() + "?mode=ro&immutable=1"
    selected: list[CorpusDocument] = []
    seen_rowids: set[int] = set()
    generator = random.Random(seed)
    try:
        with sqlite3.connect(index_uri, uri=True) as connection:
            maximum_rowid_row = connection.execute(
                "SELECT MAX(rowid) FROM final_selection"
            ).fetchone()
            maximum_rowid = int(maximum_rowid_row[0] or 0)
            if maximum_rowid < 1:
                raise RuntimeError("AO3 final_selection index is empty")

            query = """
                SELECT rowid, path
                FROM final_selection
                WHERE rowid >= ?
                  AND file_present = 1
                  AND source IN ('old016', 'old17')
                ORDER BY rowid
                LIMIT 128
            """
            for _ in range(maximum_windows):
                starting_rowid = generator.randrange(1, maximum_rowid + 1)
                for rowid, relative in connection.execute(
                    query, (starting_rowid,),
                ):
                    rowid = int(rowid)
                    if rowid in seen_rowids or not isinstance(relative, str):
                        continue
                    seen_rowids.add(rowid)
                    relative_path = pathlib.PurePosixPath(relative)
                    if (relative_path.is_absolute()
                            or ".." in relative_path.parts
                            or relative_path.suffix.lower()
                            not in RAW_TEXT_SUFFIXES):
                        continue
                    document_path = corpus_root.joinpath(*relative_path.parts)
                    try:
                        size_bytes = document_path.stat().st_size
                    except OSError:
                        continue
                    if not (
                        minimum_document_bytes
                        <= size_bytes
                        <= maximum_document_bytes
                    ):
                        continue
                    selected.append(CorpusDocument(
                        rowid=rowid,
                        relative_path=relative,
                        path=document_path,
                        size_bytes=size_bytes,
                    ))
                    if len(selected) == count:
                        return selected
    except sqlite3.Error as error:
        raise RuntimeError(
            f"AO3 corpus index is unusable: {corpus_index}: {error}; "
            "no synthetic fallback is available"
        ) from error

    raise RuntimeError(
        "AO3 corpus did not provide "
        f"{count} resolvable raw UTF-8 documents between "
        f"{minimum_document_bytes} and {maximum_document_bytes} bytes "
        f"under {corpus_root}; "
        "no synthetic fallback is available"
    )


def _decode_document(document: CorpusDocument) -> tuple[str, int]:
    try:
        source_bytes = document.path.read_bytes()
        if len(source_bytes) != document.size_bytes:
            raise RuntimeError(
                f"AO3 document {document.relative_path!r} changed size "
                "after deterministic selection; "
                "no synthetic fallback is available"
            )
        decoded = source_bytes.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"cannot read and UTF-8 decode AO3 document "
            f"{document.relative_path!r}: {error}; "
            "no synthetic fallback is available"
        ) from error
    if document.path.suffix.lower() in {".htm", ".html"}:
        parser = _VisibleText()
        parser.feed(decoded)
        decoded = parser.text()
    if not decoded.strip():
        raise RuntimeError(
            f"AO3 document {document.relative_path!r} decoded to empty text; "
            "no synthetic fallback is available"
        )
    return decoded, len(source_bytes)


def _selection_digest(documents: list[CorpusDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.relative_path.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=["cold", "warmup", "timed"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compile", action="store_true", dest="compile_step")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default="default",
    )
    parser.add_argument("--fallback-bucket")
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("steps must be positive")

    sequence_length, batch_size = parse_bucket(arguments.bucket)
    fallback_shape = (
        parse_bucket(arguments.fallback_bucket)
        if arguments.fallback_bucket else None
    )
    largest_batch = max(
        batch_size,
        fallback_shape[1] if fallback_shape is not None else 0,
    )
    # Two disjoint batches make document-set size a real scaling dimension
    # while keeping corpus discovery outside the measured input interval.
    selected_document_count = largest_batch * min(arguments.steps, 2)
    corpus_index, corpus_root, tokenizer_vocab = corpus_configuration()
    _require_file(tokenizer_vocab, "RWKV tokenizer vocabulary")
    documents = select_documents(
        corpus_index,
        corpus_root,
        seed=arguments.seed,
        count=selected_document_count,
    )

    # Imports stay inside main so a cold subprocess pays its actual startup
    # cost. Tokenizer construction is setup; encode calls remain inside each
    # measured input-wait interval below.
    try:
        import torch
        import ztok
        from torch import nn
    except ImportError as error:
        raise SystemExit(
            f"real AO3 input pipeline dependency is unavailable: {error}; "
            "no synthetic fallback is available"
        ) from error

    import rwkv_lab.training_components as components

    torch.manual_seed(arguments.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    vocabulary, width = 512, 128

    model = nn.Sequential(
        nn.Embedding(vocabulary, width),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.GELU(),
        nn.LayerNorm(width),
    )
    head = nn.Linear(width, vocabulary, bias=False)
    parameters = list(model.parameters()) + list(head.parameters())

    optimizer = components.build_registered_optimizer(
        components.OptimizerImplementation.FP32_MASTER_ADAMW_V1,
        parameters,
        components.AdamWConfiguration(learning_rate=1e-3),
    )
    schedule = components.build_registered_schedule(
        components.ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        components.LinearWarmupCosineConfiguration(
            warmup_steps=2, max_steps=max(4, arguments.steps)),
    )
    objective = components.build_registered_objective(
        components.ObjectiveImplementation.LINEAR_HEAD_CROSS_ENTROPY_V1,
        components.LinearHeadCrossEntropyConfiguration(),
    )

    step_seconds: list[float] = []
    input_wait = 0.0
    loss_value = float("nan")
    corpus_bytes_read = 0
    decoded_characters = 0
    documents_read = 0
    tokens_encoded = 0

    def training_step(tokens, targets):
        hidden = model(tokens)
        loss = objective(hidden, head, targets)
        loss.backward()
        gradient_norm = torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach().to(torch.float64))
            for parameter in parameters
            if parameter.grad is not None
        ]).sum()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach(), gradient_norm.detach()

    measured_step = training_step
    if arguments.compile_step:
        measured_step = torch.compile(
            training_step,
            dynamic=False,
            mode=arguments.compile_mode,
        )

    def load_batch(
        step_index: int, selected_sequence: int, selected_batch: int,
        tokenizer: Any,
    ):
        nonlocal corpus_bytes_read, decoded_characters
        nonlocal documents_read, tokens_encoded
        rows: list[list[int]] = []
        required_ids = selected_sequence + 1
        document_offset = (step_index * selected_batch) % len(documents)
        for batch_index in range(selected_batch):
            document = documents[
                (document_offset + batch_index) % len(documents)]
            text, bytes_read = _decode_document(document)
            token_ids = tokenizer.encode(text)
            if not token_ids:
                raise RuntimeError(
                    f"tokenizer produced no IDs for "
                    f"{document.relative_path!r}; "
                    "no synthetic fallback is available"
                )
            corpus_bytes_read += bytes_read
            decoded_characters += len(text)
            documents_read += 1
            tokens_encoded += len(token_ids)
            model_ids = [int(token_id) % vocabulary for token_id in token_ids]
            repeats = (required_ids + len(model_ids) - 1) // len(model_ids)
            rows.append((model_ids * repeats)[:required_ids])
        packed = torch.tensor(rows, dtype=torch.long)
        return packed[:, :-1], packed[:, 1:]

    final_gradient_norm = float("nan")
    with ztok.Pipeline.from_rwkv(tokenizer_vocab) as tokenizer:
        for step_index in range(arguments.steps):
            input_started = time.perf_counter()
            tokens, targets = load_batch(
                step_index, sequence_length, batch_size, tokenizer)
            input_wait += time.perf_counter() - input_started

            step_started = time.perf_counter()
            loss, gradient_norm = measured_step(tokens, targets)
            step_seconds.append(time.perf_counter() - step_started)
            loss_value = float(loss.detach())
            final_gradient_norm = float(gradient_norm.detach())

        fallback = None
        if fallback_shape is not None:
            if arguments.fallback_bucket == arguments.bucket:
                raise SystemExit("fallback bucket must be outside the compiled shape")
            fallback_sequence, fallback_batch = fallback_shape
            fallback_tokens, fallback_targets = load_batch(
                arguments.steps, fallback_sequence, fallback_batch, tokenizer)
            fallback_started = time.perf_counter()
            fallback_loss, fallback_gradient_norm = measured_step(
                fallback_tokens, fallback_targets)
            fallback_seconds = time.perf_counter() - fallback_started
            fallback = {
                "bucket": arguments.fallback_bucket,
                "step_seconds": fallback_seconds,
                "result_fingerprint": {
                    "final_loss": float(fallback_loss.detach()),
                    "gradient_norm_sum": float(
                        fallback_gradient_norm.detach()),
                },
            }

    first_step = step_seconds[0]
    sorted_step_seconds = sorted(step_seconds)
    median = sorted_step_seconds[len(sorted_step_seconds) // 2]
    training_step_seconds = sum(step_seconds)
    measured_total = input_wait + training_step_seconds
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    json.dump({
        "median_step_seconds": median,
        "first_step_seconds": first_step,
        "steps_per_second": 1.0 / median if median > 0 else 0.0,
        "peak_memory_bytes": peak_memory,
        "peak_memory_kind": "process_max_rss",
        "input_wait_seconds": input_wait,
        "training_step_seconds": training_step_seconds,
        "input_wait_ratio": input_wait / measured_total if measured_total else 0.0,
        "input_pipeline": INPUT_PIPELINE,
        "corpus_index": str(corpus_index),
        "corpus_root": str(corpus_root),
        "minimum_document_bytes": MINIMUM_DOCUMENT_BYTES,
        "maximum_document_bytes": MAXIMUM_DOCUMENT_BYTES,
        "document_set_size": len(documents),
        "documents_read": documents_read,
        "corpus_bytes_read": corpus_bytes_read,
        "decoded_characters": decoded_characters,
        "tokens_encoded": tokens_encoded,
        "document_selection_digest": _selection_digest(documents),
        "tokenizer": "ztok_rwkv_world",
        "tokenizer_version": ztok.version(),
        "final_loss": loss_value,
        "result_fingerprint": {
            "final_loss": loss_value,
            "gradient_norm_sum": final_gradient_norm,
        },
        "quality_metric": "cross_entropy",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "accelerator": False,
        "compiled": arguments.compile_step,
        "compile_mode": (
            arguments.compile_mode if arguments.compile_step else None),
        "fallback": fallback,
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
