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
import queue
import random
import resource
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

# Run as a script the sibling module is already importable; loaded by file
# path from a test it is not, so make both work.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from batch_identity import batch_digest, chain_digests  # noqa: E402

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


@dataclass(frozen=True)
class DocumentWork:
    """How much real input work one document actually cost."""

    bytes_read: int
    characters: int
    tokens: int


def prepare_document(
    document: CorpusDocument,
    required_ids: int,
    tokenizer: Any,
    vocabulary: int,
) -> tuple[list[int], DocumentWork]:
    """Decode and tokenize one document into a fixed-length row of model IDs.

    Deliberately pure: the same document and shape always produce the same
    row, so moving this onto a worker thread changes how the input work is
    scheduled and never what a step trains on. That property is what the
    batch digests published below are able to prove rather than assert.
    """
    text, bytes_read = _decode_document(document)
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise RuntimeError(
            f"tokenizer produced no IDs for {document.relative_path!r}; "
            "no synthetic fallback is available"
        )
    model_ids = [int(token_id) % vocabulary for token_id in token_ids]
    repeats = (required_ids + len(model_ids) - 1) // len(model_ids)
    row = (model_ids * repeats)[:required_ids]
    return row, DocumentWork(
        bytes_read=bytes_read,
        characters=len(text),
        tokens=len(token_ids),
    )


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
    parser.add_argument(
        "--input-workers",
        type=int,
        default=0,
        help="worker threads decoding and tokenizing a batch; 0 keeps the "
             "serial baseline loader unchanged",
    )
    parser.add_argument(
        "--input-prefetch-depth",
        type=int,
        default=0,
        help="batches prepared ahead of the training step; 0 disables "
             "prefetch so input and compute stay strictly serial",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="first step index to load, so a resumed loader can be shown to "
             "produce the same batches as an uninterrupted run",
    )
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("steps must be positive")
    if arguments.input_workers < 0:
        raise SystemExit("input workers must not be negative")
    if arguments.input_prefetch_depth < 0:
        raise SystemExit("input prefetch depth must not be negative")
    if arguments.start_step < 0:
        raise SystemExit("start step must not be negative")
    if arguments.start_step >= arguments.steps:
        raise SystemExit("start step must be inside the run; --steps is the "
                         "total run length, not the number remaining")

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
    document_order: list[str] = []
    step_digests: list[str] = []

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

    def batch_documents(
        step_index: int, selected_batch: int,
    ) -> list[CorpusDocument]:
        """The documents a step consumes: a pure function of the step index.

        Because it depends on nothing but the step index and the seeded
        document set, a run resumed at step K sees exactly the batch an
        uninterrupted run saw at step K. --start-step exists to prove that.
        """
        offset = (step_index * selected_batch) % len(documents)
        return [
            documents[(offset + batch_index) % len(documents)]
            for batch_index in range(selected_batch)
        ]

    def load_batch(
        step_index: int, selected_sequence: int, selected_batch: int,
        tokenizer: Any, executor: ThreadPoolExecutor | None,
    ):
        nonlocal corpus_bytes_read, decoded_characters
        nonlocal documents_read, tokens_encoded
        required_ids = selected_sequence + 1
        batch = batch_documents(step_index, selected_batch)

        def prepare(document: CorpusDocument):
            return prepare_document(
                document, required_ids, tokenizer, vocabulary)

        # Executor.map yields in SUBMISSION order regardless of completion
        # order, so a worker pool cannot permute a batch. The digest below
        # proves that held for this run rather than trusting the contract.
        prepared = (
            list(executor.map(prepare, batch)) if executor is not None
            else [prepare(document) for document in batch]
        )

        rows: list[list[int]] = []
        for document, (row, work) in zip(batch, prepared):
            rows.append(row)
            corpus_bytes_read += work.bytes_read
            decoded_characters += work.characters
            documents_read += 1
            tokens_encoded += work.tokens
            document_order.append(document.relative_path)
        packed = torch.tensor(rows, dtype=torch.long)
        tokens, targets = packed[:, :-1], packed[:, 1:]
        return tokens, targets, batch_digest(step_index, tokens, targets)

    def batch_stream(
        step_indices: list[int], selected_sequence: int, selected_batch: int,
        tokenizer: Any, executor: ThreadPoolExecutor | None, depth: int,
    ):
        """Yield each step's batch, optionally prepared ahead of the step.

        With depth 0 the loader stays strictly serial: load, step, load, step.
        With a positive depth a single producer prepares batches ahead into a
        bounded queue, so the measured input wait becomes the time the training
        thread actually BLOCKS. Total input work is unchanged — it is overlapped
        with compute, never skipped, which is why the batch digests must and do
        stay identical between the two.
        """
        if depth <= 0:
            for step_index in step_indices:
                yield load_batch(
                    step_index, selected_sequence, selected_batch,
                    tokenizer, executor)
            return

        pending: queue.Queue = queue.Queue(maxsize=depth)
        finished = object()
        stopping = threading.Event()

        def offer(item: Any) -> bool:
            """Hand one item over, giving up if the consumer went away.

            A plain blocking put would park forever on a full queue when the
            consumer exits early, and the join below would then deadlock.
            """
            while not stopping.is_set():
                try:
                    pending.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for step_index in step_indices:
                    batch = load_batch(
                        step_index, selected_sequence, selected_batch,
                        tokenizer, executor)
                    if not offer(batch):
                        return
            except BaseException as error:  # surfaced on the consuming thread
                offer(error)
            else:
                offer(finished)

        producer = threading.Thread(
            target=produce, name="ao3-input-prefetch", daemon=True)
        producer.start()
        try:
            while True:
                item = pending.get()
                if item is finished:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stopping.set()
            producer.join(timeout=30.0)

    final_gradient_norm = float("nan")
    executed_steps = list(range(arguments.start_step, arguments.steps))
    with ztok.Pipeline.from_rwkv(tokenizer_vocab) as tokenizer:
        # One persistent pool for the whole run: per-step pools would charge
        # every step for thread creation and measure the pool, not the loader.
        executor = (
            ThreadPoolExecutor(
                max_workers=arguments.input_workers,
                thread_name_prefix="ao3-input",
            )
            if arguments.input_workers > 0 else None
        )
        try:
            stream = iter(batch_stream(
                executed_steps, sequence_length, batch_size,
                tokenizer, executor, arguments.input_prefetch_depth))
            while True:
                # Timed around the CONSUMER pulling the next batch, so this is
                # the interval the training thread spends BLOCKED on input.
                # A `for` loop would advance the generator before the body and
                # measure nothing. Prefetch shrinks this by overlapping the
                # same work with compute; it never removes work, which is why
                # the digests below stay identical.
                input_started = time.perf_counter()
                try:
                    tokens, targets, step_digest = next(stream)
                except StopIteration:
                    break
                input_wait += time.perf_counter() - input_started
                step_digests.append(step_digest)

                step_started = time.perf_counter()
                loss, gradient_norm = measured_step(tokens, targets)
                step_seconds.append(time.perf_counter() - step_started)
                loss_value = float(loss.detach())
                final_gradient_norm = float(gradient_norm.detach())

            fallback = None
            if fallback_shape is not None:
                if arguments.fallback_bucket == arguments.bucket:
                    raise SystemExit(
                        "fallback bucket must be outside the compiled shape")
                fallback_sequence, fallback_batch = fallback_shape
                fallback_tokens, fallback_targets, _ = load_batch(
                    arguments.steps, fallback_sequence, fallback_batch,
                    tokenizer, executor)
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
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    batch_sequence_digest = chain_digests(step_digests)

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
        # How the same input work was scheduled. A receipt that shows a lower
        # input wait is only meaningful next to the mode that produced it.
        "input_workers": arguments.input_workers,
        "input_prefetch_depth": arguments.input_prefetch_depth,
        "input_pipeline_mode": (
            "serial" if arguments.input_workers == 0
            and arguments.input_prefetch_depth == 0
            else "parallel_prefetch" if arguments.input_workers > 0
            and arguments.input_prefetch_depth > 0
            else "parallel" if arguments.input_workers > 0
            else "prefetch"
        ),
        # The exact IDs each step trained on, chained in arrival order. This is
        # what makes ordering and content parity a measurement: any permutation
        # of documents within a batch, across batches, or a resumed cursor
        # landing on the wrong step changes the chained value.
        "batch_sequence_digest": batch_sequence_digest,
        "step_batch_digests": step_digests,
        "document_order_digest": "sha256:" + hashlib.sha256(
            "\0".join(document_order).encode("utf-8", "surrogatepass"),
        ).hexdigest(),
        "start_step": arguments.start_step,
        "executed_steps": len(executed_steps),
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
