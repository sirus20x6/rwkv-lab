#!/usr/bin/env python3
"""One vision step with real image decode, measured.

Every step reads real encoded images from a manifest, decodes them with PIL,
converts to RGB, resizes to the bucket geometry, stacks a batch, and copies it
to the accelerator. That is the same decode path the project's own vision
pipeline uses (rwkv_lab.vision_cache.decode), reached through the same JSONL
manifest convention (rwkv_lab.vision_cache.manifest_images), so this measures
the loader that actually exists rather than a benchmark-shaped imitation.

Why this is a separate fixture from the AO3 text one rather than a variant:
image decode is a different cost shape. JPEG decode is CPU-heavy per sample and
scales with pixels rather than bytes, and a decoded batch is orders of magnitude
larger than the token IDs a text batch produces -- 224x224x3 float32 x 32 is
19 MiB per batch against a few KiB of IDs -- so the host-to-device copy is a
real cost here and is essentially free there.

Source images stay within a bounded size band so a directory full of thumbnails
cannot turn this into a stat() benchmark, and so batch size rather than one
outlier image controls the work per step.

Run as a subprocess by scripts/run_benchmark_fixture.py, one process per phase,
and emit a single JSON report on stdout. Nothing here decides whether a result
qualifies, and there is deliberately no synthetic fallback: a missing manifest
fails the cell.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import queue
import random
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from batch_identity import batch_digest, chain_digests  # noqa: E402

DEFAULT_MANIFEST = pathlib.Path(
    "/thearray/data/vision_benchmark/images.jsonl"
)
# The band exists for the same reason the AO3 fixture's does. Below it, decode
# is dominated by per-file overhead and the fixture measures the filesystem;
# above it, a single outlier image dominates a batch and step-to-step variance
# swamps the effect of batch size.
MINIMUM_IMAGE_BYTES = 200 * 1024
MAXIMUM_IMAGE_BYTES = 4 * 1024 * 1024
# A byte band does NOT bound decode work, which is the thing being measured.
# Compression ratio varies by orders of magnitude: the corpus this was built
# against contains a 197-megapixel image inside the 200 KiB..4 MiB band, which
# alone would dominate any batch it landed in and would trip PIL's
# decompression-bomb guard besides. Decoded geometry is bounded separately,
# from the header, so batch size really is what controls work per step.
MINIMUM_IMAGE_PIXELS = 256 * 256
MAXIMUM_IMAGE_PIXELS = 24 * 1024 * 1024
INPUT_PIPELINE = "manifest_jsonl_pil_decode_rgb_resize"


@dataclass(frozen=True)
class CorpusImage:
    relative_path: str
    path: pathlib.Path
    size_bytes: int
    pixels: int


def _shape(bucket: str) -> tuple[int, int]:
    """Parse "224x224xbatch32" into (edge, batch)."""
    try:
        geometry, batch = bucket.split("xbatch")
        width, height = geometry.split("x")
        if width != height:
            raise SystemExit(
                f"non-square bucket {bucket!r}; this fixture resizes to a "
                f"square edge and would silently distort otherwise")
        return int(width), int(batch)
    except ValueError as error:
        raise SystemExit(f"unsupported shape bucket {bucket!r}") from error


def _configuration_path(environment_name: str,
                        default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ.get(environment_name, str(default)))


def load_manifest(manifest: pathlib.Path) -> list[CorpusImage]:
    """Read the JSONL manifest, keeping only images inside the size band.

    Raises rather than degrading. A vision fixture that quietly substitutes
    generated frames when its corpus is absent reports decode timings for work
    it never did, which is the exact illusion this fixture exists to remove.
    """
    if not manifest.is_file():
        raise FileNotFoundError(
            f"image manifest {manifest} is missing; set "
            f"MOE_BENCHMARK_IMAGE_MANIFEST to a JSONL file of "
            f'{{"image": "<path>"}} rows. There is no synthetic fallback: '
            f"this fixture measures real decode or it fails.")
    images: dict[str, CorpusImage] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            recorded = row.get("image")
            if not recorded:
                continue
            path = pathlib.Path(recorded)
            if not path.is_absolute():
                path = manifest.parent / path
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if not (MINIMUM_IMAGE_BYTES <= size <= MAXIMUM_IMAGE_BYTES):
                continue
            # Header only: PIL reads dimensions without decoding pixels, so
            # bounding geometry costs a stat-sized read per image, once.
            try:
                from PIL import Image, UnidentifiedImageError

                with Image.open(path) as probe:
                    width, height = probe.size
            except (OSError, UnidentifiedImageError, ValueError):
                continue
            except Image.DecompressionBombError:
                # PIL refuses to open it at all, so it is not corpus this
                # fixture can decode. Skipping is not hiding a problem: the
                # count of admitted images is reported, and an empty corpus
                # still fails loudly below.
                continue
            pixels = width * height
            if not (MINIMUM_IMAGE_PIXELS <= pixels <= MAXIMUM_IMAGE_PIXELS):
                continue
            resolved = str(path.resolve())
            images[resolved] = CorpusImage(
                relative_path=recorded, path=path, size_bytes=size,
                pixels=pixels)
    if not images:
        raise RuntimeError(
            f"image manifest {manifest} yielded no images within "
            f"{MINIMUM_IMAGE_BYTES}..{MAXIMUM_IMAGE_BYTES} bytes; the corpus "
            f"is unusable rather than empty-but-fine")
    # Sorted so the corpus order is a property of the manifest, not of
    # dictionary iteration or filesystem readdir order.
    return [images[key] for key in sorted(images)]


def decode_image(image: CorpusImage, edge: int) -> "Any":
    """Read, decode, convert and resize one image. All of it is the point.

    The read is deliberately inside the timed region rather than hoisted: the
    encoded bytes are what a real loader pulls from disk, and separating them
    would make the measurement describe a cache rather than a pipeline.
    """
    from PIL import Image

    encoded = image.path.read_bytes()
    with Image.open(io.BytesIO(encoded)) as handle:
        # convert("RGB") is not cosmetic: it is where a CMYK JPEG, a palette
        # PNG or an alpha-carrying WebP actually gets converted, and it is what
        # rwkv_lab.vision_cache.decode does.
        rgb = handle.convert("RGB")
        return rgb.resize((edge, edge), Image.BILINEAR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=["cold", "warmup", "timed"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fallback-bucket")
    parser.add_argument(
        "--input-workers", type=int, default=0,
        help="worker threads decoding a batch; 0 keeps the serial baseline")
    parser.add_argument(
        "--input-prefetch-depth", type=int, default=0,
        help="batches prepared ahead of the step; 0 keeps input and compute "
             "strictly serial")
    parser.add_argument(
        "--start-step", type=int, default=0,
        help="first step index to load, so a resumed loader can be shown to "
             "produce the same batches as an uninterrupted run")
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("steps must be positive")
    if arguments.input_workers < 0:
        raise SystemExit("input workers must not be negative")
    if arguments.input_prefetch_depth < 0:
        raise SystemExit("input prefetch depth must not be negative")
    if arguments.start_step < 0 or arguments.start_step >= arguments.steps:
        raise SystemExit("start step must be within the step range")

    import torch
    from torch import nn

    edge, batch_size = _shape(arguments.bucket)
    manifest_path = _configuration_path(
        "MOE_BENCHMARK_IMAGE_MANIFEST", DEFAULT_MANIFEST)
    images = load_manifest(manifest_path)

    if not torch.cuda.is_available():
        raise SystemExit(
            "vision.multimodal-student declares accelerator_required; the "
            "host-to-device copy of a decoded batch is most of why this "
            "fixture differs from the text one, so there is no CPU fallback")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    # Small on purpose. This fixture measures the input pipeline, so the model
    # is only enough to consume a real batch and produce a gradient.
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 16),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    def batch_for(step: int, executor: ThreadPoolExecutor | None):
        chooser = random.Random(f"{arguments.seed}:{step}")
        chosen = [chooser.choice(images) for _ in range(batch_size)]
        if executor is None:
            decoded = [decode_image(image, edge) for image in chosen]
        else:
            decoded = list(executor.map(
                lambda image: decode_image(image, edge), chosen))
        # Bytes of the decoded pixels, so the digest covers what was actually
        # produced rather than which filenames were selected.
        pixels = torch.stack([
            torch.frombuffer(bytearray(item.tobytes()), dtype=torch.uint8)
                 .reshape(edge, edge, 3)
            for item in decoded
        ]).permute(0, 3, 1, 2).contiguous()
        # Step index bound in by batch_digest, so a reordering across a short
        # repeat period in a small image set cannot pass unnoticed.
        digest = batch_digest(step, pixels)
        return pixels, digest, [image.relative_path for image in chosen]

    def batch_stream(steps: list[int], executor: ThreadPoolExecutor | None,
                     depth: int):
        if depth <= 0:
            for step in steps:
                yield batch_for(step, executor)
            return
        pending: queue.Queue = queue.Queue(maxsize=depth)

        def produce() -> None:
            try:
                for step in steps:
                    pending.put(batch_for(step, executor))
            except BaseException as error:  # surfaced to the consumer
                pending.put(error)
                return
            pending.put(None)

        producer = threading.Thread(
            target=produce, name="vision-input-prefetch", daemon=True)
        producer.start()
        while True:
            item = pending.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    input_wait = 0.0
    step_seconds: list[float] = []
    host_to_device_seconds = 0.0
    step_digests: list[str] = []
    image_order: list[str] = []
    decoded_bytes = 0
    loss_value = float("nan")
    executed_steps = list(range(arguments.start_step, arguments.steps))

    executor = (
        ThreadPoolExecutor(
            max_workers=arguments.input_workers,
            thread_name_prefix="vision-input")
        if arguments.input_workers > 0 else None
    )
    try:
        stream = iter(batch_stream(
            executed_steps, executor, arguments.input_prefetch_depth))
        while True:
            # Timed around the CONSUMER pulling the batch, so this is the
            # interval the training thread spends blocked on input.
            input_started = time.perf_counter()
            try:
                pixels, step_digest, chosen = next(stream)
            except StopIteration:
                break
            input_wait += time.perf_counter() - input_started
            step_digests.append(step_digest)
            image_order.extend(chosen)
            decoded_bytes += pixels.numel()

            step_started = time.perf_counter()
            # Measured separately: for a decoded image batch this is a real
            # cost, and it is the part a text fixture cannot exercise.
            copy_started = time.perf_counter()
            on_device = pixels.to(device, non_blocking=False).float().div_(255.0)
            torch.cuda.synchronize()
            host_to_device_seconds += time.perf_counter() - copy_started

            optimizer.zero_grad(set_to_none=True)
            loss = model(on_device).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_started)
            loss_value = float(loss.detach())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    sorted_step_seconds = sorted(step_seconds)
    median = sorted_step_seconds[len(sorted_step_seconds) // 2]
    training_step_seconds = sum(step_seconds)
    measured_total = input_wait + training_step_seconds
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    json.dump({
        "median_step_seconds": median,
        "first_step_seconds": step_seconds[0],
        "steps_per_second": 1.0 / median if median > 0 else 0.0,
        "peak_memory_bytes": peak_memory,
        "peak_memory_kind": "process_max_rss",
        "input_wait_seconds": input_wait,
        "training_step_seconds": training_step_seconds,
        "input_wait_ratio": input_wait / measured_total if measured_total else 0.0,
        "input_pipeline": INPUT_PIPELINE,
        "host_to_device_seconds": host_to_device_seconds,
        "decoded_pixel_bytes": decoded_bytes,
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
        "batch_sequence_digest": chain_digests(step_digests),
        "step_batch_digests": step_digests,
        "image_order_digest": "sha256:" + hashlib.sha256(
            "\0".join(image_order).encode("utf-8", "surrogatepass"),
        ).hexdigest(),
        "start_step": arguments.start_step,
        "executed_steps": len(executed_steps),
        "manifest": str(manifest_path),
        "minimum_image_bytes": MINIMUM_IMAGE_BYTES,
        "maximum_image_bytes": MAXIMUM_IMAGE_BYTES,
        "image_set_size": len(images),
        "bucket": arguments.bucket,
        "phase": arguments.phase,
        "final_loss": loss_value,
    }, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
