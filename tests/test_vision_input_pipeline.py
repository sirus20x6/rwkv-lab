"""The vision fixture must measure real decode, or fail.

A benchmark fixture that quietly substitutes generated frames when its corpus
is absent reports decode timings for work it never did. That is the illusion
this fixture exists to remove, so the absence of a corpus has to be a failure
rather than a fallback, and the pipeline label has to come from the code that
does the decoding rather than from a hopeful string in the matrix.

These run without CUDA and without an image corpus: they exercise the manifest
loader, the declaration, and the runner's fixture selection. The timings
themselves are a host measurement and belong in a receipt, not in a unit test.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "scripts/run_benchmark_fixture.py"
WORKLOAD_DIRECTORY = REPOSITORY / "scripts/benchmark_workloads"
VISION_WORKLOAD = WORKLOAD_DIRECTORY / "vision_corpus_image_step.py"
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"
FIXTURE_ID = "vision.multimodal-student"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution: the workload defines a dataclass, and
    # dataclasses resolves field types through sys.modules[cls.__module__].
    # Without this, collection dies with a bare AttributeError on NoneType.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(WORKLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(WORKLOAD_DIRECTORY))

workload = _load("vision_corpus_image_step", VISION_WORKLOAD)
benchmark_runner = _load("run_benchmark_fixture", RUNNER)


@pytest.fixture
def vision_fixture() -> dict:
    document = json.loads(MATRIX.read_text(encoding="utf-8"))
    for entry in document["fixtures"]:
        if entry["id"] == FIXTURE_ID:
            return entry
    raise AssertionError(f"{FIXTURE_ID} is missing from the benchmark matrix")


def test_the_matrix_declares_the_pipeline_the_workload_implements(
        vision_fixture):
    """The label must come from the decoder, not from a hopeful string.

    If these two drift, a receipt names a pipeline nothing runs -- which is
    exactly how a synthetic fixture ends up wearing a real one's label.
    """
    assert vision_fixture["input_pipeline"] == workload.INPUT_PIPELINE


def test_the_declared_pipeline_is_not_synthetic(vision_fixture):
    declared = vision_fixture["input_pipeline"]
    assert "synthetic" not in declared
    # It has to name what it actually does, or the declaration is decoration.
    for expected in ("decode", "resize", "manifest"):
        assert expected in declared, (
            f"{declared!r} does not name {expected}, so it does not describe "
            f"the work the fixture performs")


def test_the_runner_selects_the_vision_workload(vision_fixture):
    selected = benchmark_runner.workload_for_fixture(vision_fixture)
    assert selected == VISION_WORKLOAD, (
        "the vision fixture would run the generic accelerator workload, whose "
        "input is synthetic")


def test_a_missing_manifest_fails_rather_than_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError) as raised:
        workload.load_manifest(tmp_path / "absent.jsonl")
    assert "no synthetic fallback" in str(raised.value)


def test_a_manifest_of_unusable_images_fails_rather_than_running_empty(
        tmp_path):
    """An empty corpus is not an empty-but-fine corpus.

    Silently proceeding with nothing admitted would produce a receipt whose
    input wait is near zero and whose label still says real decode.
    """
    manifest = tmp_path / "images.jsonl"
    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"\0" * 1024)  # inside no band, and not an image
    manifest.write_text(json.dumps({"image": str(tiny)}) + "\n")
    with pytest.raises(RuntimeError, match="unusable"):
        workload.load_manifest(manifest)


def test_decode_work_is_bounded_by_pixels_not_only_by_bytes():
    """Encoded size does not bound decode cost, so pixels are bounded too.

    The corpus this was built against contains a 197-megapixel image inside the
    200 KiB..4 MiB band. Bounding bytes alone would have let one image dominate
    any batch it landed in -- and PIL refuses to open it at all.
    """
    assert workload.MAXIMUM_IMAGE_PIXELS < 197_942_589, (
        "the pixel ceiling must exclude the decompression-bomb-shaped images "
        "that a byte band admits")
    assert workload.MINIMUM_IMAGE_PIXELS > 0


def test_the_size_band_is_ordered_and_nonempty():
    assert 0 < workload.MINIMUM_IMAGE_BYTES < workload.MAXIMUM_IMAGE_BYTES
    assert 0 < workload.MINIMUM_IMAGE_PIXELS < workload.MAXIMUM_IMAGE_PIXELS


def test_a_non_square_bucket_is_refused_rather_than_distorted():
    """This fixture resizes to a square edge; a rectangular bucket would be
    silently distorted, which would make its timings describe other work."""
    with pytest.raises(SystemExit, match="non-square"):
        workload._shape("448x224xbatch8")


def test_the_declared_buckets_parse():
    document = json.loads(MATRIX.read_text(encoding="utf-8"))
    entry = next(e for e in document["fixtures"] if e["id"] == FIXTURE_ID)
    for bucket in entry["shape_buckets"]:
        edge, batch = workload._shape(bucket)
        assert edge > 0 and batch > 0
