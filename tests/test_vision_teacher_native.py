import hashlib

import pytest
import torch

from rwkv_lab.vision_teacher_native import (
    GridGeometry, NativeCacheReceipt, NativeEntryMetadata,
    load_native_entry, preprocessing_fingerprint, save_native_entry,
    sha256_file, validate_native_tensors)


DIGEST = "a" * 64


def geometry(grid=(32, 32)):
    return GridGeometry(
        source_hw=(720, 1280), source_yxyx=(0, 0, 720, 1280),
        processed_hw=(512, 512), content_yxyx=(0, 0, 512, 512),
        grid_hw=grid)


def metadata(teacher, geometries=None):
    return NativeEntryMetadata(
        sample_id="sample-1", teacher=teacher, source_path="/images/one.png",
        source_sha256=DIGEST, teacher_fingerprint="b" * 64,
        preprocessing_fingerprint="c" * 64,
        geometries=tuple(geometries or [geometry()]))


def test_native_siglip_roundtrip_is_atomic_and_strict(tmp_path):
    target = tmp_path / "entry.safetensors"
    tensors = {
        "dense": torch.randn(32, 32, 1152, dtype=torch.bfloat16),
        "summary": torch.randn(1152, dtype=torch.bfloat16),
    }
    save_native_entry(target, metadata("siglip2"), tensors)
    loaded_metadata, loaded = load_native_entry(target)
    assert loaded_metadata == metadata("siglip2")
    assert loaded.keys() == tensors.keys()
    assert torch.equal(loaded["dense"], tensors["dense"])
    assert not list(tmp_path.glob(".*.tmp"))


def test_old_pooled_fusion_payload_is_rejected():
    with pytest.raises(ValueError, match="dense and summary"):
        validate_native_tensors(
            metadata("siglip2"),
            {"fusion": torch.randn(128, 2176, dtype=torch.bfloat16)})


def test_full_sam_grid_is_required():
    sam_metadata = metadata("sam", [geometry((64, 64))])
    validate_native_tensors(sam_metadata, {
        "dense": torch.randn(64, 64, 256, dtype=torch.float16)})
    with pytest.raises(ValueError, match="\[64,64,256\]"):
        validate_native_tensors(sam_metadata, {
            "dense": torch.randn(128, 256, dtype=torch.float16)})


def test_moonvit_preserves_each_tap_view_and_four_subtokens():
    views = [geometry((8, 12)), geometry((7, 9))]
    values = {
        f"tap_{tap}.view_{view}": torch.randn(
            *item.grid_hw, 4, 1152, dtype=torch.bfloat16)
        for view, item in enumerate(views) for tap in (8, 17, 26)
    }
    validate_native_tensors(metadata("moonvit", views), values)
    values["tap_17.view_1"] = values["tap_17.view_1"].mean(dim=2)
    with pytest.raises(ValueError, match="\[H,W,4,1152\]"):
        validate_native_tensors(metadata("moonvit", views), values)


def test_geometry_rejects_padding_outside_processed_image():
    bad = GridGeometry(
        source_hw=(10, 20), source_yxyx=(0, 0, 10, 20),
        processed_hw=(16, 16), content_yxyx=(0, 0, 17, 16), grid_hw=(1, 1))
    with pytest.raises(ValueError, match="content_yxyx"):
        bad.validate()


def test_fingerprints_cover_bytes_and_canonical_preprocessing(tmp_path):
    source = tmp_path / "image"
    source.write_bytes(b"pixels")
    assert sha256_file(source) == hashlib.sha256(b"pixels").hexdigest()
    assert preprocessing_fingerprint({"b": 2, "a": 1}) == \
        preprocessing_fingerprint({"a": 1, "b": 2})


def test_receipt_deduplicates_completed_samples_and_resumes(tmp_path):
    target = tmp_path / "receipt.json"
    receipt = NativeCacheReceipt(
        teacher="sam", teacher_fingerprint="a" * 64,
        preprocessing_fingerprint="b" * 64,
        manifest_fingerprint="c" * 64, completed=[])
    receipt.mark_completed(["one", "two", "one"])
    receipt.save(target)
    loaded = NativeCacheReceipt.load(target)
    assert loaded.completed == ["one", "two"]
