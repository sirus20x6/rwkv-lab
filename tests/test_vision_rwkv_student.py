import torch
from PIL import Image

from rwkv_lab.vision_rwkv_student import (
    VisionRWKVStudent,
    compact_config,
    spatial_orders,
)
from rwkv_lab.vision_rwkv_student_train import image_tensor
from rwkv_lab.vision_teacher_compressor import EpochBatchSampler


def test_spatial_scan_orders_are_permutations():
    for order in spatial_orders(4):
        assert sorted(order.tolist()) == list(range(16))


def test_pixel_student_emits_canonical_and_native_contracts():
    model = VisionRWKVStudent(compact_config()).eval()
    pixels = torch.randn(2, 3, 64, 64)
    geometry = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.2, 0.8]])
    with torch.no_grad():
        canonical, native = model(pixels, geometry)
    assert canonical.shape == (2, 2, 32)
    assert native.shape == (2, 1, 64)
    assert torch.isfinite(canonical).all()
    assert torch.isfinite(native).all()


def test_calibrated_native_head_loads_exactly():
    model = VisionRWKVStudent(compact_config())
    state = {
        "output_norm": model.native_norm.state_dict(),
        "output_projection": model.native_projection.state_dict(),
        "output_position": model.native_position.detach().clone(),
    }
    expected = model.native_position.detach().clone()
    with torch.no_grad():
        model.native_position.zero_()
    model.load_native_head(state)
    assert torch.equal(model.native_position, expected)


def test_letterbox_preserves_aspect_metadata(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (300, 100), (255, 0, 0)).save(path)
    pixels, geometry = image_tensor(path, 64)
    assert pixels.shape == (3, 64, 64)
    assert geometry[0] > 1.0
    assert geometry[1] == 1.0
    assert 0.3 < float(geometry[2]) < 0.35


def test_batch_resize_keeps_exact_unseen_sampler_cursor():
    original = EpochBatchSampler(20, 1, 7)
    original.consumed(3)
    state = original.state_dict()
    expected = original.order()[3:7]
    resized = EpochBatchSampler(20, 4, 7)
    state["batch_size"] = 4
    resized.load_state_dict(state)
    assert next(iter(resized)) == expected
