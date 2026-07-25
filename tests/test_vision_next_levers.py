from pathlib import Path
from types import SimpleNamespace

import torch

from rwkv_lab.deep_vision import DeepVisionInjector, LayerMatchedVisionInjector
from rwkv_lab.generate import WorldVocab
from rwkv_lab.lookahead_module import NextLatPredictor
from rwkv_lab.moonvit import (
    MoonViTPrefixProjector, feature_cache_key, pool_features,
    valid_pooled_feature)
from rwkv_lab.vision_fusion import VisionFusionResidual
from rwkv_lab.vision_train import (
    EpochBatchSampler, _initialize_adapters, add_fusion_residual,
    insert_boundary_ids, insert_visual_span,
    multimodal_loss, prepare_examples, remove_visual_span, supervised_positions,
    visual_insert_positions)


def test_sam_global_residual_only_modifies_leading_radio_span():
    prefix = torch.ones(2, 640, 8)
    residual = torch.full((2, 256, 8), 3.0)
    fused = add_fusion_residual(prefix, residual)
    assert torch.equal(fused[:, :256], torch.full((2, 256, 8), 4.0))
    assert torch.equal(fused[:, 256:], prefix[:, 256:])


def test_sandwich_prompt_places_image_between_two_prompt_copies(tmp_path: Path):
    vocab = WorldVocab()
    rows, _ = prepare_examples(
        [{"image": tmp_path / "x.jpg", "text": "a red kite"}], vocab,
        prompt="Describe:\n", max_text_tokens=64, sandwich_prompt=True)
    row = rows[0]
    prompt = vocab.encode("Describe:\n")
    assert row["tokens"][:len(prompt) * 2] == prompt + prompt
    assert row["vision_insert"] == len(prompt)
    assert row["prompt_len"] == len(prompt) * 2
    assert supervised_positions(rows, 5, device="cpu")[0, 1] == 5 + len(prompt) * 2 - 1


def test_asymmetric_sandwich_uses_lead_then_image_then_task(tmp_path: Path):
    vocab = WorldVocab()
    rows, _ = prepare_examples(
        [{"image": tmp_path / "x.jpg", "text": "a red kite"}], vocab,
        prompt="Describe accurately:\n", max_text_tokens=64,
        sandwich_prompt=True, sandwich_lead_prompt="An image follows:\n")
    lead = vocab.encode("An image follows:\n")
    task = vocab.encode("Describe accurately:\n")
    assert rows[0]["tokens"][:len(lead) + len(task)] == lead + task
    assert rows[0]["vision_insert"] == len(lead)
    assert rows[0]["prompt_len"] == len(lead) + len(task)


def test_sampler_never_crosses_exact_radio_tile_bucket():
    keys = [1, 1, 3, 3, 3, 7, 7, 7]
    sampler = EpochBatchSampler(
        list(range(len(keys))), [10] * len(keys), batch_size=4, seed=7,
        group_keys=keys)
    seen = []
    while len(seen) < len(keys):
        batch = sampler.next_budget_batch(
            [100] * len(keys), target_tokens=1000, min_items=1, max_items=4)
        assert len({keys[index] for index in batch}) == 1
        seen.extend(batch)
    assert sorted(seen) == list(range(len(keys)))


def test_sampler_interleaves_bounded_radio_tile_bucket_runs():
    keys = [1] * 24 + [3] * 24 + [7] * 24
    sampler = EpochBatchSampler(
        list(range(len(keys))), [10] * len(keys), batch_size=2, seed=7,
        bucket_batches=2, group_keys=keys)
    run_keys = []
    for index in sampler.order:
        key = keys[index]
        if not run_keys or run_keys[-1] != key:
            run_keys.append(key)
    # Whole-bucket concatenation produces only three runs.  Bounded
    # round-robin chunks expose every exact shape throughout the epoch.
    assert len(run_keys) > 3
    assert set(run_keys[:3]) == {1, 3, 7}
    assert sorted(sampler.order) == list(range(len(keys)))


def test_arbitrary_visual_span_round_trip_and_boundaries():
    text = torch.arange(2 * 6).reshape(2, 6)
    visual = torch.full((2, 3), -1)
    starts = (2, 4)
    joined = insert_visual_span(text, visual, starts)
    assert joined.tolist() == [
        [0, 1, -1, -1, -1, 2, 3, 4, 5],
        [6, 7, 8, 9, -1, -1, -1, 10, 11],
    ]
    torch.testing.assert_close(remove_visual_span(joined, starts, 3), text)
    assert insert_boundary_ids(text, starts, 3, 99)[:, 2:5].shape == (2, 3)


def test_staged_pooling_and_projector_use_deepest_stage():
    raw = torch.randn(3, 11, 4, 1152)
    pooled = pool_features(raw, 5).squeeze(0)
    assert pooled.shape == (3, 5, 4, 1152)
    assert valid_pooled_feature(pooled, 5, 3)
    projector = MoonViTPrefixProjector(32, 5)
    torch.testing.assert_close(
        projector([pooled]), projector([pooled[-1]]), rtol=0, atol=0)


def test_staged_multiview_cache_key_does_not_alias_legacy(tmp_path: Path):
    image = tmp_path / "x.jpg"
    image.write_bytes(b"x")
    legacy = feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=64,
        vision_fingerprint="v")
    staged = feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=64,
        vision_fingerprint="v", tap_layers=(8, 17, 26),
        view_mode="full-quadrants")
    assert legacy != staged


def test_layer_matched_adapter_starts_as_noop_and_trains():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    layers = torch.nn.ModuleList([Layer(), Layer()])
    injector = LayerMatchedVisionInjector(16, (0, 1), rank=4)
    injector.install(layers)
    hidden = torch.randn(2, 9, 16, requires_grad=True)
    features = torch.randn(2, 2, 3, 4, 1152)
    with injector.use_features(features, (1, 4)):
        output = hidden
        for layer in layers:
            output = layer(output)
    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    output.sum().backward()
    assert injector.adapters["0"].up.weight.grad is not None
    injector.close()


def test_prefix_migration_interpolates_position_and_resampler_queries(tmp_path: Path):
    source = MoonViTPrefixProjector(
        32, 5, resampler_layers=1, resampler_width=16, resampler_heads=4)
    saved_args = {
        "rwkv_fingerprint": "rwkv", "moonvit_fingerprint": "moonvit",
        "prefix_tokens": 5, "max_input_patches": 1024,
        "nextlat_hidden": 1024, "loop_count": 1, "loop_index": True,
        "loop_gate_cap": 0.25, "engram": False,
        "vision_resampler_layers": 1, "vision_resampler_width": 16,
        "vision_resampler_heads": 4,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({
        "schema": 3, "step": 7, "args": saved_args,
        "projector": source.state_dict(), "nextlat": None,
        "engram": None, "loops": [],
    }, checkpoint)
    args = SimpleNamespace(**{**saved_args, "prefix_tokens": 8})
    destination = MoonViTPrefixProjector(
        32, 8, resampler_layers=1, resampler_width=16, resampler_heads=4)
    assert _initialize_adapters(
        checkpoint, projector=destination, nextlat=None, engram=None,
        wrappers=[], args=args) == 7
    assert destination.position.shape[1] == 8
    assert destination.resampler.queries.shape[1] == 8
    torch.testing.assert_close(destination.position[:, 0], source.position[:, 0])
    torch.testing.assert_close(destination.position[:, -1], source.position[:, -1])


def test_fusion_residual_is_zero_init_but_receives_gradient():
    adapter = VisionFusionResidual(32, rank=8)
    features = torch.randn(2, 5, 1792)
    output = adapter(features)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert adapter.up.weight.grad is not None


def test_selected_levers_share_one_loss_sequence_contract():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states + 0.01

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(128, 16)
            self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

        def forward(self, inputs_embeds, **_kwargs):
            value = inputs_embeds
            for layer in self.layers:
                value = layer(value)
            return SimpleNamespace(last_hidden_state=value)

    class RWKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.lm_head = torch.nn.Linear(16, 128, bias=False)

    rwkv = RWKV()
    projector = MoonViTPrefixProjector(16, 4)
    fusion = VisionFusionResidual(16, rank=8)
    nextlat = NextLatPredictor(16, hidden=8)
    deep = DeepVisionInjector(16, (1,), rank=4)
    deep.install(rwkv.model.layers)
    layered = LayerMatchedVisionInjector(16, (0, 2), rank=4)
    layered.install(rwkv.model.layers)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6]])
    rows = [{"tokens": ids[0].tolist(), "prompt_len": 4, "vision_insert": 2}]

    loss, metrics = multimodal_loss(
        rwkv, projector, None, (), ids, labels, torch.ones_like(ids, dtype=torch.bool),
        nextlat=nextlat, nextlat_weight=0.1,
        features=[torch.randn(2, 4, 4, 1152)],
        selected_positions=supervised_positions(rows, 4, device="cpu"),
        deep_vision=deep, layer_vision=layered, visual_starts=(2,),
        fusion_adapter=fusion, fusion_features=[torch.randn(4, 1792)])
    loss.backward()

    assert torch.isfinite(loss)
    assert {"nextlat_loss", "deep_vision_inj_rms", "layer_vision_inj_rms",
            "vision_fusion_rms"} <= metrics.keys()
    assert projector.project[0].weight.grad is not None
    assert fusion.up.weight.grad is not None
    assert layered.adapters["0"].up.weight.grad is not None
    assert nextlat.net[0].weight.grad is not None
    deep.close()
    layered.close()


def test_token_budget_no_longer_shrinks_as_images_grow():
    """A 13-tile image once received 46% FEWER tokens than a 12-tile one.

    tokens_per_tile halved at the threshold while tile count rose by one, so
    total visual tokens fell off a cliff exactly on the large, detailed images
    (97% of the OCR shard) where representation matters most.
    """
    from rwkv_lab.radio1d_rwkv import (
        DEFAULT_MAX_DETAIL_TILES, token_budget_is_monotone,
        tokens_per_tile_for_tile_count)

    assert token_budget_is_monotone()
    totals = [t * tokens_per_tile_for_tile_count(t)
              for t in range(1, DEFAULT_MAX_DETAIL_TILES + 2)]
    assert all(a <= b for a, b in zip(totals, totals[1:]))
    assert totals[12] > totals[11]          # the old 12 -> 13 cliff
    assert not token_budget_is_monotone(threshold=12)   # the previous default


def test_letterbox_content_box_matches_the_real_letterbox():
    """The derived content extent must equal what _letterbox actually produces."""
    from PIL import Image
    from rwkv_lab.radio1d_rwkv import (
        RADIO_TILE_SIZE, _letterbox, letterbox_content_box)

    for width, height in ((512, 512), (1920, 1080), (480, 640), (3790, 1000)):
        canvas = _letterbox(Image.new("RGB", (width, height), (255, 255, 255)),
                            RADIO_TILE_SIZE, fill=(0, 0, 0))
        pixels = canvas.convert("L").point(lambda v: 255 if v > 0 else 0)
        x0, y0, x1, y1 = pixels.getbbox()
        derived = letterbox_content_box(width, height)
        actual = (x0 / RADIO_TILE_SIZE, y0 / RADIO_TILE_SIZE,
                  x1 / RADIO_TILE_SIZE, y1 / RADIO_TILE_SIZE)
        assert all(abs(a - b) < 0.01 for a, b in zip(derived, actual)), (
            f"{width}x{height}: derived {derived} vs actual {actual}")


def test_content_boxes_recover_tile_geometry_from_source_boxes():
    """Detail crops and full-image thumbnails must both letterbox correctly."""
    import torch
    from rwkv_lab.radio1d_rwkv import (
        content_boxes_from_source, letterbox_content_box)

    # 16:9 image; thumbnail covers all of it, one detail crop covers a square.
    source = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5625, 1.0]]])
    content = content_boxes_from_source(source, torch.tensor([16 / 9]))
    expected_thumb = letterbox_content_box(1920, 1080)
    expected_square = letterbox_content_box(1080, 1080)
    assert torch.allclose(content[0, 0], torch.tensor(expected_thumb), atol=1e-4)
    assert torch.allclose(content[0, 1], torch.tensor(expected_square), atol=1e-4)
    # A square source needs no bars at all.
    square = content_boxes_from_source(source[:, :1], torch.tensor([1.0]))
    assert torch.allclose(square[0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]), atol=1e-6)


def test_bridge_content_geometry_is_a_no_op_at_initialization():
    """Supplying letterbox geometry must not perturb a resumed checkpoint."""
    import torch
    from rwkv_lab.radio1d_rwkv import RadioRWKVBridge

    bridge = RadioRWKVBridge(hidden_size=32, rank=8, tokens_per_tile=4, max_tiles=3)
    features = torch.randn(2, 2, 4, 32)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0]]]).repeat(2, 1, 1)
    roles = torch.zeros(2, 2, dtype=torch.long)
    without = bridge(features, boxes, roles).embeddings
    with_geometry = bridge(features, boxes, roles,
                           image_aspect=torch.tensor([1.5, 0.5])).embeddings
    assert torch.allclose(without, with_geometry, atol=1e-6)


def test_optimizer_moments_survive_a_grown_parameter_group():
    """Adding a zero-init module must not discard 43k steps of Adam state."""
    import torch
    from rwkv_lab.vision_train import _load_optimizer_with_grown_groups

    old_params = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(3)]
    old = torch.optim.AdamW([{"params": old_params, "name": "bridge"},
                             {"params": [torch.nn.Parameter(torch.randn(2))],
                              "name": "head"}], lr=1e-3)
    for parameter in old_params:
        parameter.grad = torch.ones_like(parameter)
    list(old.param_groups[1]["params"])[0].grad = torch.ones(2)
    old.step()
    saved = old.state_dict()

    # Same topology, but the bridge group gained two appended parameters.
    grown_params = [torch.nn.Parameter(p.detach().clone()) for p in old_params]
    grown_params += [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(2)]
    grown = torch.optim.AdamW([{"params": grown_params, "name": "bridge"},
                               {"params": [torch.nn.Parameter(torch.randn(2))],
                                "name": "head"}], lr=1e-3)
    _load_optimizer_with_grown_groups(grown, saved, expected_growth=2)

    restored = grown.state_dict()["state"]
    assert len(restored) == len(saved["state"])          # every moment kept
    for old_index, new_index in zip(sorted(saved["state"]), sorted(restored)):
        assert torch.allclose(saved["state"][old_index]["exp_avg"],
                              restored[new_index]["exp_avg"])
    # The head group's index must have shifted by the growth, not been dropped.
    assert grown.state_dict()["param_groups"][1]["params"] == [5]


def test_grown_group_migration_refuses_unexpected_topology_changes():
    import pytest
    import torch
    from rwkv_lab.vision_train import _load_optimizer_with_grown_groups

    params = [torch.nn.Parameter(torch.randn(2))]
    saved = torch.optim.AdamW([{"params": params, "name": "a"}], lr=1e-3).state_dict()
    bigger = torch.optim.AdamW(
        [{"params": params + [torch.nn.Parameter(torch.randn(2))], "name": "a"}],
        lr=1e-3)
    with pytest.raises(ValueError, match="expected"):
        _load_optimizer_with_grown_groups(bigger, saved, expected_growth=7)
    with pytest.raises(ValueError, match="group count"):
        _load_optimizer_with_grown_groups(
            bigger, {"param_groups": [], "state": {}}, expected_growth=1)


def test_token_threshold_is_a_resumable_budget_change():
    """Changing it must be accepted under --allow-batch-resize, not fatal."""
    import argparse
    from rwkv_lab.radio1d_rwkv import DEFAULT_ADAPTIVE_TOKEN_THRESHOLD
    from rwkv_lab.vision_train import _budget_resume_differences

    args = argparse.Namespace(
        batch=1, max_batch=4, min_batch=0, target_batch_tokens=4096,
        loop_token_budget_scale=1.0, radio_adaptive_complexity=True,
        radio_complexity_budget_ratio=0.75, radio_complexity_token_quantum=16,
        radio_adaptive_token_threshold=DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)
    saved = {"batch": 1, "max_batch": 4, "min_batch": 0,
             "target_batch_tokens": 4096, "loop_token_budget_scale": 1.0,
             "radio_adaptive_complexity": True,
             "radio_complexity_budget_ratio": 0.75,
             "radio_complexity_token_quantum": 16,
             "radio_adaptive_token_threshold": 12}
    assert _budget_resume_differences(saved, args) == [
        "radio_adaptive_token_threshold"]


def test_bridge_growth_stays_appended_for_moment_migration():
    """_load_optimizer_with_grown_groups maps moments POSITIONALLY.

    That is only valid while new parameters are appended. Registering a module
    earlier in RadioRWKVBridge.__init__ would shift every later position and
    silently reassign 43k steps of Adam state to the wrong tensors, so pin the
    ordering here rather than discovering it in a run.
    """
    from rwkv_lab.radio1d_rwkv import RadioFeatureProjector

    names = [name for name, _ in RadioFeatureProjector().named_parameters()]
    new = [i for i, name in enumerate(names) if "content_embedding" in name]
    assert new, "content_embedding disappeared from the bridge"
    assert new == list(range(len(names) - len(new), len(names))), (
        f"content_embedding must stay last; got positions {new} of {len(names)}")
