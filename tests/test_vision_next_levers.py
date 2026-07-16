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
    _initialize_adapters, insert_boundary_ids, insert_visual_span,
    multimodal_loss, prepare_examples, remove_visual_span, supervised_positions,
    visual_insert_positions)


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
