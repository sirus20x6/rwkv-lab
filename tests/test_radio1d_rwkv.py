import torch
from PIL import Image

from rwkv_lab.radio1d_rwkv import (
    DEFAULT_MAX_DETAIL_TILES,
    RADIO_HIDDEN_SIZE,
    RADIO_COMPACT_TOKENS_PER_TILE,
    RADIO_TOKENS_PER_TILE,
    RadioPrefixInjector,
    RadioRWKVBridge,
    RadioFeatureProjector,
    RadioVisualPrefix,
    build_radio_tiles,
    choose_detail_grid,
    estimate_context,
    extract_radio_global_tokens,
    pad_radio_features,
    pad_tile_metadata,
    tile_metadata,
    tiles_to_tensor,
    tokens_per_tile_for_tile_count,
    adaptive_tokens_per_tile,
    allocate_radio_token_counts,
)


def test_small_image_is_not_duplicated_or_stretched():
    image = Image.new("RGB", (320, 160), "red")
    tiles = build_radio_tiles(image)
    assert len(tiles) == 1
    assert tiles[0].role == "thumbnail"
    assert tiles[0].image.size == (512, 512)
    # Aspect preservation gives a 256px-tall red image with letterbox bars.
    assert tiles[0].image.getpixel((256, 10)) == (0, 0, 0)
    assert tiles[0].image.getpixel((256, 256)) == (255, 0, 0)


def test_large_image_gets_thumbnail_and_generous_row_major_details():
    image = Image.new("RGB", (3840, 2160), "white")
    tiles = build_radio_tiles(image)
    rows, columns = choose_detail_grid(3840, 2160)
    assert 13 < rows * columns <= DEFAULT_MAX_DETAIL_TILES
    assert len(tiles) == rows * columns + 1
    assert tiles[0].role == "thumbnail"
    assert all(tile.role == "detail" for tile in tiles[1:])
    assert [(tile.row, tile.column) for tile in tiles[1:]] == [
        (row, column) for row in range(rows) for column in range(columns)]
    assert all(tile.image.size == (512, 512) for tile in tiles)
    # Corners are covered and overlap never leaves the normalized source.
    boxes = [tile.source_box for tile in tiles[1:]]
    assert boxes[0][0:2] == (0.0, 0.0)
    assert boxes[-1][2:4] == (1.0, 1.0)
    assert all(0 <= coordinate <= 1 for box in boxes for coordinate in box)


def test_large_images_keep_the_full_native_token_budget():
    """Tokens scale with image size; the compact tier is opt-in, not default.

    The old default halved tokens-per-tile above 12 tiles, so a 13-tile image
    received 46% fewer TOTAL tokens than a 12-tile one. The mechanism still
    works when a threshold is passed explicitly — it is just no longer on.
    """
    estimate = estimate_context(49, 2048)
    assert estimate.visual_tokens == 49 * 256
    assert estimate.total_tokens == 49 * 256 + 2048
    assert tokens_per_tile_for_tile_count(12) == 256
    assert tokens_per_tile_for_tile_count(13) == 256
    assert tokens_per_tile_for_tile_count(49) == 256
    # The tier itself is intact for callers that explicitly ask for it.
    assert tokens_per_tile_for_tile_count(
        13, threshold=12) == RADIO_COMPACT_TOKENS_PER_TILE


def test_adaptive_budget_is_quantized_and_bounded():
    assert adaptive_tokens_per_tile(256, ratio=.75, quantum=16) == 192
    assert adaptive_tokens_per_tile(128, ratio=.75, quantum=16) == 96


def test_complex_tile_receives_more_tokens_under_fixed_total():
    torch.manual_seed(3)
    # The first tile is nearly redundant; the second has diverse token
    # directions and a strong early-to-tail change.
    simple = torch.ones(1, 64, 32)
    complex_tile = torch.randn(1, 64, 32)
    features = torch.cat((simple, complex_tile), dim=0)
    counts = allocate_radio_token_counts(
        features, torch.tensor([1, 1]), ratio=.75, quantum=8)
    assert counts.sum().item() == 2 * 48
    assert counts[1] > counts[0]
    assert all(int(value) % 8 == 0 for value in counts)


def test_adaptive_projector_packs_variable_tiles_without_padding():
    torch.manual_seed(4)
    bridge = RadioRWKVBridge(hidden_size=16, rank=4,
                             tokens_per_tile=16, max_tiles=3)
    projector = RadioFeatureProjector(
        bridge, adaptive_complexity=True,
        complexity_budget_ratio=.75, complexity_token_quantum=4)
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., .5, 1.]])
    roles = torch.tensor([0, 1])
    samples = [
        (torch.randn(2, 16, 16), boxes, roles),
        (torch.randn(2, 16, 16), boxes, roles),
    ]
    output = projector(samples)
    assert output.shape == (2, 24, 16)
    assert projector.last_token_counts is not None
    assert projector.last_token_counts.sum(dim=1).tolist() == [24, 24]
    assert projector.last_token_counts.min() >= 4
    assert projector.last_token_counts.max() <= 16


def test_variable_tile_batch_padding_has_explicit_mask():
    one = torch.ones(2, 256, 2560, dtype=torch.bfloat16)
    two = torch.ones(5, 256, 2560, dtype=torch.bfloat16)
    values, mask = pad_radio_features([one, two])
    assert values.shape == (2, 5, 256, 2560)
    assert mask.tolist() == [[True, True, False, False, False],
                             [True, True, True, True, True]]
    assert not values[0, 2:].bool().any()


def test_bridge_preserves_all_tokens_and_zeroes_only_padding():
    torch.manual_seed(1)
    bridge = RadioRWKVBridge(hidden_size=16, rank=4,
                             tokens_per_tile=8, max_tiles=4)
    features = torch.randn(2, 3, 8, 16)
    boxes = torch.tensor([
        [[0, 0, 1, 1], [0, 0, .5, 1], [.5, 0, 1, 1]],
        [[0, 0, 1, 1], [0, 0, .5, 1], [0, 0, 0, 0]],
    ], dtype=torch.float32)
    roles = torch.tensor([[0, 1, 1], [0, 1, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    result = bridge(features, boxes, roles, mask)
    assert result.embeddings.shape == (2, 24, 16)
    assert result.mask.shape == (2, 24)
    assert result.tile_count.tolist() == [3, 2]
    assert result.mask.sum(dim=1).tolist() == [24, 16]
    assert not result.embeddings[1, 16:].bool().any()
    assert result.embeddings[0].bool().any()


def test_real_bridge_is_small_and_does_not_change_width_or_budget():
    bridge = RadioRWKVBridge()
    trainable = sum(parameter.numel() for parameter in bridge.parameters())
    assert 2_000_000 < trainable < 3_000_000
    assert bridge.hidden_size == RADIO_HIDDEN_SIZE
    assert bridge.tokens_per_tile == RADIO_TOKENS_PER_TILE


def test_bridge_zero_gate_is_noop_but_not_gradient_dead():
    bridge = RadioRWKVBridge(hidden_size=16, rank=4,
                             tokens_per_tile=8, max_tiles=2)
    features = torch.randn(1, 1, 8, 16)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    roles = torch.zeros(1, 1, dtype=torch.long)
    output = bridge(features, boxes, roles).embeddings
    torch.testing.assert_close(output, bridge.norm(features).flatten(1, 2))
    output.sum().backward()
    assert bridge.gate.grad is not None and bridge.gate.grad.abs() > 0


def test_tile_metadata_tracks_roles_and_normalized_boxes():
    tiles = build_radio_tiles(Image.new("RGB", (1024, 1024), "blue"))
    boxes, roles = tile_metadata(tiles)
    assert boxes.shape == (len(tiles), 4)
    assert roles[0].item() == 0
    assert roles[1:].eq(1).all()


def test_metadata_and_pixels_batch_without_geometry_loss():
    first = build_radio_tiles(Image.new("RGB", (320, 160), "blue"))
    second = build_radio_tiles(Image.new("RGB", (1024, 1024), "green"))
    boxes, roles, mask = pad_tile_metadata([first, second])
    assert boxes.shape == (2, len(second), 4)
    assert roles.shape == mask.shape == (2, len(second))
    assert mask[0].sum() == 1 and mask[1].sum() == len(second)
    pixels = tiles_to_tensor(first)
    assert pixels.shape == (1, 3, 512, 512)
    assert 0 <= pixels.min() <= pixels.max() <= 1


class _FakeEncoder(torch.nn.Module):
    def __init__(self, mask_value=True):
        super().__init__()
        self.mask_value = mask_value

    def forward_encoder(self, pixels, num_tokens):
        batch = pixels.shape[0]
        return {
            "global_tokens": torch.ones(batch, num_tokens, 2560),
            "global_token_mask": torch.full(
                (batch, num_tokens), self.mask_value, dtype=torch.bool),
            # Deliberately present: the extractor must not return this sequence.
            "encoder": torch.zeros(batch, num_tokens + 10, 2560),
        }


class _FakeWrapper:
    def __init__(self, mask_value=True):
        self.input_conditioner = torch.nn.Identity()
        self.model = _FakeEncoder(mask_value)


class _FakeRADIO(torch.nn.Module):
    def __init__(self, mask_value=True):
        super().__init__()
        self.radio_model = _FakeWrapper(mask_value)


def test_extractor_returns_256_globals_and_not_internal_prefix_tokens():
    tokens, mask = extract_radio_global_tokens(
        _FakeRADIO(), torch.zeros(2, 3, 512, 512))
    assert tokens.shape == (2, 256, 2560)
    assert mask.shape == (2, 256)


def test_extractor_rejects_an_incomplete_global_budget():
    import pytest
    with pytest.raises(RuntimeError, match="all requested"):
        extract_radio_global_tokens(
            _FakeRADIO(False), torch.zeros(1, 3, 512, 512))


def test_aligned_radio_prefix_reinjects_at_each_selected_rwkv_depth():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    layers = torch.nn.ModuleList([Layer() for _ in range(5)])
    injector = RadioPrefixInjector(
        hidden_size=16, layer_indices=(1, 3, 4), rank=4, tokens_per_tile=8)
    injector.install(layers)
    embeddings = torch.randn(2, 16, 16, requires_grad=True)
    prefix = RadioVisualPrefix(
        embeddings=embeddings,
        mask=torch.ones(2, 16, dtype=torch.bool),
        tile_count=torch.tensor([2, 2]))
    hidden = torch.randn(2, 23, 16, requires_grad=True)
    with injector.use_aligned_prefix(prefix, starts=(2, 2)):
        output = hidden
        for layer in layers:
            output = layer(output)
    # Every residual starts at zero, so enabling the lever is an exact no-op.
    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    output.sum().backward()
    assert all(injector.adapters[str(site)].up.weight.grad is not None
               for site in (1, 3, 4))
    assert set(injector.injection_rms_by_layer()) == {1, 3, 4}
    injector.close()


def test_radio_reinjection_refuses_padded_interior_visual_steps():
    import pytest
    injector = RadioPrefixInjector(
        hidden_size=8, layer_indices=(0,), rank=2, tokens_per_tile=4)
    padded = RadioVisualPrefix(
        embeddings=torch.randn(1, 8, 8),
        mask=torch.tensor([[True, True, True, True, False, False, False, False]]),
        tile_count=torch.tensor([1]))
    with pytest.raises(ValueError, match="without padding"):
        with injector.use_aligned_prefix(padded):
            pass
