import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

from rwkv_lab.mage_flow_expert_train import (
    MageFlowExpertTrainConfig,
    encode_domain_batch,
)
from rwkv_lab.mage_flow_optimizations import (
    FrozenEncoderCache,
    _float8_candidate_names,
    cache_coverage,
    compile_transformer_blocks,
    configure_activation_checkpointing,
)


class TinyBlock(nn.Module):
    def __init__(self, *, trainable: bool):
        super().__init__()
        self.img_mlp = nn.Module()
        self.img_mlp.proj = nn.Linear(16, 32)
        self.txt_mlp = nn.Linear(16, 32)
        self.requires_grad_(trainable)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TinyBlock(trainable=False),
                TinyBlock(trainable=True),
                TinyBlock(trainable=True),
            ]
        )
        self.checkpoint = True
        self.checkpoint_block_indices = None
        self.checkpoint_context_fn = None


def test_activation_checkpoint_modes_preserve_full_and_narrow_trainable_blocks():
    transformer = TinyTransformer()
    full = configure_activation_checkpointing(transformer, "full")
    assert full["block_indices"] == [0, 1, 2]
    assert transformer.checkpoint_block_indices is None

    narrowed = configure_activation_checkpointing(transformer, "trainable")
    assert narrowed["block_indices"] == [1, 2]
    assert transformer.checkpoint_block_indices == frozenset({1, 2})
    assert transformer.checkpoint_context_fn is None

    selective = configure_activation_checkpointing(transformer, "selective")
    assert selective["block_indices"] == [1, 2]
    assert selective["selective_saved_ops"] == [
        "aten.mm",
        "aten.addmm",
        "aten.bmm",
    ]
    assert callable(transformer.checkpoint_context_fn)

    disabled = configure_activation_checkpointing(transformer, "none")
    assert not disabled["enabled"]
    assert transformer.checkpoint_block_indices == frozenset()


def test_float8_allowlist_contains_only_trainable_image_ffn_linears():
    transformer = TinyTransformer()
    names = _float8_candidate_names(transformer)
    assert names == [
        "transformer_blocks.1.img_mlp.proj",
        "transformer_blocks.2.img_mlp.proj",
    ]


def test_regional_compile_calls_each_block_in_place():
    transformer = TinyTransformer()
    calls = []
    for index, block in enumerate(transformer.transformer_blocks):
        block.compile = lambda *, _index=index, **kwargs: calls.append(
            (_index, kwargs)
        )
    report = compile_transformer_blocks(
        transformer,
        enabled=True,
        mode="reduce-overhead",
        dynamic=True,
        backend="inductor",
    )
    assert report["block_indices"] == [0, 1, 2]
    assert calls == [
        (
            index,
            {
                "mode": "reduce-overhead",
                "dynamic": True,
                "backend": "inductor",
            },
        )
        for index in range(3)
    ]


def test_regional_compile_includes_terminal_expert_blocks():
    transformer = TinyTransformer()
    transformer.terminal_expert_blocks = nn.ModuleList(
        [TinyBlock(trainable=True), TinyBlock(trainable=True)]
    )
    calls = []
    for block in [
        *transformer.transformer_blocks,
        *transformer.terminal_expert_blocks,
    ]:
        block.compile = lambda **_kwargs: calls.append(block)
    report = compile_transformer_blocks(transformer, enabled=True)
    assert len(calls) == 5
    assert report["regions"] == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "transformer_blocks.2",
        "terminal_expert_blocks.0",
        "terminal_expert_blocks.1",
    ]


def _cache_row(image: Path, index: int = 0):
    return {
        "image": str(image),
        "image_id": f"id-{index}",
        "train_width": 512,
        "train_height": 512,
    }


def test_frozen_encoder_cache_round_trip_and_contract(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"frozen-input")
    row = _cache_row(image)
    root = tmp_path / "cache"
    cache = FrozenEncoderCache(
        root,
        mode="read_write",
        model_id="mage",
        model_revision="revision-a",
    )
    txt = torch.randn(5, 8, dtype=torch.bfloat16)
    mean = torch.randn(1, 4, 3, 2, dtype=torch.bfloat16)
    logvar = torch.randn_like(mean)
    cache.save_text("caption", "{}", 0, txt)
    cache.save_moments(row, mean, logvar)

    read_only = FrozenEncoderCache(
        root,
        mode="read_only",
        model_id="mage",
        model_revision="revision-a",
    )
    assert torch.equal(read_only.load_text("caption", "{}", 0), txt)
    observed_mean, observed_logvar = read_only.load_moments(row)
    assert torch.equal(observed_mean, mean)
    assert torch.equal(observed_logvar, logvar)
    with pytest.raises(ValueError, match="contract"):
        FrozenEncoderCache(
            root,
            mode="read_only",
            model_id="mage",
            model_revision="revision-b",
        )


def test_cache_stores_moments_and_preserves_posterior_sampling(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"frozen-input")
    row = _cache_row(image)
    cache = FrozenEncoderCache(
        tmp_path / "cache",
        mode="read_write",
        model_id="mage",
        model_revision="revision-a",
    )
    mean = torch.zeros(1, 4, 2, 2)
    logvar = torch.zeros_like(mean)
    cache.save_moments(row, mean, logvar)
    cached_mean, cached_logvar = cache.load_moments(row)
    torch.manual_seed(1)
    first = cache.sample_moments(
        cached_mean, cached_logvar, sample_posterior=True
    )
    second = cache.sample_moments(
        cached_mean, cached_logvar, sample_posterior=True
    )
    assert not torch.equal(first, second)
    assert torch.equal(
        cache.sample_moments(
            cached_mean, cached_logvar, sample_posterior=False
        ),
        mean,
    )


def test_cache_coverage_includes_cfg_null_prompt(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"frozen-input")
    row = _cache_row(image)
    cache = FrozenEncoderCache(
        tmp_path / "cache",
        mode="read_write",
        model_id="mage",
        model_revision="revision-a",
    )
    cache.save_text("caption", "{}", 0, torch.ones(2, 4))
    cache.save_moments(row, torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))
    incomplete = cache_coverage(
        cache,
        [row],
        prompts=["caption"],
        template="{}",
        drop_idx=0,
        require_null_prompt=True,
    )
    assert incomplete["missing_text"] == 1
    assert not incomplete["complete"]
    cache.save_text(" ", "{}", 0, torch.ones(1, 4))
    complete = cache_coverage(
        cache,
        [row],
        prompts=["caption"],
        template="{}",
        drop_idx=0,
        require_null_prompt=True,
    )
    assert complete["complete"]


def test_cached_batch_needs_neither_decoded_image_nor_live_encoder(
    tmp_path, monkeypatch
):
    mage_flow = ModuleType("mage_flow")
    models = ModuleType("mage_flow.models")
    utils = ModuleType("mage_flow.models.utils")
    pipeline = ModuleType("mage_flow.pipeline")
    utils.PROMPT_TEMPLATE = {
        "mage-flow": {"template": "{}", "start_idx": 0}
    }
    pipeline._encode_texts_packed = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("cached conditioning must not call Qwen")
    )
    for name, module in (
        ("mage_flow", mage_flow),
        ("mage_flow.models", models),
        ("mage_flow.models.utils", utils),
        ("mage_flow.pipeline", pipeline),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    image = tmp_path / "image.png"
    image.write_bytes(b"frozen-input")
    row = {
        **_cache_row(image),
        "domain": "photo",
        "conditioning_text": "caption",
        "conditioning_kind": "human",
        "is_captioned": True,
        "latent_tokens": 4,
    }
    cache = FrozenEncoderCache(
        tmp_path / "cache",
        mode="read_write",
        model_id="mage",
        model_revision="revision-a",
    )
    info = utils.PROMPT_TEMPLATE["mage-flow"]
    cache.save_text(
        "caption",
        info["template"],
        int(info["start_idx"]),
        torch.ones(2, 8, dtype=torch.bfloat16),
    )
    cache.save_moments(
        row,
        torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16),
    )

    class CachedOnlyModel:
        _training_encoder_cache = cache

    config = MageFlowExpertTrainConfig(
        train_manifest=str(tmp_path / "unused.jsonl"),
        output_dir=str(tmp_path / "output"),
        vae_sample_posterior=True,
    )
    flow = encode_domain_batch(
        CachedOnlyModel(),
        [row],
        [None],
        config,
        torch.device("cpu"),
        caption_dropout=0.0,
    )
    assert flow["img"].shape == (1, 4, 4)
    assert flow["txt"].shape == (1, 2, 8)
    assert flow["image_lens"] == [4]
    assert flow["text_lens"] == [2]
