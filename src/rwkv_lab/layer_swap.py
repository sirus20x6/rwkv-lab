"""
Swap GQA attention modules for MLA in a hybrid model (e.g. Qwen3-Next) that
interleaves full-attention layers with linear-attention (Gated DeltaNet) layers.
Only the full-attention layers are converted; DeltaNet layers stay untouched.

Usage sketch:

    from transformers import AutoModelForCausalLM
    from .svd_init import GQAConfig, MLAConfig
    from .layer_swap import (
        find_full_attn_layers, convert_layer_to_mla, freeze_except_mla,
    )

    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)

    # Predicate: identify full-attention layers. For Qwen3-Next this is layers
    # whose self_attn is the full-attention class, not the DeltaNet class.
    def is_full_attn(layer):
        return type(layer.self_attn).__name__ not in ("Qwen3NextGatedDeltaNet",)

    idx = find_full_attn_layers(model, is_full_attn)
    mla_cfg = MLAConfig(
        hidden_size=model.config.hidden_size,
        num_heads=model.config.num_attention_heads,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,       # must sum to original head_dim
        v_head_dim=model.config.head_dim,
        kv_lora_rank=512,
    )
    new_modules = [
        convert_layer_to_mla(model, i, mla_cfg, layers_path="model.layers")
        for i in idx
    ]
    freeze_except_mla(model, new_modules)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .svd_init import GQAConfig, MLAConfig, gqa_to_mla_svd
from .mla_module import MLAAttention
from .rwkv8_deltanet import (
    RWKV8ChannelMixDeltaNet,
    RWKV8TimeMixDeltaNet,
    rwkv8_from_config,
    rwkv8_timemix_from_config,
)


def _resolve(obj: nn.Module, dotted: str) -> nn.Module:
    for attr in dotted.split("."):
        obj = getattr(obj, attr)
    return obj


def find_full_attn_layers(
    model: nn.Module,
    is_full_attn: Callable[[nn.Module], bool],
    layers_path: str = "model.layers",
) -> list[int]:
    """Return indices of decoder layers where `is_full_attn(layer)` is True."""
    layers = _resolve(model, layers_path)
    return [i for i, layer in enumerate(layers) if is_full_attn(layer)]


def find_linear_attn_layers(
    model: nn.Module,
    layers_path: str = "model.language_model.layers",
    attn_attr: str = "linear_attn",
) -> list[int]:
    """Return decoder-layer indices that expose a DeltaNet-style linear mixer."""
    layers = _resolve(model, layers_path)
    return [i for i, layer in enumerate(layers) if hasattr(layer, attn_attr)]


def _infer_gqa_config(attn: nn.Module, hidden_size: int) -> GQAConfig:
    """Infer a GQAConfig from a standard HF attention module.

    Looks for common attribute names (num_heads / num_attention_heads,
    num_key_value_heads, head_dim). Raises if nothing matches — pass an
    explicit GQAConfig to convert_layer_to_mla in that case.
    """
    num_q = (
        getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
    )
    num_kv = (
        getattr(attn, "num_key_value_heads", None)
        or getattr(attn, "num_kv_heads", None)
    )
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and num_q is not None:
        head_dim = hidden_size // num_q
    if not (num_q and num_kv and head_dim):
        raise RuntimeError(
            f"could not infer GQA config from {type(attn).__name__}; "
            f"pass gqa_cfg= explicitly"
        )
    return GQAConfig(
        hidden_size=hidden_size,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
    )


def gqa_config_from_hf(hf_text_cfg: dict) -> GQAConfig:
    """Build a GQAConfig from an HF text-model config dict. Handles Qwen3.5-MoE
    extras: partial_rotary_factor, attn_output_gate, (q|k)_norm presence."""
    head_dim = hf_text_cfg.get("head_dim") or (
        hf_text_cfg["hidden_size"] // hf_text_cfg["num_attention_heads"]
    )
    return GQAConfig(
        hidden_size=hf_text_cfg["hidden_size"],
        num_q_heads=hf_text_cfg["num_attention_heads"],
        num_kv_heads=hf_text_cfg["num_key_value_heads"],
        head_dim=head_dim,
        has_output_gate=bool(hf_text_cfg.get("attn_output_gate", False)),
        # Qwen3.5-MoE has q_norm/k_norm unconditionally when attn_output_gate=True;
        # treat them as part of the same feature. If you have a model with qk_norm
        # but no gate, pass gqa_cfg= explicitly.
        has_qk_norm=bool(hf_text_cfg.get("attn_output_gate", False)),
        rope_position=(
            "first"
            if (
                hf_text_cfg.get("partial_rotary_factor")
                or hf_text_cfg.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)
            ) < 1.0
            else "last"
        ),
    )


def convert_layer_to_mla(
    model: nn.Module,
    layer_idx: int,
    mla_cfg: MLAConfig,
    *,
    gqa_cfg: Optional[GQAConfig] = None,
    layers_path: str = "model.language_model.layers",
    attn_attr: str = "self_attn",
    softmax_scale: Optional[float] = None,
    attention_dropout: float = 0.0,
) -> MLAAttention:
    """Replace layers[layer_idx].<attn_attr> with an SVD-initialized MLA module."""
    layers = _resolve(model, layers_path)
    decoder_layer = layers[layer_idx]
    old = getattr(decoder_layer, attn_attr)

    if gqa_cfg is None:
        gqa_cfg = _infer_gqa_config(old, mla_cfg.hidden_size)

    sd = old.state_dict()
    mla_sd = gqa_to_mla_svd(sd, gqa_cfg, mla_cfg)

    mla = MLAAttention(
        hidden_size=mla_cfg.hidden_size,
        num_heads=mla_cfg.num_heads,
        qk_nope_head_dim=mla_cfg.qk_nope_head_dim,
        qk_rope_head_dim=mla_cfg.qk_rope_head_dim,
        v_head_dim=mla_cfg.v_head_dim,
        kv_lora_rank=mla_cfg.kv_lora_rank,
        q_lora_rank=mla_cfg.q_lora_rank,
        softmax_scale=softmax_scale,
        attention_dropout=attention_dropout,
        use_latent_norm=False,
        has_output_gate=mla_cfg.has_output_gate,
        has_qk_norm=mla_cfg.has_qk_norm,
        rope_position=gqa_cfg.rope_position,
        num_kv_rope_heads=mla_cfg.num_kv_rope_heads,
        layer_idx=layer_idx,
    )
    missing, unexpected = mla.load_state_dict(mla_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys loading MLA state dict: {unexpected}")

    example_param = next(old.parameters())
    mla = mla.to(device=example_param.device, dtype=example_param.dtype)

    setattr(decoder_layer, attn_attr, mla)
    return mla


def convert_deltanet_layer_to_rwkv8(
    model: nn.Module,
    layer_idx: int,
    *,
    mode: str = "channelmix",
    layers_path: str = "model.language_model.layers",
    attn_attr: str = "linear_attn",
    ffn_hidden_size: Optional[int] = None,
    init_output_scale: float = 1e-3,
    timemix_num_heads: int = 64,
    timemix_head_size: int = 64,
    timemix_depth_n_layer: Optional[int] = None,
    timemix_decay_cap_delta: float = 0.0,
    timemix_allow_neg_eigval: bool = False,
) -> nn.Module:
    """Replace layers[layer_idx].linear_attn with an RWKV-8 module.

    ``mode`` selects which RWKV-8 component to install:
      * ``"channelmix"`` (legacy) — ``RWKV8ChannelMixDeltaNet``, the cheap
        FFN-style stand-in. Useful as a smallest-possible probe.
      * ``"timemix"`` — ``RWKV8TimeMixDeltaNet``, the BlinkDL ``RWKV_Tmix_x070``
        port. Inherits compatible weights from the original DeltaNet
        (``out_proj`` → ``output``, V slice of ``in_proj_qkv`` → ``value``)
        before instantiation; remaining params keep BlinkDL paper init.

    The same ``_save_key`` namespace (``rwkv8_layer_{idx}``) is reused for
    both modes; resume must check the installed class matches the
    checkpoint's, otherwise state-dict keys won't align.
    """
    if mode not in ("channelmix", "timemix"):
        raise ValueError(f"unknown rwkv8 swap mode: {mode!r}")

    layers = _resolve(model, layers_path)
    decoder_layer = layers[layer_idx]
    old = getattr(decoder_layer, attn_attr)

    example_param = next(old.parameters(), None)
    if example_param is None:
        example_param = next(model.parameters())

    if mode == "channelmix":
        rwkv: nn.Module = rwkv8_from_config(
            model.config,
            layer_idx=layer_idx,
            ffn_hidden_size=ffn_hidden_size,
            init_output_scale=init_output_scale,
        )
    else:  # timemix
        # Capture DeltaNet state_dict on CPU before we replace the module so
        # init_from_deltanet can copy out_proj / V-slice into the fresh module.
        deltanet_sd = {k: v.detach().cpu() for k, v in old.state_dict().items()}
        rwkv = rwkv8_timemix_from_config(
            model.config,
            layer_idx=layer_idx,
            init_from_deltanet=deltanet_sd,
            num_heads=timemix_num_heads,
            head_size=timemix_head_size,
            depth_n_layer=timemix_depth_n_layer,
            decay_cap_delta=timemix_decay_cap_delta,
            allow_neg_eigval=timemix_allow_neg_eigval,
        )

    rwkv = rwkv.to(device=example_param.device, dtype=example_param.dtype)
    rwkv._save_key = f"rwkv8_layer_{layer_idx}"
    rwkv._swap_mode = mode
    setattr(decoder_layer, attn_attr, rwkv)
    return rwkv


def freeze_except_mla(model: nn.Module, mla_modules: list[MLAAttention]) -> tuple[int, int]:
    """Freeze every parameter in `model`, then unfreeze parameters inside the
    given MLA modules. Returns (trainable_count, total_count)."""
    total = 0
    trainable = 0
    mla_params = {id(p) for m in mla_modules for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in mla_params)
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def freeze_except_modules(model: nn.Module, modules: list[nn.Module]) -> tuple[int, int]:
    """Freeze every parameter except parameters inside the supplied modules."""
    total = 0
    trainable = 0
    trainable_params = {id(p) for m in modules for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_params)
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


# ---------------------------------------------------------------------------
# End-to-end smoke test using a toy GQA layer wired into a 1-layer model stub.
# Exercises: find layers, convert layer, freeze, forward pass through the MLA.
# ---------------------------------------------------------------------------

class _ToyGQA(nn.Module):
    """Minimal stand-in for an HF GQA attention module (q/k/v/o_proj only)."""

    def __init__(self, hidden_size: int, num_q: int, num_kv: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_q
        self.num_key_value_heads = num_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden_size, bias=False)


class _ToyDeltaNet(nn.Module):
    """Stand-in for the linear-attention layer type we should leave alone."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, self_attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = self_attn


class _ToyModelInner(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _ToyLayer(_ToyDeltaNet(hidden_size)),
            _ToyLayer(_ToyGQA(hidden_size, num_q=8, num_kv=1, head_dim=64)),
            _ToyLayer(_ToyDeltaNet(hidden_size)),
            _ToyLayer(_ToyGQA(hidden_size, num_q=8, num_kv=1, head_dim=64)),
        ])


class _ToyModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.model = _ToyModelInner(hidden_size)


def _smoke() -> None:
    torch.manual_seed(0)
    H = 512
    model = _ToyModel(H).to(torch.float32)

    idx = find_full_attn_layers(
        model, lambda layer: isinstance(layer.self_attn, _ToyGQA)
    )
    print(f"full-attn layer indices: {idx}")
    assert idx == [1, 3]

    mla_cfg = MLAConfig(
        hidden_size=H, num_heads=8,
        qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=64,
        kv_lora_rank=256,
    )
    new_modules = [
        convert_layer_to_mla(model, i, mla_cfg, layers_path="model.layers")
        for i in idx
    ]

    for i, m in zip(idx, new_modules):
        swapped = model.model.layers[i].self_attn
        assert swapped is m, "layer attention was not swapped in place"
        assert isinstance(swapped, MLAAttention)

    trainable, total = freeze_except_mla(model, new_modules)
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Forward through one of the new MLA modules to confirm it runs.
    B, T = 2, 16
    x = torch.randn(B, T, H)
    cos = torch.ones(B, T, mla_cfg.qk_rope_head_dim)
    sin = torch.zeros(B, T, mla_cfg.qk_rope_head_dim)
    with torch.no_grad():
        y, _ = new_modules[0](x, position_embeddings=(cos, sin))
    print(f"mla[0] forward: {tuple(y.shape)}   norm={y.norm().item():.4f}")

    # Frozen DeltaNet params should still be non-trainable.
    dn_params = list(model.model.layers[0].self_attn.parameters())
    assert all(not p.requires_grad for p in dn_params), "DeltaNet must stay frozen"
    # MLA params should be trainable.
    assert all(p.requires_grad for p in new_modules[0].parameters()), "MLA must be trainable"
    print("freeze/unfreeze policy: ok")


if __name__ == "__main__":
    _smoke()
