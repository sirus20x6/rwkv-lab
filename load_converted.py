"""
Load the original Qwen3.6-35B-A3B checkpoint, swap the full-attention layers
with MLAAttention instances initialized from the patch file written by
convert.py, and optionally freeze everything except the MLA modules for
finetuning.

Usage:
    from load_converted import load_converted_model

    model = load_converted_model(
        model_dir="/thearray/git/moe-mla/Qwen3.6-35B-A3B",
        patch_dir="/thearray/git/moe-mla/converted",
        device_map="auto",
        freeze_non_mla=True,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from svd_init import GQAConfig, MLAConfig
from mla_module import MLAAttention
from mtp_module import Qwen3_5MoeMTPModule, load_mtp_weights_from_checkpoint
from layer_swap import convert_deltanet_layer_to_rwkv8
from rwkv8_deltanet import parse_layer_list, linear_attention_layer_indices_from_config


def _load_patch(patch_path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(patch_path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _build_mla(mla_cfg: MLAConfig, gqa_cfg: GQAConfig, dtype: torch.dtype,
               device: torch.device) -> MLAAttention:
    m = MLAAttention(
        hidden_size=mla_cfg.hidden_size,
        num_heads=mla_cfg.num_heads,
        qk_nope_head_dim=mla_cfg.qk_nope_head_dim,
        qk_rope_head_dim=mla_cfg.qk_rope_head_dim,
        v_head_dim=mla_cfg.v_head_dim,
        kv_lora_rank=mla_cfg.kv_lora_rank,
        q_lora_rank=mla_cfg.q_lora_rank,
        has_output_gate=mla_cfg.has_output_gate,
        has_qk_norm=mla_cfg.has_qk_norm,
        num_kv_rope_heads=mla_cfg.num_kv_rope_heads,
        rope_position=gqa_cfg.rope_position,
        use_latent_norm=False,
    )
    return m.to(device=device, dtype=dtype)


def _resolve_mtp(model) -> Optional[object]:
    """Locate the MTP block on the loaded HF model. Returns a module that has
    a `.layers` list with at least one element (MTP's single decoder block).
    Path varies between loaded variants (CausalLM vs ConditionalGeneration);
    we first try known paths, then scan named_modules."""
    for get in (
        lambda m: m.mtp,
        lambda m: m.model.mtp,
        lambda m: m.model.language_model.mtp,
    ):
        try:
            mtp = get(model)
            if mtp is not None and hasattr(mtp, "layers"):
                return mtp
        except AttributeError:
            continue
    # Fallback scan
    for name, mod in model.named_modules():
        tail = name.rsplit(".", 1)[-1]
        if tail == "mtp" and hasattr(mod, "layers"):
            return mod
    return None


def _load_attn_from_patch(patch: dict[str, torch.Tensor], mla: MLAAttention,
                          prefix: str, dtype: torch.dtype,
                          device: torch.device, label: str) -> None:
    """Load the MLA state dict from `patch[prefix + key] -> mla.state_dict()[key]`."""
    local_sd = {
        k[len(prefix):]: v.to(device=device, dtype=dtype)
        for k, v in patch.items() if k.startswith(prefix)
    }
    missing, unexpected = mla.load_state_dict(local_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{label}: unexpected keys {unexpected}")
    if missing:
        allowed = {"kv_a_layernorm.weight", "q_a_layernorm.weight"}
        hard = [m for m in missing if m not in allowed]
        if hard:
            raise RuntimeError(f"{label}: missing keys {hard}")


def load_converted_model(
    model_dir: str = "/thearray/git/moe-mla/Qwen3.6-35B-A3B",
    patch_dir: str = "/thearray/git/moe-mla/converted",
    device_map: str | dict = "auto",
    dtype: torch.dtype = torch.bfloat16,
    freeze_non_mla: bool = True,
    install_mtp: bool = False,
    rwkv8_deltanet_layers: str | list[int] | tuple[int, ...] | None = None,
    rwkv8_ffn_hidden_size: int | None = None,
    rwkv8_init_output_scale: float = 1e-3,
    rwkv8_swap_mode: str = "timemix",
):
    """Load original model, swap full-attn with MLA, load patch weights.

    If install_mtp=True (and the patch's manifest has mtp_converted), also build
    a Qwen3_5MoeMTPModule, load its non-attention weights from the BASE
    checkpoint's mtp.* tensors (HF transformers silently drops these via
    _keys_to_ignore_on_load_unexpected, so we read them directly), swap its
    decoder's self_attn for MLA, and attach it to the model as `.mtp_trainer`.

    `rwkv8_deltanet_layers` optionally replaces selected linear-attention
    DeltaNet layers with RWKV-8 ChannelMix modules. Those modules are appended
    to the returned trainable-module list and tagged `_save_key`.

    Returns (model, list_of_trainable_attention_modules). If MTP was installed,
    an extra module is appended (tagged _save_key="mtp"), and model.mtp_trainer
    is set.
    """
    from transformers import AutoModelForCausalLM  # lazy: import only when used

    model_dir_p = Path(model_dir)
    patch_dir_p = Path(patch_dir)
    manifest = json.loads((patch_dir_p / "manifest.json").read_text())

    gqa_cfg = GQAConfig(**manifest["gqa_config"])
    mla_cfg = MLAConfig(**manifest["mla_config"])
    full_attn_idx: list[int] = manifest["full_attn_layer_indices"]

    print(f"loading base model: {model_dir_p}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_p,
        dtype=dtype,
        device_map=device_map,
    )

    patch = _load_patch(patch_dir_p / "patch.safetensors")
    print(f"patch has {len(patch)} tensors")

    # Decoder layers live at model.model.language_model.layers on Qwen3.5_moe.
    # Also cover the top-level "language_model" attr path that older wrappers use.
    inner = model.model
    layers = getattr(inner, "layers", None)
    layers_path = "model.layers"
    if layers is None:
        inner = inner.language_model
        layers = inner.layers
        layers_path = "model.language_model.layers"

    new_modules: list[MLAAttention] = []
    for li in full_attn_idx:
        layer = layers[li]
        old = layer.self_attn
        example_param = next(old.parameters())
        mla = _build_mla(mla_cfg, gqa_cfg, dtype=dtype, device=example_param.device)
        prefix = f"model.language_model.layers.{li}.self_attn."
        _load_attn_from_patch(patch, mla, prefix, dtype, example_param.device,
                              label=f"layer {li}")
        mla._save_key = f"layer_{li}"  # used by save_checkpoint/resume
        layer.self_attn = mla
        new_modules.append(mla)

    rwkv8_idx = parse_layer_list(rwkv8_deltanet_layers)
    if rwkv8_idx:
        valid_linear = set(linear_attention_layer_indices_from_config(model.config))
        bad = [li for li in rwkv8_idx if li not in valid_linear]
        if bad:
            raise ValueError(
                f"rwkv8_deltanet_layers includes non-linear-attention layers {bad}; "
                f"valid linear-attention layers are {sorted(valid_linear)}"
            )
        for li in rwkv8_idx:
            rwkv = convert_deltanet_layer_to_rwkv8(
                model,
                li,
                mode=rwkv8_swap_mode,
                layers_path=layers_path,
                ffn_hidden_size=rwkv8_ffn_hidden_size or None,
                init_output_scale=rwkv8_init_output_scale,
            )
            new_modules.append(rwkv)
        _label = "TimeMix" if rwkv8_swap_mode == "timemix" else "ChannelMix"
        print(f"installed RWKV-8 {_label} DeltaNet replacements at layers {rwkv8_idx}")

    # MTP setup. HF transformers silently drops mtp.* weights from the state
    # dict, so we build Qwen3_5MoeMTPModule ourselves, then:
    #   - non-attention weights (fc, norms, mlp) come from the BASE checkpoint's
    #     mtp.* tensors (read directly from safetensors)
    #   - attention weights come from our patch's MTP MLA keys (if converted)
    #   - embed_tokens is tied to the backbone's embed_tokens
    if install_mtp and manifest.get("mtp_converted"):
        # Locate backbone config + embed_tokens (paths differ by loaded variant)
        backbone_text_cfg = getattr(model.config, "text_config", model.config)
        embed_tokens = model.get_input_embeddings()
        device = next(model.parameters()).device

        mtp = Qwen3_5MoeMTPModule(backbone_text_cfg)
        mtp = mtp.to(device=device, dtype=dtype)
        mtp.set_shared_embed_tokens(embed_tokens)

        # Load base MTP weights (fc, norms, mlp, + original-GQA self_attn
        # keys we'll replace in a moment).
        base_mtp_sd = load_mtp_weights_from_checkpoint(model_dir, device, dtype)
        # Rename: "mtp.layers.0.self_attn.*" in ckpt -> "layers.0.self_attn.*"
        # in our module. Strip the "mtp." prefix to match state_dict keys.
        target_sd = {}
        for k, v in base_mtp_sd.items():
            # Skip keys that will be replaced by MLA (we'll load those separately)
            if k.startswith("layers.0.self_attn."):
                continue
            target_sd[k] = v.to(device=device)
        missing, unexpected = mtp.load_state_dict(target_sd, strict=False)
        # Unexpected is OK for weights not represented in our module (e.g.,
        # MoE expert weight layout differences); we error only on hard misses.
        allowed_missing = {"layers.0.self_attn." + k for k in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "q_norm.weight", "k_norm.weight"
        )}
        hard_missing = [m for m in missing if m not in allowed_missing]
        if hard_missing:
            print(f"mtp: {len(hard_missing)} non-attn weights missing "
                  f"(expected if module structure differs from checkpoint): "
                  f"{hard_missing[:5]}...")
        if unexpected:
            print(f"mtp: {len(unexpected)} unexpected keys ignored: {unexpected[:5]}")

        # Swap MTP's self_attn for MLA, load from patch.
        mla = _build_mla(mla_cfg, gqa_cfg, dtype=dtype, device=device)
        prefix = manifest.get("mtp_attn_prefix") or "mtp.layers.0.self_attn."
        _load_attn_from_patch(patch, mla, prefix, dtype, device,
                              label="mtp.layers.0 (via our MTPModule)")
        mla._save_key = "mtp"
        mtp.layers[0].self_attn = mla
        new_modules.append(mla)

        # Attach to the model so callers (trainer) can reach it.
        model.mtp_trainer = mtp
        print(f"installed MTP module with MLA self_attn (key: mtp)")

    if freeze_non_mla:
        trainable_ids = {id(p) for m in new_modules for p in m.parameters()}
        # When MTP is installed, everything inside it (fc, norms, the decoder
        # layer's MoE, etc.) is trainable — EXCEPT the shared embed_tokens,
        # which is tied to the backbone and stays frozen.
        mtp = getattr(model, "mtp_trainer", None)
        if mtp is not None:
            shared_embed_id = (
                id(mtp.embed_tokens.weight) if mtp.embed_tokens is not None else None
            )
            for p in mtp.parameters():
                if id(p) != shared_embed_id:
                    trainable_ids.add(id(p))

        trainable = 0
        total = 0
        # Assigning nn.Module attribute (mtp_trainer) auto-registers it, so
        # mtp params already appear in model.parameters().
        for p in model.parameters():
            p.requires_grad_(id(p) in trainable_ids)
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"frozen; trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")

    return model, new_modules


# ---------------------------------------------------------------------------
# Entry point: run a forward smoke test for the converted model.
# MLAAttention does not implement KV-cache decoding yet, so generation is not
# a valid smoke test until cache support exists.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    from transformers import AutoTokenizer

    model, new_modules = load_converted_model(freeze_non_mla=False)
    tok = AutoTokenizer.from_pretrained("/thearray/git/moe-mla/Qwen3.6-35B-A3B")

    prompt = "The fundamental theorem of calculus states that"
    ids = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    print(f"\nprompt: {prompt!r}")
    with torch.no_grad():
        out = model(**ids, use_cache=False)
    print(f"forward ok: logits={tuple(out.logits.shape)}")


if __name__ == "__main__":
    _smoke()
