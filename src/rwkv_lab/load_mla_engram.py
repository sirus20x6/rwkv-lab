"""
Load the MLA-converted Qwen3.6-35B-A3B, install Engram modules, and load
both patches. Returns a ready-to-train model and the lists of MLA + Engram
modules for param-group management.

Usage:
    model, mla_mods, eng_mods = load_mla_engram(
        mla_trained_ckpt="/thearray/git/moe-mla/runs/mla_ft_50m_v4/step_001735/ckpt.pt",
        engram_patch_dir="/thearray/git/moe-mla/engram_converted",
    )
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from .load_converted import load_converted_model
from .engram_integration import install_engram, offload_engram_embedding
from .safe_torch import safe_torch_load

_ENGRAM_PATH = Path("/thearray/git/engram/python")
if str(_ENGRAM_PATH) not in sys.path:
    sys.path.insert(0, str(_ENGRAM_PATH))

from engram_ext.engram_module import EngramConfig  # noqa: E402


def _read_patch(patch_path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(patch_path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _apply_mla_trained_ckpt(mla_mods, ckpt_path: str, full_attn_idx: list[int]) -> int:
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    for li, m in zip(full_attn_idx, mla_mods):
        sd = {k: v.to(device=next(m.parameters()).device, dtype=next(m.parameters()).dtype)
              for k, v in ckpt["mla_state_dicts"][f"layer_{li}"].items()}
        m.load_state_dict(sd, strict=False)
    step = int(ckpt["step"])
    del ckpt
    return step


def _apply_engram_patch(engram_mods, engram_patch: dict[str, torch.Tensor],
                       layer_indices: list[int]) -> None:
    """Copy patch tensors into each EngramModule. Each module's own param
    resides on its target device already (CPU for host-offloaded embeddings,
    GPU for projections/norms/conv) — we convert the patch tensor to match
    that per-param device and dtype before load_state_dict."""
    for li, em in zip(layer_indices, engram_mods):
        prefix = f"layer_{li}."
        em_params = dict(em.named_parameters())
        em_buffers = dict(em.named_buffers())
        sd = {}
        for k, v in engram_patch.items():
            if not k.startswith(prefix):
                continue
            local_k = k[len(prefix):]
            target = em_params.get(local_k)
            if target is None:
                target = em_buffers.get(local_k)
            if target is None:
                # Unknown key — will show up as "unexpected" below
                sd[local_k] = v
            else:
                sd[local_k] = v.to(device=target.device, dtype=target.dtype)
        missing, unexpected = em.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"engram layer {li}: unexpected keys {unexpected}")


def load_mla_engram(
    model_dir: str = "/thearray/git/moe-mla/Qwen3.6-35B-A3B",
    mla_patch_dir: str = "/thearray/git/moe-mla/converted",
    mla_trained_ckpt: Optional[str] = None,
    engram_patch_dir: str = "/thearray/git/moe-mla/engram_converted",
    device_map: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
):
    """Build the full MLA+Engram model. Returns (model, mla_modules, engram_modules).

    - Loads base Qwen checkpoint
    - Replaces 10 GQA attention layers with SVD-initialized MLA (via MLA patch)
    - If mla_trained_ckpt given, loads its MLA weights on top
    - Installs Engram modules at layers listed in engram_patch's manifest
    - Loads Engram weights from patch (random-init, zero-conv)
    """
    # 1. MLA load (SVD-init + optional trained state)
    model, mla_modules = load_converted_model(
        model_dir=model_dir, patch_dir=mla_patch_dir,
        device_map=device_map, dtype=dtype, freeze_non_mla=False,
    )
    full_attn_idx = json.loads((Path(mla_patch_dir) / "manifest.json").read_text())[
        "full_attn_layer_indices"]

    if mla_trained_ckpt:
        step = _apply_mla_trained_ckpt(mla_modules, mla_trained_ckpt, full_attn_idx)
        print(f"loaded trained MLA checkpoint at step {step}")

    # 2. Engram install + load
    eng_manifest = json.loads((Path(engram_patch_dir) / "manifest.json").read_text())
    eng_cfg = EngramConfig(**eng_manifest["engram_config"])
    host_offload = bool(eng_manifest.get("host_offload", False))
    engram_modules = install_engram(
        model,
        layer_indices=eng_manifest["layer_indices"],
        engram_cfg=eng_cfg,
        hidden_size=eng_manifest["hidden_size"],
        host_offload_embedding=host_offload,
    )

    engram_patch = _read_patch(Path(engram_patch_dir) / "patch.safetensors")
    _apply_engram_patch(engram_modules, engram_patch, eng_manifest["layer_indices"])

    # 3. Report
    total = sum(p.numel() for p in model.parameters())
    mla_params = sum(p.numel() for m in mla_modules for p in m.parameters())
    eng_params = sum(p.numel() for m in engram_modules for p in m.parameters())
    print(f"total model params:   {total/1e9:.3f} B")
    print(f"  of which MLA:       {mla_params/1e6:.1f} M ({100*mla_params/total:.2f}%)")
    print(f"  of which Engram:    {eng_params/1e6:.1f} M ({100*eng_params/total:.2f}%)")

    return model, mla_modules, engram_modules
