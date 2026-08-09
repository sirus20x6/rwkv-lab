"""
Smoke test: can we load the MLA+Engram combined model, run a forward+backward
on a tiny batch, and collect gradients on both MLA and Engram parameters?

Does not run training. Just validates that everything's wired up correctly
and reports total VRAM used so we know if a real run will fit.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from .host_paths import require_host_path, resolve_host_path
from .load_mla_engram import ENGRAM_PATCH_DIR_ENV, load_mla_engram
from .train_mla import chunked_ce


# Host-specific artifact locations; see rwkv_lab.host_paths for the resolution
# policy and the README section "Host-specific path defaults".
#
# The Engram patch is the same artifact load_mla_engram loads — this module
# only hands its own flag to that loader — so it shares that loader's variable
# and keeps only its own historical vintage. The MLA checkpoint is a run
# output, one particular trained step under the maintainer's run root, so it
# gets a variable of its own; nothing else in the tree names it.
ENGRAM_PATCH_DIR_HISTORICAL_PATH = "/thearray/git/moe-mla/engram_converted_v2"
MLA_CKPT_ENV = "MOE_MLA_CKPT"
MLA_CKPT_HISTORICAL_PATH = (
    "/thearray/git/moe-mla/runs/mla_ft_50m_v4/step_001735/ckpt.pt"
)


def default_engram_patch_dir() -> str | None:
    return resolve_host_path(
        ENGRAM_PATCH_DIR_ENV, ENGRAM_PATCH_DIR_HISTORICAL_PATH
    )


def default_mla_ckpt() -> str | None:
    return resolve_host_path(MLA_CKPT_ENV, MLA_CKPT_HISTORICAL_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mla-ckpt", default=default_mla_ckpt())
    ap.add_argument("--engram-patch-dir", default=default_engram_patch_dir())
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    args.mla_ckpt = require_host_path(
        args.mla_ckpt, field="mla_ckpt", env_var=MLA_CKPT_ENV,
        flag="mla-ckpt",
    )
    args.engram_patch_dir = require_host_path(
        args.engram_patch_dir, field="engram_patch_dir",
        env_var=ENGRAM_PATCH_DIR_ENV, flag="engram-patch-dir",
    )

    print("Loading MLA + Engram model...")
    t0 = time.time()
    model, mla_mods, eng_mods = load_mla_engram(
        mla_trained_ckpt=args.mla_ckpt,
        engram_patch_dir=args.engram_patch_dir,
    )
    print(f"load time: {time.time()-t0:.1f}s")

    # Freeze everything except MLA + Engram so backward only allocates grad
    # buffers for the trainable 1.1B-ish params, not the full 36.7B.
    trainable_ids = {id(p) for m in mla_mods for p in m.parameters()}
    trainable_ids |= {id(p) for m in eng_mods for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)

    model.train()
    model.gradient_checkpointing_enable()

    # Synthetic input_ids (random valid tokens)
    device = next(model.parameters()).device
    # AutoModelForCausalLM variant exposes the text config directly, while the
    # multimodal variant wraps it in .text_config. Handle both.
    cfg = getattr(model.config, "text_config", model.config)
    vocab = cfg.vocab_size
    torch.manual_seed(0)
    x = torch.randint(0, min(vocab, 200_000),
                      (args.batch_size, args.seq_len + 1), device=device)
    input_ids, labels = x[:, :-1], x[:, 1:]

    print(f"\nforward: batch={args.batch_size}  seq={args.seq_len}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(input_ids=input_ids)
    fwd_dt = time.time() - t0
    print(f"  forward done in {fwd_dt:.2f}s")
    print(f"  logits shape: {tuple(out.logits.shape)}")

    print("\nbackward (loss.backward):")
    loss = chunked_ce(out.logits, labels)
    print(f"  loss: {loss.item():.4f}")
    t0 = time.time()
    loss.backward()
    bwd_dt = time.time() - t0
    print(f"  backward done in {bwd_dt:.2f}s")

    # Check gradients — handle sparse grads (from CPU host-offloaded embeddings)
    def _grad_abs_sum(p):
        if p.grad is None:
            return 0.0
        g = p.grad
        if g.is_sparse:
            g = g.coalesce()
            return g._values().abs().sum().item()
        return g.abs().sum().item()

    mla_grad = sum(_grad_abs_sum(p) for m in mla_mods for p in m.parameters())
    eng_grad = sum(_grad_abs_sum(p) for m in eng_mods for p in m.parameters())
    n_mla_with_grad = sum(
        sum(1 for p in m.parameters() if p.grad is not None) for m in mla_mods
    )
    n_eng_with_grad = sum(
        sum(1 for p in m.parameters() if p.grad is not None) for m in eng_mods
    )
    n_cpu_grads = sum(
        1 for m in eng_mods for p in m.parameters()
        if p.grad is not None and p.grad.device.type == "cpu"
    )
    print(f"\ngradient sanity:")
    print(f"  mla params with grad:    {n_mla_with_grad}  (sum |grad|: {mla_grad:.3e})")
    print(f"  engram params with grad: {n_eng_with_grad}  (sum |grad|: {eng_grad:.3e})")
    print(f"  engram grads on CPU:     {n_cpu_grads}  (host-offload verification)")

    peak = torch.cuda.max_memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e9
    print(f"\nVRAM:  peak during forward+backward: {peak:.1f} GB")
    print(f"VRAM:  currently used (full GPU):    {used:.1f} / {total/1e9:.1f} GB")

    assert mla_grad > 0, "MLA params did not receive gradients"
    assert eng_grad > 0, "Engram params did not receive gradients"
    print("\n✓ smoke test passed")


if __name__ == "__main__":
    main()
