"""
Continued-pretrain the MLA+Engram combined model.

Trainable params are BOTH the 10 MLA modules (swapped-in attention layers) AND
the Engram modules at the chosen indices. Per the Engram paper, the embedding
tables train at 5x the backbone lr with zero weight decay. Implemented via two
param groups.

Usage (1M-token test from our 57M MLA checkpoint, Engram freshly patched):
    python train_mla_engram.py \
        --mla-ckpt /thearray/git/moe-mla/runs/mla_ft_50m_v4/step_001735/ckpt.pt \
        --engram-patch-dir /thearray/git/moe-mla/engram_converted \
        --tokens-bin /thearray/data/non_cvevc_tokens.bin \
        --total-tokens-in-bin 29284583603 \
        --max-steps 30 --log-every 1 --eval-every 10 --save-every 30 \
        --out-dir /thearray/git/moe-mla/runs/mla_engram_1m_test
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from train_mla import (
    chunked_ce, sample_windows, open_tokens, lr_at, TrainConfig,
    validate_train_config,
)
from load_mla_engram import load_mla_engram
from engram_integration import collect_host_embedding_params, collect_gpu_engram_params
from safe_torch import safe_torch_load


@dataclass
class EngramTrainConfig(TrainConfig):
    # Additional knobs for the combined run
    mla_ckpt: str = ""                          # path to a prior MLA-only checkpoint
    engram_patch_dir: str = "/thearray/git/moe-mla/engram_converted"
    engram_lr_mult: float = 5.0                 # paper: 5x lr for embedding tables


def save_checkpoint(step: int, mla_mods, engram_mods,
                    optimizer_gpu, optimizer_host, cfg) -> None:
    out_final = Path(cfg.out_dir) / f"step_{step:06d}"
    out = Path(cfg.out_dir) / f".step_{step:06d}.tmp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    mla_sds = {}
    for m in mla_mods:
        key = getattr(m, "_save_key", None)
        if key is None:
            raise RuntimeError("MLA module missing _save_key (re-load via load_converted_model)")
        mla_sds[key] = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    eng_sds = {}
    eng_indices = json.loads(
        (Path(cfg.engram_patch_dir) / "manifest.json").read_text()
    )["layer_indices"]
    for li, m in zip(eng_indices, engram_mods):
        eng_sds[f"layer_{li}"] = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    payload = {
        "step": step,
        "mla_state_dicts": mla_sds,
        "engram_state_dicts": eng_sds,
        "optimizer_state_gpu": optimizer_gpu.state_dict(),
        "config": asdict(cfg),
    }
    if optimizer_host is not None:
        payload["optimizer_state_host"] = optimizer_host.state_dict()
    torch.save(payload, out / "ckpt.pt")
    if out_final.exists():
        shutil.rmtree(out_final)
    out.rename(out_final)


@torch.no_grad()
def eval_loss(model, arr: np.memmap, eval_start: int, eval_end: int,
              cfg: EngramTrainConfig, device: torch.device) -> tuple[float, float]:
    model.eval()
    rng = np.random.default_rng(12345)
    total_loss = 0.0
    total_tokens = 0
    for _ in range(cfg.eval_batches):
        ids = sample_windows(arr, eval_start, eval_end, cfg.seq_len,
                             cfg.micro_batch_size, rng).to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(input_ids=x).logits
        loss = chunked_ce(logits, y) * y.numel()
        total_loss += loss.item()
        total_tokens += y.numel()
        del ids, x, y, logits, loss
    model.train()
    torch.cuda.empty_cache()
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in EngramTrainConfig.__dataclass_fields__.values():
        t = type(f.default) if not isinstance(f.default, bool) else bool
        ap.add_argument(f"--{f.name.replace('_','-')}", type=t, default=f.default)
    args = ap.parse_args()
    cfg = EngramTrainConfig(**{k.replace("-","_"): v for k, v in vars(args).items()})
    validate_train_config(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "train.jsonl").open("a")
    def log(row: dict) -> None:
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

    print("Loading MLA+Engram model...")
    model, mla_mods, eng_mods = load_mla_engram(
        model_dir=cfg.model_dir,
        mla_patch_dir=cfg.patch_dir,
        mla_trained_ckpt=cfg.mla_ckpt or None,
        engram_patch_dir=cfg.engram_patch_dir,
        dtype=torch.bfloat16,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    device = next(model.parameters()).device

    # Freeze everything except MLA + Engram params
    mla_param_ids = {id(p) for m in mla_mods for p in m.parameters()}
    eng_param_ids = {id(p) for m in eng_mods for p in m.parameters()}
    trainable_ids = mla_param_ids | eng_param_ids
    trainable_total = 0
    param_total = 0
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)
        param_total += p.numel()
        if p.requires_grad:
            trainable_total += p.numel()
    print(f"trainable params: {trainable_total/1e9:.3f}B / {param_total/1e9:.3f}B "
          f"(mla={sum(p.numel() for m in mla_mods for p in m.parameters())/1e9:.3f}B, "
          f"eng={sum(p.numel() for m in eng_mods for p in m.parameters())/1e9:.3f}B)")

    # Partition Engram params by location:
    #   - embedding tables on pinned CPU memory (if host_offload=True) -> SparseAdam
    #   - projections / norms / conv on GPU -> 8-bit AdamW (5x lr, wd=0)
    # MLA params all live on GPU -> 8-bit AdamW (base lr, wd=cfg.weight_decay)
    mla_params = [p for m in mla_mods for p in m.parameters() if p.requires_grad]
    eng_host_params = collect_host_embedding_params(eng_mods)
    eng_gpu_params = collect_gpu_engram_params(eng_mods)

    import bitsandbytes as bnb
    optimizer_gpu = bnb.optim.AdamW8bit(
        [
            {"params": mla_params, "lr": cfg.lr,
             "weight_decay": cfg.weight_decay},
            {"params": eng_gpu_params, "lr": cfg.lr * cfg.engram_lr_mult,
             "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    # SparseAdam (CPU) handles the big host-resident tables: only rows that
    # were accessed during the forward get moved.
    optimizer_host = (
        torch.optim.SparseAdam(
            eng_host_params, lr=cfg.lr * cfg.engram_lr_mult, betas=(0.9, 0.95)
        )
        if eng_host_params else None
    )
    print(f"  MLA params: {sum(p.numel() for p in mla_params)/1e6:.1f}M (GPU, 8-bit AdamW)")
    print(f"  Engram GPU params: {sum(p.numel() for p in eng_gpu_params)/1e6:.1f}M (GPU, 8-bit AdamW)")
    print(f"  Engram host params: {sum(p.numel() for p in eng_host_params)/1e9:.2f}B (CPU, SparseAdam)")

    # Resume from a combined MLA+Engram checkpoint (same format as save_checkpoint)
    start_step = 0
    if cfg.resume:
        print(f"resuming from: {cfg.resume}")
        ckpt = safe_torch_load(cfg.resume, map_location="cpu")
        saved_mla = ckpt["mla_state_dicts"]
        for m in mla_mods:
            key = getattr(m, "_save_key", None)
            if key is None or key not in saved_mla:
                continue
            sd = {k: v.to(device=next(m.parameters()).device, dtype=next(m.parameters()).dtype)
                  for k, v in saved_mla[key].items()}
            m.load_state_dict(sd, strict=False)
        if "engram_state_dicts" in ckpt:
            eng_indices = json.loads(
                (Path(cfg.engram_patch_dir) / "manifest.json").read_text()
            )["layer_indices"]
            for li, m in zip(eng_indices, eng_mods):
                em_params = dict(m.named_parameters())
                sd = {}
                for k, v in ckpt["engram_state_dicts"][f"layer_{li}"].items():
                    p = em_params.get(k)
                    if p is not None:
                        sd[k] = v.to(device=p.device, dtype=p.dtype)
                    else:
                        sd[k] = v
                m.load_state_dict(sd, strict=False)
        if "optimizer_state_gpu" in ckpt:
            optimizer_gpu.load_state_dict(ckpt["optimizer_state_gpu"])
        if "optimizer_state_host" in ckpt and optimizer_host is not None:
            optimizer_host.load_state_dict(ckpt["optimizer_state_host"])
        start_step = int(ckpt["step"])
        del ckpt
        print(f"resumed at step {start_step}")

    arr, train_end, eval_end = open_tokens(cfg)
    eval_start = train_end
    print(f"data: {train_end/1e9:.2f}B train tokens, "
          f"{(eval_end-train_end)/1e6:.1f}M eval tokens")

    print("baseline eval...")
    t0 = time.time()
    el, ep = eval_loss(model, arr, eval_start, eval_end, cfg, device)
    print(f"  step {start_step} | eval_loss={el:.4f}  ppl={ep:.2f}  ({time.time()-t0:.0f}s)")
    log({"step": start_step, "kind": "eval", "loss": el, "ppl": ep})

    # Training loop — apply 5x multiplier per group when scheduling
    model.train()
    step = start_step
    running_loss = 0.0
    running_tokens = 0
    t_win = time.time()
    while step < cfg.max_steps:
        optimizer_gpu.zero_grad(set_to_none=True)
        if optimizer_host is not None:
            optimizer_host.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            ids = sample_windows(arr, 0, train_end, cfg.seq_len,
                                 cfg.micro_batch_size, rng).to(device, non_blocking=True)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(input_ids=x).logits
            loss = chunked_ce(logits, y)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps
            running_tokens += y.numel()
            del logits, loss

        base_lr = lr_at(step, cfg, start_step=start_step)
        optimizer_gpu.param_groups[0]["lr"] = base_lr
        optimizer_gpu.param_groups[1]["lr"] = base_lr * cfg.engram_lr_mult
        if optimizer_host is not None:
            for g in optimizer_host.param_groups:
                g["lr"] = base_lr * cfg.engram_lr_mult

        # Clip GPU params only; sparse CPU grads would require a mixed-device
        # joint norm that's slow. Tables rely on SparseAdam + 5x lr for control.
        gnorm = torch.nn.utils.clip_grad_norm_(mla_params + eng_gpu_params, cfg.grad_clip)
        optimizer_gpu.step()
        if optimizer_host is not None:
            optimizer_host.step()
        step += 1
        running_loss += accum_loss

        if step % cfg.log_every == 0:
            dt = time.time() - t_win
            tps = running_tokens / dt
            avg = running_loss / cfg.log_every
            print(f"  step {step:5d} | loss={avg:.4f}  lr={base_lr:.2e}  "
                  f"eng_lr={base_lr*cfg.engram_lr_mult:.2e}  "
                  f"gnorm={gnorm:.2f}  tok/s={tps:.0f}")
            log({"step": step, "kind": "train", "loss": avg,
                 "lr": base_lr, "eng_lr": base_lr*cfg.engram_lr_mult,
                 "gnorm": float(gnorm), "tok_per_sec": tps})
            running_loss = 0.0
            running_tokens = 0
            t_win = time.time()

        if step % cfg.eval_every == 0:
            el, ep = eval_loss(model, arr, eval_start, eval_end, cfg, device)
            print(f"  step {step:5d} | eval_loss={el:.4f}  ppl={ep:.2f}")
            log({"step": step, "kind": "eval", "loss": el, "ppl": ep})
            t_win = time.time()

        if step % cfg.save_every == 0:
            save_checkpoint(step, mla_mods, eng_mods, optimizer_gpu, optimizer_host, cfg)
            log({"step": step, "kind": "checkpoint"})
            t_win = time.time()

    save_checkpoint(step, mla_mods, eng_mods, optimizer_gpu, optimizer_host, cfg)
    el, ep = eval_loss(model, arr, eval_start, eval_end, cfg, device)
    log({"step": step, "kind": "eval", "loss": el, "ppl": ep})
    log_f.close()
    print(f"done. final eval_ppl={ep:.2f}")


if __name__ == "__main__":
    main()
