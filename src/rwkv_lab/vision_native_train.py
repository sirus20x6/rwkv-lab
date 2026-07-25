"""Calibrate a frozen multi-teacher compressor to emit native RWKV tokens.

This is the bridge-free arm of the representation experiment.  The established
MoonViT caption checkpoint supplies the frozen RWKV/Engram/loop/deep-vision
stack and the exact evaluation contract.  Only the output head owned by the
vision compressor is optimized; its output is inserted directly as RWKV input
embeddings through a parameter-free identity shim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rwkv_lab.deep_vision import DeepVisionInjector
from rwkv_lab.engram_lmb import LexicalMemoryBank, attach_engram, float_growth_params
from rwkv_lab.generate import WorldVocab
from rwkv_lab.lookahead_module import NextLatPredictor
from rwkv_lab.moonvit import valid_torch_archive_storages
from rwkv_lab.rwkv_finetune import load_g1g_fla
from rwkv_lab.vision_caption import checkpoint_runtime_scales
from rwkv_lab.vision_compressor_features import (
    FrozenTeacherCompressor,
    NativePrefixIdentity,
    RWKVNativeTeacherCompressor,
)
from rwkv_lab.vision_grounding import ImageTextContrastiveHead
from rwkv_lab.vision_loop import (
    install_factored_timemix,
    load_loop_adapter_state,
    set_loop_enabled,
    set_loop_scale,
)
from rwkv_lab.vision_teacher_compressor import TeacherCacheDataset
from rwkv_lab.vision_train import (
    load_examples,
    make_batch,
    multimodal_loss,
    prepare_examples,
    supervised_positions,
    visual_insert_positions,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


class CaptionTeacherDataset(Dataset):
    """Keep caption rows and paired cache entries under one stable index."""

    def __init__(self, manifest: Path, cache: TeacherCacheDataset, rows: list[dict]):
        if len(cache) != len(rows):
            raise ValueError(
                f"manifest/cache row mismatch for {manifest}: {len(rows)} != {len(cache)}")
        self.manifest = manifest
        self.cache = cache
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        moon, fusion = self.cache[index]
        return index, moon, fusion


class LengthBucketBatchSampler:
    """Shuffle caption-length-homogeneous batches once per epoch.

    Random per-example batching makes the padded RWKV sequence shape jump
    wildly and can trigger avoidable compiler specializations.  We keep nearby
    lengths together and shuffle whole batches, preserving stochastic order
    with far less padding and shape churn.
    """

    def __init__(self, rows: Sequence[dict], batch_size: int, seed: int):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        ordered = sorted(range(len(rows)), key=lambda index: len(rows[index]["tokens"]))
        self.batches = [ordered[start:start + self.batch_size]
                        for start in range(0, len(ordered), self.batch_size)
                        if len(ordered[start:start + self.batch_size]) == self.batch_size]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.batches), generator=generator).tolist()
        self.epoch += 1
        for index in order:
            yield self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def _head_state(model: RWKVNativeTeacherCompressor) -> dict:
    def cpu_state(module: nn.Module) -> dict:
        return {name: value.detach().cpu()
                for name, value in module.state_dict().items()}
    return {
        "output_norm": cpu_state(model.output_norm),
        "output_projection": cpu_state(model.output_projection),
        "output_position": model.output_position.detach().cpu(),
    }


def _load_head_state(model: RWKVNativeTeacherCompressor, state: dict) -> None:
    model.output_norm.load_state_dict(state["output_norm"])
    model.output_projection.load_state_dict(state["output_projection"])
    with torch.no_grad():
        model.output_position.copy_(state["output_position"])


def _save_checkpoint(path: Path, *, model: RWKVNativeTeacherCompressor,
                     optimizer: torch.optim.Optimizer, step: int,
                     best_eval: float, args: argparse.Namespace) -> None:
    payload = {
        "schema": SCHEMA,
        "step": int(step),
        "best_eval": float(best_eval),
        "native_head": _head_state(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_frozen_caption_stack(checkpoint: Path):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(blob.get("schema", -1)) != 3:
        raise ValueError(f"unsupported baseline schema {blob.get('schema')}")
    if not valid_torch_archive_storages(checkpoint, blob):
        raise ValueError(f"baseline checkpoint failed archive integrity: {checkpoint}")
    args = blob["args"]
    rwkv = load_g1g_fla(args["rwkv"], device="cuda")
    rwkv.requires_grad_(False).eval()

    engram = None
    if bool(args.get("engram", False)):
        sites = sorted({int(value.strip()) for value in
                        str(args.get("engram_sites", "3,15")).split(",")
                        if value.strip()})
        vocab_size = int(rwkv.config.vocab_size)
        engram = LexicalMemoryBank(
            hidden_size=int(rwkv.config.hidden_size),
            vocab_size=vocab_size,
            layer_sites=sites,
            d_row=int(args.get("engram_drow", 128)),
            table_rows=min(int(args.get("engram_rows", vocab_size)), vocab_size),
            num_heads=int(rwkv.config.num_heads),
            max_loops=int(args.get("loop_count", 1)),
            boundary_id=args.get("engram_boundary_id", 0),
        ).to(device="cuda", dtype=torch.bfloat16)
        float_growth_params(engram)
        attach_engram(rwkv, engram, resolve="model.layers")
        rwkv.engram = engram
        engram.load_state_dict(blob["engram"])

    wrappers = install_factored_timemix(
        rwkv,
        n_loops=int(args.get("loop_count", 1)),
        gate_cap=float(args.get("loop_gate_cap", 0.25)),
        loop_index=bool(args.get("loop_index", False)),
    )
    load_loop_adapter_state(wrappers, blob["loops"])
    loop_enabled, loop_scale, engram_scale = checkpoint_runtime_scales(
        args, int(blob["step"]))
    set_loop_enabled(wrappers, loop_enabled)
    set_loop_scale(wrappers, loop_scale)
    if engram is not None:
        engram.set_warmup(engram_scale)

    deep_vision = None
    deep_sites = sorted({int(value.strip()) for value in
                         str(args.get("deep_vision_layers", "")).split(",")
                         if value.strip()})
    if deep_sites:
        deep_vision = DeepVisionInjector(
            int(rwkv.config.hidden_size), deep_sites,
            rank=int(args.get("deep_vision_rank", 256))).cuda().float()
        deep_vision.install(rwkv.model.layers)
        deep_vision.load_state_dict(blob["deep_vision"])

    nextlat = None
    if blob.get("nextlat") is not None:
        nextlat = NextLatPredictor(
            int(rwkv.config.hidden_size),
            hidden=int(args.get("nextlat_hidden", 1024))).cuda().float()
        nextlat.load_state_dict(blob["nextlat"])

    grounding = None
    if blob.get("grounding") is not None:
        grounding = ImageTextContrastiveHead(
            int(rwkv.config.hidden_size),
            width=int(args.get("grounding_contrastive_dim", 512)),
            temperature=float(args.get("grounding_temperature", 0.07)),
        ).cuda().float()
        grounding.load_state_dict(blob["grounding"])

    # Freeze after hooks and wrappers are installed so no late-created adapter
    # can accidentally enter the native-head optimizer.
    rwkv.requires_grad_(False)
    for module in (engram, deep_vision, nextlat, grounding):
        if module is not None:
            module.requires_grad_(False).eval()
    rwkv.eval()
    return blob, rwkv, engram, deep_vision, nextlat, grounding


def _batch_loss(*, indices: torch.Tensor, moon: torch.Tensor,
                fusion: torch.Tensor, rows: list[dict], rwkv: nn.Module,
                native: RWKVNativeTeacherCompressor,
                identity: NativePrefixIdentity,
                engram: LexicalMemoryBank | None,
                deep_vision: DeepVisionInjector | None,
                nextlat: NextLatPredictor | None,
                grounding: ImageTextContrastiveHead | None,
                baseline_args: dict, training: bool):
    chosen = [rows[int(index)] for index in indices.tolist()]
    ids, labels, mask = make_batch(chosen, device="cuda")
    positions = supervised_positions(
        chosen, identity.prefix_tokens, device="cuda")
    starts = visual_insert_positions(chosen)
    moon_items = list(moon.to("cuda", non_blocking=True).unbind(0))
    fusion_items = list(fusion.to("cuda", non_blocking=True).unbind(0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, metrics = multimodal_loss(
            rwkv, identity, None, (), ids, labels, mask,
            nextlat=nextlat if training else None,
            nextlat_weight=float(baseline_args.get("nextlat_weight", 0.0))
            if training else 0.0,
            nextlat_kl_weight=float(baseline_args.get("nextlat_kl_weight", 0.0))
            if training else 0.0,
            engram=engram,
            features=moon_items,
            selected_positions=positions,
            deep_vision=deep_vision,
            visual_starts=starts,
            fusion_features=fusion_items,
            vision_compressor=native,
            grounding=grounding if training else None,
            grounding_contrastive_weight=float(
                baseline_args.get("grounding_contrastive_weight", 0.0))
            if training else 0.0,
            grounding_early_tokens=int(
                baseline_args.get("grounding_early_tokens", 0)) if training else 0,
            grounding_early_weight=float(
                baseline_args.get("grounding_early_weight", 1.0)) if training else 1.0,
        )
    tokens = int((labels != -100).sum())
    return loss, metrics, tokens


@torch.no_grad()
def _evaluate(loader: DataLoader, rows: list[dict], *, max_examples: int,
              rwkv: nn.Module, native: RWKVNativeTeacherCompressor,
              identity: NativePrefixIdentity, engram, deep_vision,
              baseline_args: dict) -> float:
    native.eval()
    total_loss = 0.0
    total_tokens = 0
    seen = 0
    for indices, moon, fusion in loader:
        if seen >= max_examples:
            break
        remaining = max_examples - seen
        if len(indices) > remaining:
            indices, moon, fusion = indices[:remaining], moon[:remaining], fusion[:remaining]
        _, metrics, tokens = _batch_loss(
            indices=indices, moon=moon, fusion=fusion, rows=rows,
            rwkv=rwkv, native=native, identity=identity, engram=engram,
            deep_vision=deep_vision, nextlat=None, grounding=None,
            baseline_args=baseline_args, training=False)
        total_loss += float(metrics["ce_loss"]) * tokens
        total_tokens += tokens
        seen += len(indices)
    native.train()
    return total_loss / max(total_tokens, 1)


def _cache_dataset(manifest: Path, args: argparse.Namespace) -> TeacherCacheDataset:
    return TeacherCacheDataset(
        manifest,
        Path(args.moon_cache),
        Path(args.fusion_cache),
        moon_checkpoint=Path(args.moonvit),
        siglip2=args.siglip2,
        dinov2=args.dinov2,
        sam=args.sam,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=(
        "runs/moonvit_rwkv_eight_hour_grounded/best/ckpt_step_00021900.pt"))
    parser.add_argument("--compressor", default=(
        "models/vision/teacher-compressor-so400m-dinov2-sam-128x1024-v1.pt"))
    parser.add_argument("--data", default="curated_vision/vision_next_shard_000_train.jsonl")
    parser.add_argument("--eval-data", default="curated_vision/vision_eight_hour_eval.jsonl")
    parser.add_argument("--moon-cache", default=(
        "/thearray/downloads/cache/moe-mla/moonvit_next_128_shard_000"))
    parser.add_argument("--fusion-cache", default=(
        "/thearray/downloads/cache/moe-mla/fusion_so400m_next_128_shard_000"))
    parser.add_argument("--moonvit", default=(
        "models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors"))
    # These identities are part of the cache key. Keep them aligned with
    # ``cache_vision_teacher_manifest.sh`` rather than substituting equivalent
    # Hub aliases, which correctly produce different fingerprints.
    parser.add_argument("--siglip2", default="models/vision/siglip2-so400m-patch16-512")
    parser.add_argument("--dinov2", default="models/vision/dinov2-base")
    parser.add_argument("--sam", default="models/vision/sam-vit-base")
    parser.add_argument("--out", default="runs/native_compressor_rwkv_arm2")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    log_path = out / "train.jsonl"
    checkpoint_path = out / "last.pt"
    best_dir = out / "best"
    best_dir.mkdir(exist_ok=True)
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _atomic_json(status_path, {"state": "loading_baseline", "updated": time.time()})
    baseline_path = Path(args.baseline)
    blob, rwkv, engram, deep_vision, nextlat, grounding = \
        _load_frozen_caption_stack(baseline_path)
    baseline_args = blob["args"]
    hidden = int(rwkv.config.hidden_size)
    prefix_tokens = int(baseline_args["prefix_tokens"])
    if prefix_tokens != 64:
        raise ValueError(f"arm-2 comparison requires baseline prefix 64, got {prefix_tokens}")

    frozen = FrozenTeacherCompressor.from_checkpoint(
        args.compressor, device="cuda", dtype=torch.bfloat16)
    native = RWKVNativeTeacherCompressor(
        frozen, hidden, prefix_tokens=prefix_tokens).cuda()
    identity = NativePrefixIdentity(prefix_tokens, hidden).cuda()
    trainable = list(native.native_parameters())
    trainable_ids = {id(parameter) for parameter in trainable}
    all_trainable = [parameter for parameter in native.parameters()
                     if parameter.requires_grad]
    if {id(parameter) for parameter in all_trainable} != trainable_ids:
        raise RuntimeError("native compressor exposes unexpected trainable parameters")
    for module in (rwkv, engram, deep_vision, nextlat, grounding, frozen, identity):
        if module is not None and any(p.requires_grad for p in module.parameters()):
            raise RuntimeError(f"frozen module is trainable: {type(module).__name__}")
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "name": "vision_native_output"}],
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), fused=True)

    vocab = WorldVocab()
    train_rows, _ = prepare_examples(
        load_examples(args.data, stat_workers=32), vocab,
        prompt=str(baseline_args["prompt"]),
        max_text_tokens=int(baseline_args["max_text_tokens"]),
        sandwich_prompt=bool(baseline_args.get("sandwich_prompt", False)))
    eval_rows, _ = prepare_examples(
        load_examples(args.eval_data, stat_workers=32), vocab,
        prompt=str(baseline_args["prompt"]),
        max_text_tokens=int(baseline_args["max_text_tokens"]),
        sandwich_prompt=bool(baseline_args.get("sandwich_prompt", False)))
    _atomic_json(status_path, {"state": "validating_caches", "updated": time.time()})
    train_dataset = CaptionTeacherDataset(
        Path(args.data), _cache_dataset(Path(args.data), args), train_rows)
    eval_dataset = CaptionTeacherDataset(
        Path(args.eval_data), _cache_dataset(Path(args.eval_data), args), eval_rows)
    loader_args = dict(num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)
    if args.workers > 0:
        loader_args["prefetch_factor"] = 3
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=LengthBucketBatchSampler(train_rows, args.batch, args.seed),
        **loader_args)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch, shuffle=False, **loader_args)

    step = 0
    best_eval = float("inf")
    if args.resume == "auto" and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(saved.get("schema", -1)) != SCHEMA:
            raise ValueError("native arm checkpoint schema mismatch")
        _load_head_state(native, saved["native_head"])
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        best_eval = float(saved["best_eval"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    config = {
        **vars(args),
        "architecture": "six_teacher_compressor_native_rwkv_output",
        "deployment_bridge": False,
        "prefix_tokens": prefix_tokens,
        "rwkv_hidden": hidden,
        "baseline_step": int(blob["step"]),
        "baseline_best_ppl": 4.641567364290378,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_compressor_parameters": sum(p.numel() for p in frozen.parameters()),
        "frozen_rwkv_parameters": sum(p.numel() for p in rwkv.parameters()),
    }
    _atomic_json(out / "config.json", config)
    log_mode = "a" if step else "w"
    with log_path.open(log_mode) as log:
        if step == 0:
            _append_json(log, {
                "kind": "startup", "step": 0,
                "architecture": config["architecture"],
                "deployment_bridge": False,
                "baseline_run": "moonvit_rwkv_eight_hour_grounded",
                "baseline_step": int(blob["step"]),
                "baseline_ppl": config["baseline_best_ppl"],
                "train_examples": len(train_rows),
                "val_examples": len(eval_rows),
                "trainable_parameters": config["trainable_parameters"],
                "frozen_compressor_parameters": config["frozen_compressor_parameters"],
                "frozen_rwkv_parameters": config["frozen_rwkv_parameters"],
            })
        _atomic_json(status_path, {
            "state": "training", "step": step, "updated": time.time(),
            "deployment_bridge": False,
            "trainable_scope": "vision_compressor.native_output_head",
        })

        while step < args.steps and not stop:
            for indices, moon, fusion in train_loader:
                if step >= args.steps or stop:
                    break
                started = time.perf_counter()
                native.train()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics, tokens = _batch_loss(
                    indices=indices, moon=moon, fusion=fusion, rows=train_rows,
                    rwkv=rwkv, native=native, identity=identity, engram=engram,
                    deep_vision=deep_vision, nextlat=nextlat,
                    grounding=grounding, baseline_args=baseline_args,
                    training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite native arm loss: {loss}")
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                step += 1
                elapsed = time.perf_counter() - started
                event = {
                    "kind": "train", "step": step, "loss": float(loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "gnorm": float(gnorm), "tok_per_sec": tokens / max(elapsed, 1e-9),
                    "batch": len(indices), "step_seconds": elapsed,
                    "ce_loss": float(metrics["ce_loss"]),
                    "grounded_ce_loss": float(metrics["grounded_ce_loss"]),
                    "deployment_bridge": False,
                }
                for name in ("nextlat_loss", "grounding_contrastive_loss",
                             "grounding_retrieval_accuracy", "deep_vision_inj_rms"):
                    if name in metrics:
                        event[name] = float(metrics[name])
                _append_json(log, event)
                _atomic_json(status_path, {
                    "state": "training", "step": step,
                    "loss": event["loss"], "updated": time.time(),
                    "step_seconds": elapsed, "deployment_bridge": False,
                    "trainable_scope": "vision_compressor.native_output_head",
                })

                if step % args.checkpoint_every == 0:
                    _save_checkpoint(
                        checkpoint_path, model=native, optimizer=optimizer,
                        step=step, best_eval=best_eval, args=args)
                    _append_json(log, {"kind": "checkpoint", "step": step,
                                       "reason": "periodic", "path": str(checkpoint_path)})

                if step % args.eval_every == 0:
                    _atomic_json(status_path, {
                        "state": "evaluating", "step": step,
                        "updated": time.time(), "deployment_bridge": False})
                    eval_loss = _evaluate(
                        eval_loader, eval_rows, max_examples=args.eval_examples,
                        rwkv=rwkv, native=native, identity=identity,
                        engram=engram, deep_vision=deep_vision,
                        baseline_args=baseline_args)
                    ppl = math.exp(min(eval_loss, 20.0))
                    _append_json(log, {
                        "kind": "eval", "step": step, "loss": eval_loss,
                        "ppl": ppl, "baseline_ppl": config["baseline_best_ppl"],
                        "ppl_delta": ppl - config["baseline_best_ppl"],
                        "examples": min(args.eval_examples, len(eval_rows)),
                        "deployment_bridge": False,
                    })
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        best_path = best_dir / f"native_head_step_{step:08d}.pt"
                        _save_checkpoint(
                            best_path, model=native, optimizer=optimizer,
                            step=step, best_eval=best_eval, args=args)
                        _atomic_json(best_dir / "best.json", {
                            "step": step, "loss": eval_loss, "ppl": ppl,
                            "checkpoint": best_path.name,
                            "baseline_ppl": config["baseline_best_ppl"],
                            "deployment_bridge": False,
                        })
                    _save_checkpoint(
                        checkpoint_path, model=native, optimizer=optimizer,
                        step=step, best_eval=best_eval, args=args)
                    _atomic_json(status_path, {
                        "state": "training", "step": step,
                        "eval_loss": eval_loss, "eval_ppl": ppl,
                        "baseline_ppl": config["baseline_best_ppl"],
                        "updated": time.time(), "deployment_bridge": False,
                    })

    _save_checkpoint(checkpoint_path, model=native, optimizer=optimizer,
                     step=step, best_eval=best_eval, args=args)
    _atomic_json(status_path, {
        "state": "stopped" if stop else "complete", "step": step,
        "updated": time.time(), "deployment_bridge": False,
    })


if __name__ == "__main__":
    main()
