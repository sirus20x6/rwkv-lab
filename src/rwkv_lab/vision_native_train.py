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
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rwkv_lab.trainvm_worker.mutation_sentinel import (
    OptimizerMutationSentinel,
)
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
from rwkv_lab.vision_step_zero import (
    attempt_baseline_gate,
    caption_eval_examples,
    publish_attempt_baseline_evidence,
    refuse_ungated_attempt_baseline,
    select_heldout_indices,
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
                     best_eval: float, args: argparse.Namespace,
                     component_evidence: dict | None = None,
                     component_composition_digest: str | None = None,
                     worker_control_state: dict | None = None) -> None:
    payload = {
        "schema": SCHEMA,
        "step": int(step),
        "best_eval": float(best_eval),
        "native_head": _head_state(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "component_evidence": component_evidence,
        "component_composition_digest": component_composition_digest,
        "worker_control_state": worker_control_state,
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


def _replace_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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
                baseline_args: dict, training: bool,
                return_selected_predictions: bool = False):
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
            return_selected_predictions=return_selected_predictions,
        )
    tokens = int((labels != -100).sum())
    return loss, metrics, tokens


@torch.no_grad()
def _heldout_caption_examples(*, loader: DataLoader, rows: list[dict],
                              wanted: tuple[int, ...], vocab: WorldVocab,
                              optimizer_step: int, rwkv: nn.Module,
                              native: RWKVNativeTeacherCompressor,
                              identity: NativePrefixIdentity, engram,
                              deep_vision, baseline_args: dict) -> tuple:
    """Render the frozen held-out rows as typed attempt-baseline evidence.

    Teacher-forced next-token evidence rather than free generation, for the same
    reason the Transformer MLA routes publish it: this composition contract
    declares no ``generation_policy`` slot, so there is no authored decode
    policy that would make a sampled caption reproducible across attempts.

    The evaluation loader is deterministic (``shuffle=False``) and yields its
    own dataset indices, so the frozen selection is applied by filtering rather
    than by re-batching. That keeps the collation this dataset was built for and
    avoids a second, subtly different, batching path existing only for evidence.
    """

    selected = set(wanted)
    was_training = native.training
    native.eval()
    rendered: dict[int, tuple[str, str, str, str]] = {}
    try:
        for indices, moon, fusion in loader:
            keep = [position for position, index in enumerate(indices.tolist())
                    if int(index) in selected]
            if not keep:
                continue
            picker = torch.tensor(keep, dtype=torch.long)
            _, metrics, _ = _batch_loss(
                indices=indices[picker], moon=moon[picker], fusion=fusion[picker],
                rows=rows, rwkv=rwkv, native=native, identity=identity,
                engram=engram, deep_vision=deep_vision, nextlat=None,
                grounding=None, baseline_args=baseline_args, training=False,
                return_selected_predictions=True)
            row_of = metrics["_eval_selected_rows"].tolist()
            predicted = metrics["_eval_selected_predictions"].tolist()
            targeted = metrics["_eval_selected_targets"].tolist()
            grouped: dict[int, tuple[list[int], list[int]]] = {}
            for position, prediction, target in zip(row_of, predicted, targeted):
                bucket = grouped.setdefault(int(position), ([], []))
                bucket[0].append(int(target))
                bucket[1].append(int(prediction))
            for position, (targets, predictions) in grouped.items():
                index = int(indices[picker][position])
                row = rows[index]
                rendered[index] = (
                    str(row.get("image", index)),
                    str(row.get("prompt") or ""),
                    vocab.decode(targets),
                    vocab.decode(predictions),
                )
            if len(rendered) >= len(selected):
                break
    finally:
        native.train(was_training)
    missing = selected - set(rendered)
    if missing:
        raise ValueError(
            f"native-head held-out selection lost {len(missing)} of "
            f"{len(selected)} rows before evidence was rendered")
    return caption_eval_examples(
        [rendered[index] for index in wanted], optimizer_step=optimizer_step)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab", default=str(ROOT / "src/rwkv_lab/assets/rwkv_vocab_v20230424.txt"))
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "batch",
        "eval_every",
        "eval_examples",
        "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.device != "cuda":
        parser.error("--device must be cuda")
    return args


def train(
    args: argparse.Namespace,
    *,
    worker_components: Any | None = None,
    worker_step_profiler: Any | None = None,
    worker_observability: Any | None = None,
    worker_controls: Any | None = None,
    worker_eval_examples: Any | None = None,
) -> dict[str, Any]:
    """Execute one native-head arm under optional sealed TrainVM services."""

    component_evidence = None
    component_composition_digest = None
    learning_rate_schedule = None
    weight_decay_schedule = None
    if worker_components is not None:
        worker_components.require_implementation(
            "optimizer",
            category="optimizer",
            allowed=frozenset({"rwkv_lab.optimizer.torch_adamw_no_decay.v2"}),
        )
        if dict(
            worker_components.configuration("optimizer", category="optimizer")
        ) != {
            "learning_rate": args.lr,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1.0e-8,
            "foreach": False,
            "fused": True,
        }:
            raise ValueError(
                "authority optimizer composition disagrees with native-head configuration"
            )
        worker_components.require_implementation(
            "learning_rate",
            category="learning_rate_schedule",
            allowed=frozenset({"rwkv_lab.schedule.constant.v1"}),
        )
        if dict(
            worker_components.configuration(
                "learning_rate", category="learning_rate_schedule"
            )
        ):
            raise ValueError("native-head constant learning-rate config must be empty")
        if dict(
            worker_components.configuration(
                "gradient_clipping", category="gradient_clipping"
            )
        ) != {
            "max_norm": args.grad_clip,
            "norm_type": 2.0,
            "error_if_nonfinite": False,
        }:
            raise ValueError(
                "authority gradient clipping disagrees with native-head configuration"
            )
        if dict(
            worker_components.configuration(
                "weight_decay", category="weight_decay_schedule"
            )
        ) != {"weight_decay": args.weight_decay}:
            raise ValueError(
                "authority weight decay disagrees with native-head configuration"
            )
        worker_components.require_implementation(
            "precision",
            category="precision",
            allowed=frozenset(
                {"rwkv_lab.precision.fp32_parameters_bf16_compute.v1"}
            ),
        )
        component_evidence = dict(worker_components.evidence())
        component_composition_digest = (
            worker_components.composition.composition_digest
        )

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
    parameter_groups = [{"params": trainable, "name": "vision_native_output"}]
    optimizer = (
        worker_components.optimizer(parameter_groups)
        if worker_components is not None
        else torch.optim.AdamW(
            parameter_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
    )
    if worker_components is not None:
        learning_rate_schedule = worker_components.learning_rate_schedule(optimizer)
        weight_decay_schedule = worker_components.weight_decay_schedule(optimizer)

    vocab = WorldVocab(args.vocab)
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
    loader_args = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
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
    resume_path = (
        Path(args.resume_from)
        if args.resume_from
        else checkpoint_path
        if args.resume == "auto" and checkpoint_path.is_file()
        else None
    )
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(saved.get("schema", -1)) != SCHEMA:
            raise ValueError("native arm checkpoint schema mismatch")
        if worker_components is not None and (
            saved.get("component_evidence") != component_evidence
            or saved.get("component_composition_digest")
            != component_composition_digest
        ):
            raise ValueError("native-head checkpoint component identity mismatch")
        if worker_controls is not None:
            control_state = saved.get("worker_control_state")
            if not isinstance(control_state, dict):
                raise ValueError("native-head checkpoint control state is missing")
            worker_controls.verify_checkpoint_state(control_state)
        _load_head_state(native, saved["native_head"])
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        best_eval = float(saved["best_eval"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    # Before the baseline evaluation and long before the loop, so a
    # disagreement with the controller's immutable attempt baseline is
    # diagnosed as itself rather than as a refused first crossing after a full
    # frozen-stack load.
    refuse_ungated_attempt_baseline(
        worker_controls,
        step,
        family="Vision native head",
        can_publish_baseline_evidence=worker_eval_examples is not None,
    )
    gate_baseline = attempt_baseline_gate(worker_controls)
    if (
        gate_baseline is None
        and worker_controls is not None
        and worker_controls.step_zero_eval_gate_required
    ):
        print(
            "attempt-baseline eval gate: already durable at step "
            f"{worker_controls.attempt_baseline_optimizer_step}",
            flush=True,
        )

    checkpoint_directory = out / "checkpoint-current"
    checkpoint_state = checkpoint_directory / "state.pt"

    def reject_live_controls(_effective: object, assignments: object) -> None:
        if assignments:
            raise ValueError("vision native-head v1 has no live-mutable controls")

    def save_checkpoint(path: Path = checkpoint_path) -> None:
        _save_checkpoint(
            path,
            model=native,
            optimizer=optimizer,
            step=step,
            best_eval=best_eval,
            args=args,
            component_evidence=component_evidence,
            component_composition_digest=component_composition_digest,
            worker_control_state=(
                worker_controls.checkpoint_state()
                if worker_controls is not None
                else None
            ),
        )
        if path == checkpoint_path:
            _replace_hardlink(checkpoint_path, checkpoint_state)

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
        "component_evidence": component_evidence,
        "component_composition_digest": component_composition_digest,
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

        if gate_baseline is not None:
            # Scalars, then checkpoint, then examples -- the order the
            # controller's `validate_eval_examples_gate_provenance` requires,
            # and all three before the loop below can reach a mutation. The
            # step is the controller's immutable baseline, never a literal
            # zero: this route resumes, so keying publication to `step == 0`
            # would owe evidence at a step a replacement attempt never reaches
            # again while `pre_optimizer_step` refuses every mutation until it
            # exists.
            baseline_loss = _evaluate(
                eval_loader, eval_rows, max_examples=args.eval_examples,
                rwkv=rwkv, native=native, identity=identity,
                engram=engram, deep_vision=deep_vision,
                baseline_args=baseline_args)
            baseline_ppl = math.exp(min(baseline_loss, 20.0))
            _append_json(log, {
                "kind": "eval", "step": gate_baseline, "loss": baseline_loss,
                "ppl": baseline_ppl, "reason": "attempt_baseline",
                "examples": min(args.eval_examples, len(eval_rows)),
                "deployment_bridge": False,
            })
            if worker_observability is not None:
                worker_observability.publish_if_declared(
                    "eval.loss", baseline_loss, step=gate_baseline)
                worker_observability.publish_if_declared(
                    "eval.perplexity", baseline_ppl, step=gate_baseline)
            wanted = select_heldout_indices(
                len(eval_rows), worker_eval_examples.sample_count)

            def stage_baseline_checkpoint() -> str:
                save_checkpoint()
                return str(checkpoint_directory)

            publish_attempt_baseline_evidence(
                worker_controls=worker_controls,
                worker_observability=worker_observability,
                policy=worker_eval_examples,
                baseline=gate_baseline,
                series_id="vision-native-head-caption",
                identities=[str(eval_rows[index].get("image", index))
                            for index in wanted],
                selector={
                    "eval_data": str(args.eval_data),
                    "eval_rows": len(eval_rows),
                    "sample_count": worker_eval_examples.sample_count,
                },
                examples=_heldout_caption_examples(
                    loader=eval_loader, rows=eval_rows, wanted=wanted,
                    vocab=vocab, optimizer_step=gate_baseline, rwkv=rwkv,
                    native=native, identity=identity, engram=engram,
                    deep_vision=deep_vision, baseline_args=baseline_args),
                stage_checkpoint=stage_baseline_checkpoint,
                resume_grade="compatible",
                state_components=(
                    "component_composition",
                    "control_revision",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_python",
                    "rng_torch",
                ),
            )
            gate_baseline = None

        # Installed for the whole loop, so the ordering below is enforced
        # against every optimizer instance in the process rather than the one
        # this trainer constructed. A fused update, a second optimizer, or a
        # later edit that moves the crossing below the mutation all fail closed
        # here instead of mutating parameters the controller never authorized.
        # The controller-facing call the sentinel crosses. It is None when this
        # trainer runs outside TrainVM authority; the sentinel still binds every
        # mutation to one crossing, so the ordering discipline is identical and a
        # standalone run cannot silently acquire a second update path either.
        pre_optimizer_step = (
            (lambda next_step: worker_controls.pre_optimizer_step(
                next_step, reject_live_controls
            ))
            if worker_controls is not None
            else None
        )
        mutation_sentinel = OptimizerMutationSentinel()
        with mutation_sentinel.installed():
            while step < args.steps and not stop:
                batches = (
                    worker_step_profiler.track_input(train_loader)
                    if worker_step_profiler is not None
                    else train_loader
                )
                for indices, moon, fusion in batches:
                    if step >= args.steps or stop:
                        break
                    if worker_controls is not None:
                        worker_controls.microbatch(step + 1, reject_live_controls)
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
                    gnorm = (
                        worker_components.gradient_clipping(trainable)
                        if worker_components is not None
                        else torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    )
                    # The mandatory pre-mutation boundary, immediately before
                    # the only optimizer mutation in this loop. A controller
                    # refusal leaves the sentinel disarmed, so the `step()`
                    # below raises rather than mutating unguarded. This
                    # replaces the post-mutation `worker_controls
                    # .optimizer_step` that used to sit after the update: same
                    # safe point, same effective step number, now on the side
                    # of the mutation where a refusal can still prevent it.
                    mutation_sentinel.cross(step + 1, pre_optimizer_step)
                    optimizer.step()
                    step += 1
                    if learning_rate_schedule is not None:
                        learning_rate_schedule.step()
                    if weight_decay_schedule is not None:
                        weight_decay_schedule.step(step)
                    if worker_step_profiler is not None:
                        worker_step_profiler.step(step)
                    if worker_observability is not None:
                        worker_observability.optimizer_step(step)
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
                    if worker_observability is not None:
                        for name, value in (
                            ("train.loss", event["loss"]),
                            ("train.learning_rate", event["lr"]),
                            ("train.gradient_norm", event["gnorm"]),
                            ("train.tokens_per_second", event["tok_per_sec"]),
                            ("train.step_seconds", event["step_seconds"]),
                            ("train.cross_entropy", event["ce_loss"]),
                        ):
                            worker_observability.publish_if_declared(
                                name, value, step=step
                            )
                    _atomic_json(status_path, {
                        "state": "training", "step": step,
                        "loss": event["loss"], "updated": time.time(),
                        "step_seconds": elapsed, "deployment_bridge": False,
                        "trainable_scope": "vision_compressor.native_output_head",
                    })

                    if step % args.checkpoint_every == 0:
                        save_checkpoint()
                        _append_json(log, {"kind": "checkpoint", "step": step,
                                           "reason": "periodic", "path": str(checkpoint_path)})

                    if step % args.eval_every == 0:
                        if worker_controls is not None:
                            worker_controls.evaluation(step, reject_live_controls)
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
                        if worker_observability is not None:
                            worker_observability.publish_if_declared(
                                "eval.loss", eval_loss, step=step
                            )
                            worker_observability.publish_if_declared(
                                "eval.perplexity", ppl, step=step
                            )
                        if eval_loss < best_eval:
                            best_eval = eval_loss
                            best_path = best_dir / f"native_head_step_{step:08d}.pt"
                            save_checkpoint(best_path)
                            _atomic_json(best_dir / "best.json", {
                                "step": step, "loss": eval_loss, "ppl": ppl,
                                "checkpoint": best_path.name,
                                "baseline_ppl": config["baseline_best_ppl"],
                                "deployment_bridge": False,
                            })
                        save_checkpoint()
                        _atomic_json(status_path, {
                            "state": "training", "step": step,
                            "eval_loss": eval_loss, "eval_ppl": ppl,
                            "baseline_ppl": config["baseline_best_ppl"],
                            "updated": time.time(), "deployment_bridge": False,
                        })

                    checkpoint_requested = bool(
                        worker_controls is not None
                        and worker_controls.checkpoint_boundary_requested
                    )
                    if checkpoint_requested:
                        worker_controls.checkpoint(step, reject_live_controls)
                        with (
                            worker_observability.keepalive(step, "checkpointing")
                            if worker_observability is not None
                            else nullcontext()
                        ):
                            save_checkpoint()
                        if worker_controls.checkpoint_completion_requested:
                            worker_controls.publish_requested_checkpoint_directory(
                                str(checkpoint_directory),
                                optimizer_step=step,
                                resume_grade="compatible",
                                state_components=(
                                    "component_composition",
                                    "control_revision",
                                    "model",
                                    "optimizer",
                                    "rng_accelerator",
                                    "rng_python",
                                    "rng_torch",
                                ),
                            )

    save_checkpoint()
    _atomic_json(status_path, {
        "state": "stopped" if stop else "complete", "step": step,
        "updated": time.time(), "deployment_bridge": False,
    })
    return {
        "status": "interrupted" if stop else "complete",
        "step": step,
        "checkpoint": str(checkpoint_directory.resolve()),
        "best_eval": best_eval,
    }


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
