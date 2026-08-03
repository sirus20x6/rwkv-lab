"""Train a standalone raw-pixel Vision-RWKV from the frozen teacher codec.

Training graph (teacher side is no-grad):

    cached MoonViT/SigLIP2/DINOv2/SAM -> frozen 54M compressor -> canonical target
    raw image pixels                  -> trainable Vision-RWKV   -> canonical/native
                                                                   |
                                                     frozen 2.9B caption RWKV

The deployable artifact is the Vision-RWKV alone; no teacher cache, tower, or
canonical compressor is an inference dependency.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rwkv_lab.vision_compressor_features import (
    FrozenTeacherCompressor,
    NativePrefixIdentity,
)
from rwkv_lab.vision_native_train import _load_frozen_caption_stack
from rwkv_lab.vision_rwkv_student import VisionRWKVConfig, VisionRWKVStudent
from rwkv_lab.vision_teacher_compressor import (
    EpochBatchSampler,
    TeacherCacheDataset,
    compressor_loss,
    relational_loss,
    split_cached_features,
)
from rwkv_lab.vision_train import (
    WorldVocab,
    load_examples,
    make_batch,
    multimodal_loss,
    prepare_examples,
    supervised_positions,
    visual_insert_positions,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(handle, value: dict) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()


def _durable_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def image_tensor(path: Path, size: int) -> tuple[Tensor, Tensor]:
    """Aspect-preserving letterbox preprocessing and geometry metadata."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(resized, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    left, top = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas.paste(image, (left, top))
    array = np.asarray(canvas, dtype=np.float32).copy()
    pixels = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    geometry = torch.tensor((
        max(-4.0, min(4.0, math.log(max(width, 1) / max(height, 1)))),
        resized[0] / size,
        resized[1] / size,
    ), dtype=torch.float32)
    return pixels, geometry


class PixelTeacherDataset(Dataset):
    """Keep raw pixels, captions, and paired teacher caches on one row index."""
    def __init__(self, manifest: Path, rows: list[dict], cache: TeacherCacheDataset,
                 image_size: int):
        if len(rows) != len(cache):
            raise ValueError(f"prepared rows/cache mismatch: {len(rows)} != {len(cache)}")
        self.manifest, self.rows, self.cache = manifest, rows, cache
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        pixels, geometry = image_tensor(Path(self.rows[index]["image"]), self.image_size)
        moon, fusion = self.cache[index]
        return index, pixels, geometry, moon, fusion


def _teacher_cache(manifest: Path, args: argparse.Namespace) -> TeacherCacheDataset:
    return TeacherCacheDataset(
        manifest, Path(args.moon_cache), Path(args.fusion_cache),
        moon_checkpoint=Path(args.moonvit), siglip2=args.siglip2,
        dinov2=args.dinov2, sam=args.sam)


def _native_head(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "native_head" not in blob:
        raise ValueError(f"{path} is not a calibrated native-head checkpoint")
    return blob["native_head"]


def _caption_loss(*, native: Tensor, chosen: list[dict], rwkv,
                  identity: NativePrefixIdentity, engram, deep_vision,
                  nextlat, grounding, baseline_args: dict,
                  training: bool) -> tuple[Tensor, dict[str, Tensor], int]:
    ids, labels, mask = make_batch(chosen, device="cuda")
    positions = supervised_positions(chosen, identity.prefix_tokens, device="cuda")
    starts = visual_insert_positions(chosen)
    loss, metrics = multimodal_loss(
        rwkv, identity, None, (), ids, labels, mask,
        features=list(native.unbind(0)), selected_positions=positions,
        visual_starts=starts, engram=engram, deep_vision=deep_vision,
        nextlat=nextlat if training else None,
        nextlat_weight=(float(baseline_args.get("nextlat_weight", 0.0))
                        if training else 0.0),
        nextlat_kl_weight=(float(baseline_args.get("nextlat_kl_weight", 0.0))
                           if training else 0.0),
        grounding=grounding if training else None,
        grounding_contrastive_weight=(float(
            baseline_args.get("grounding_contrastive_weight", 0.0))
            if training else 0.0),
        grounding_early_tokens=(int(baseline_args.get("grounding_early_tokens", 0))
                                if training else 0),
        grounding_early_weight=float(baseline_args.get("grounding_early_weight", 1.0)),
    )
    return loss, metrics, int((labels != -100).sum())


def _representation_loss(student_latent: Tensor, streams: Sequence[Tensor],
                         target: Tensor, compressor: FrozenTeacherCompressor,
                         args: argparse.Namespace) -> tuple[Tensor, dict[str, Tensor]]:
    smooth = F.smooth_l1_loss(student_latent.float(), target.float())
    cosine = (1 - F.cosine_similarity(
        student_latent.float(), target.float(), dim=-1)).mean()
    relation = relational_loss(student_latent, target)
    predictions = [head(student_latent) for head in compressor.model.reconstruction]
    decoded, decoded_metrics = compressor_loss(
        student_latent, predictions, streams,
        relational_weight=args.teacher_relational_weight,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        diversity_weight=args.diversity_weight)
    latent = smooth + cosine + args.latent_relational_weight * relation
    total = latent + args.teacher_reconstruction_weight * decoded
    metrics = {
        "latent_smooth_l1": smooth.detach(),
        "latent_cosine": cosine.detach(),
        "latent_relational": relation.detach(),
        "teacher_decoded_loss": decoded.detach(),
        **{f"student_{key.replace('/', '_')}": value.detach()
           for key, value in decoded_metrics.items()},
    }
    return total, metrics


def _forward_batch(*, indices: Tensor, pixels: Tensor, geometry: Tensor,
                   moon: Tensor, fusion: Tensor, rows: list[dict], student,
                   compressor, rwkv, identity, engram, deep_vision, nextlat,
                   grounding, baseline_args, args, training: bool):
    chosen = [rows[int(index)] for index in indices.tolist()]
    pixels = pixels.cuda(non_blocking=True)
    geometry = geometry.cuda(non_blocking=True)
    moon = moon.cuda(non_blocking=True)
    fusion = fusion.cuda(non_blocking=True)
    streams = split_cached_features(moon, fusion)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        target, _ = compressor.model(streams)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        canonical, native = student(pixels, geometry)
        representation, repr_metrics = _representation_loss(
            canonical, streams, target, compressor, args)
        caption, caption_metrics, tokens = _caption_loss(
            native=native, chosen=chosen, rwkv=rwkv, identity=identity,
            engram=engram, deep_vision=deep_vision, nextlat=nextlat,
            grounding=grounding, baseline_args=baseline_args,
            training=training)
        total = representation + args.caption_weight * caption
    metrics = {**repr_metrics, **caption_metrics,
               "representation_loss": representation.detach(),
               "caption_loss": caption.detach()}
    return total, metrics, tokens


@torch.no_grad()
def _evaluate(loader: DataLoader, rows: list[dict], *, student, compressor,
              rwkv, identity, engram, deep_vision, baseline_args, args) -> dict[str, float]:
    student.eval()
    totals: dict[str, float] = {}
    examples = tokens = 0
    for indices, pixels, geometry, moon, fusion in loader:
        if examples >= args.eval_examples:
            break
        remaining = args.eval_examples - examples
        if len(indices) > remaining:
            indices, pixels, geometry = indices[:remaining], pixels[:remaining], geometry[:remaining]
            moon, fusion = moon[:remaining], fusion[:remaining]
        _, metrics, batch_tokens = _forward_batch(
            indices=indices, pixels=pixels, geometry=geometry, moon=moon,
            fusion=fusion, rows=rows, student=student, compressor=compressor,
            rwkv=rwkv, identity=identity, engram=engram,
            deep_vision=deep_vision, nextlat=None, grounding=None,
            baseline_args=baseline_args, args=args, training=False)
        count = len(indices)
        examples += count
        tokens += batch_tokens
        for name, value in metrics.items():
            weight = batch_tokens if name in ("caption_loss", "ce_loss") else count
            totals[name] = totals.get(name, 0.0) + float(value) * weight
    result = {name: value / max(tokens if name in ("caption_loss", "ce_loss")
                                else examples, 1)
              for name, value in totals.items()}
    student.train()
    return result


def _checkpoint(*, student, optimizer, sampler, step: int, best_ppl: float,
                args, config: VisionRWKVConfig, component_evidence=None,
                component_composition_digest=None, worker_control_state=None) -> dict:
    return {
        "schema": SCHEMA, "step": int(step), "best_ppl": float(best_ppl),
        "student": {name: value.detach().cpu()
                    for name, value in student.state_dict().items()},
        "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
        "config": asdict(config), "args": vars(args),
        "component_evidence": component_evidence,
        "component_composition_digest": component_composition_digest,
        "worker_control_state": worker_control_state,
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runs/moonvit_rwkv_eight_hour_grounded/best/ckpt_step_00021900.pt")
    parser.add_argument("--compressor", default="models/vision/teacher-compressor-so400m-dinov2-sam-128x1024-v1.pt")
    parser.add_argument("--native-head", default="runs/native_compressor_rwkv_arm2/best/native_head_step_00004900.pt")
    parser.add_argument("--data", default="curated_vision/vision_next_shard_000_ocr10_train.jsonl")
    parser.add_argument("--eval-data", default="curated_vision/vision_next_shard_000_ocr10_eval.jsonl")
    parser.add_argument("--moon-cache", default="/thearray/downloads/cache/moe-mla/moonvit_next_128_shard_000_ocr10")
    parser.add_argument("--fusion-cache", default="/thearray/downloads/cache/moe-mla/fusion_so400m_next_128_shard_000_ocr10")
    parser.add_argument("--moonvit", default="models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    parser.add_argument("--siglip2", default="models/vision/siglip2-so400m-patch16-512")
    parser.add_argument("--dinov2", default="models/vision/dinov2-base")
    parser.add_argument("--sam", default="models/vision/sam-vit-base")
    parser.add_argument("--out", default="runs/vision_rwkv_student_1p22b")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--caption-weight", type=float, default=0.1)
    parser.add_argument("--teacher-reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--latent-relational-weight", type=float, default=0.2)
    parser.add_argument("--teacher-relational-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vocab",
        default=str(ROOT / "src/rwkv_lab/assets/rwkv_vocab_v20230424.txt"),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=26)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--ffn-hidden", type=int, default=7168)
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.device != "cuda":
        parser.error("--device must be cuda")
    return args


def train(
    args: argparse.Namespace,
    *,
    worker_components=None,
    worker_step_profiler=None,
    worker_observability=None,
    worker_controls=None,
) -> dict:
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
                "authority optimizer composition disagrees with student configuration"
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
            raise ValueError("student constant learning-rate config must be empty")
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
                "authority gradient clipping disagrees with student configuration"
            )
        if dict(
            worker_components.configuration(
                "weight_decay", category="weight_decay_schedule"
            )
        ) != {"weight_decay": args.weight_decay}:
            raise ValueError(
                "authority weight decay disagrees with student configuration"
            )
        worker_components.require_implementation(
            "precision",
            category="precision",
            allowed=frozenset(
                {"rwkv_lab.precision.bf16_parameters_fp32_reductions.v1"}
            ),
        )
        component_evidence = dict(worker_components.evidence())
        component_composition_digest = (
            worker_components.composition.composition_digest
        )
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("CUDA is required for student training")
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status_path, log_path = out / "status.json", out / "train.jsonl"
    last_path, best_dir = out / "last.pt", out / "best"
    best_dir.mkdir(exist_ok=True)
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    config = VisionRWKVConfig(
        image_size=args.image_size, grid_size=args.grid_size,
        canonical_tokens=args.grid_size * args.grid_size // 2,
        canonical_width=1024, native_tokens=64, native_width=2560,
        hidden_size=args.hidden_size, layers=args.layers,
        head_size=args.head_size, ffn_hidden=args.ffn_hidden,
        checkpoint_blocks=not args.no_checkpoint_blocks)
    config.validate()

    _atomic_json(status_path, {"state": "validating_data", "updated": time.time()})
    baseline_blob = torch.load(args.baseline, map_location="cpu", weights_only=False)
    baseline_args = dict(baseline_blob["args"])
    del baseline_blob
    if int(baseline_args["prefix_tokens"]) != config.native_tokens:
        raise ValueError("student native-token count does not match frozen caption RWKV")
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
    train_cache, eval_cache = _teacher_cache(Path(args.data), args), _teacher_cache(Path(args.eval_data), args)
    train_data = PixelTeacherDataset(Path(args.data), train_rows, train_cache, config.image_size)
    eval_data = PixelTeacherDataset(Path(args.eval_data), eval_rows, eval_cache, config.image_size)
    if args.preflight_only:
        index, pixels, geometry, moon, fusion = train_data[0]
        print({"kind": "vision_rwkv_student_preflight", "ready": True,
               "train": len(train_data), "eval": len(eval_data), "index": index,
               "pixels": list(pixels.shape), "geometry": geometry.tolist(),
               "moon": list(moon.shape), "fusion": list(fusion.shape),
               "config": asdict(config)}, flush=True)
        return {"status": "preflight", "step": 0, "checkpoint": ""}

    _atomic_json(status_path, {"state": "loading_models", "updated": time.time()})
    compressor = FrozenTeacherCompressor.from_checkpoint(
        args.compressor, device="cuda", dtype=torch.bfloat16)
    _, rwkv, engram, deep_vision, nextlat, grounding = _load_frozen_caption_stack(Path(args.baseline))
    identity = NativePrefixIdentity(config.native_tokens, config.native_width).cuda()
    student = VisionRWKVStudent(config)
    if args.resume == "none" or not last_path.is_file():
        student.load_native_head(_native_head(Path(args.native_head)))
    student = student.to(device="cuda", dtype=torch.bfloat16)
    frozen_modules = (compressor, rwkv, engram, deep_vision, nextlat, grounding, identity)
    for module in frozen_modules:
        if module is not None:
            module.requires_grad_(False).eval()
    if any(parameter.requires_grad for module in frozen_modules if module is not None
           for parameter in module.parameters()):
        raise RuntimeError("a teacher or caption module escaped the freeze contract")
    optimizer = (
        worker_components.optimizer(student.parameters())
        if worker_components is not None
        else torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
    )
    if worker_components is not None:
        learning_rate_schedule = worker_components.learning_rate_schedule(optimizer)
        weight_decay_schedule = worker_components.weight_decay_schedule(optimizer)
    sampler = EpochBatchSampler(len(train_data), args.batch, args.seed)
    step, best_ppl = 0, math.inf
    resume_path = (
        Path(args.resume_from)
        if args.resume_from
        else last_path
        if args.resume == "auto" and last_path.is_file()
        else None
    )
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(saved.get("schema", -1)) != SCHEMA or saved.get("config") != asdict(config):
            raise ValueError("student checkpoint schema/config mismatch")
        if worker_components is not None and (
            saved.get("component_evidence") != component_evidence
            or saved.get("component_composition_digest")
            != component_composition_digest
        ):
            raise ValueError("student checkpoint component identity mismatch")
        if worker_controls is not None:
            control_state = saved.get("worker_control_state")
            if not isinstance(control_state, dict):
                raise ValueError("student checkpoint control state is missing")
            worker_controls.verify_checkpoint_state(control_state)
        student.load_state_dict(saved["student"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler_state = dict(saved["sampler"])
        # Batch size is execution geometry, not sample-order identity. Preserve
        # the exact shuffled order/cursor while allowing a larger batch after a
        # measured VRAM smoke run; no example is repeated or skipped.
        sampler_state["batch_size"] = args.batch
        sampler.load_state_dict(sampler_state)
        step, best_ppl = int(saved["step"]), float(saved["best_ppl"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    checkpoint_directory = out / "checkpoint-current"
    checkpoint_state = checkpoint_directory / "state.pt"

    def reject_live_controls(_effective: object, assignments: object) -> None:
        if assignments:
            raise ValueError("vision student v1 has no live-mutable controls")

    def checkpoint_payload() -> dict:
        return _checkpoint(
            student=student,
            optimizer=optimizer,
            sampler=sampler,
            step=step,
            best_ppl=best_ppl,
            args=args,
            config=config,
            component_evidence=component_evidence,
            component_composition_digest=component_composition_digest,
            worker_control_state=(
                worker_controls.checkpoint_state()
                if worker_controls is not None
                else None
            ),
        )

    def save_current_checkpoint() -> None:
        _durable_save(checkpoint_payload(), last_path)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_state.with_name(
            f".{checkpoint_state.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.unlink(missing_ok=True)
            os.link(last_path, temporary)
            os.replace(temporary, checkpoint_state)
        finally:
            temporary.unlink(missing_ok=True)

    loader_args = {"num_workers": args.workers, "pin_memory": True,
                   "persistent_workers": args.workers > 0}
    if args.workers > 0:
        loader_args["prefetch_factor"] = 2
    train_loader = DataLoader(train_data, batch_sampler=sampler, **loader_args)
    eval_loader = DataLoader(eval_data, batch_size=args.batch, shuffle=False,
                             **loader_args)
    config_payload = {**vars(args), "architecture": "raw_pixel_spatial_rwkv_student",
                      "teacher_path": "cached_six_stream_to_frozen_compressor",
                      "deployment_teacher_free": True,
                      "student_parameters": student.parameter_count,
                      "frozen_compressor_parameters": sum(p.numel() for p in compressor.parameters()),
                      "frozen_caption_rwkv_parameters": sum(p.numel() for p in rwkv.parameters()),
                      "student_config": asdict(config),
                      "component_evidence": component_evidence,
                      "component_composition_digest": component_composition_digest}
    _atomic_json(out / "config.json", config_payload)
    mode = "a" if step else "w"
    with log_path.open(mode) as log:
        if step == 0:
            _append_json(log, {"kind": "startup", "step": 0,
                "architecture": config_payload["architecture"],
                "deployment_teacher_free": True,
                "train_examples": len(train_data), "val_examples": len(eval_data),
                "trainable_parameters": student.parameter_count,
                "frozen_compressor_parameters": config_payload["frozen_compressor_parameters"],
                "frozen_rwkv_parameters": config_payload["frozen_caption_rwkv_parameters"]})
        _atomic_json(status_path, {"state": "training", "step": step,
            "updated": time.time(), "trainable_scope": "raw_pixel_vision_rwkv_student",
            "deployment_teacher_free": True})
        while step < args.steps and not stop:
            batches = (
                worker_step_profiler.track_input(train_loader)
                if worker_step_profiler is not None
                else train_loader
            )
            for indices, pixels, geometry, moon, fusion in batches:
                if step >= args.steps or stop:
                    break
                if worker_controls is not None:
                    worker_controls.microbatch(step + 1, reject_live_controls)
                started = time.perf_counter()
                student.train()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics, tokens = _forward_batch(
                    indices=indices, pixels=pixels, geometry=geometry, moon=moon,
                    fusion=fusion, rows=train_rows, student=student,
                    compressor=compressor, rwkv=rwkv, identity=identity,
                    engram=engram, deep_vision=deep_vision, nextlat=nextlat,
                    grounding=grounding, baseline_args=baseline_args,
                    args=args, training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite student loss: {loss}")
                loss.backward()
                gradient_norm = (
                    worker_components.gradient_clipping(student.parameters())
                    if worker_components is not None
                    else torch.nn.utils.clip_grad_norm_(
                        student.parameters(), args.grad_clip
                    )
                )
                optimizer.step()
                sampler.consumed(len(indices))
                step += 1
                if learning_rate_schedule is not None:
                    learning_rate_schedule.step()
                if weight_decay_schedule is not None:
                    weight_decay_schedule.step(step)
                if worker_step_profiler is not None:
                    worker_step_profiler.step(step)
                if worker_observability is not None:
                    worker_observability.optimizer_step(step)
                if worker_controls is not None:
                    worker_controls.optimizer_step(step, reject_live_controls)
                elapsed = time.perf_counter() - started
                event = {"kind": "train", "step": step, "loss": float(loss.detach()),
                         "lr": optimizer.param_groups[0]["lr"],
                         "gnorm": float(gradient_norm), "batch": len(indices),
                         "step_seconds": elapsed, "tok_per_sec": tokens / max(elapsed, 1e-9),
                         **{name: float(value) for name, value in metrics.items()}}
                _append_json(log, event)
                if worker_observability is not None:
                    for name, value in (
                        ("train.loss", event["loss"]),
                        ("train.learning_rate", event["lr"]),
                        ("train.gradient_norm", event["gnorm"]),
                        ("train.tokens_per_second", event["tok_per_sec"]),
                        ("train.step_seconds", event["step_seconds"]),
                        ("train.representation_loss", event["representation_loss"]),
                        ("train.caption_loss", event["caption_loss"]),
                    ):
                        worker_observability.publish_if_declared(
                            name, value, step=step
                        )
                _atomic_json(status_path, {"state": "training", "step": step,
                    "loss": event["loss"], "step_seconds": elapsed,
                    "updated": time.time(),
                    "trainable_scope": "raw_pixel_vision_rwkv_student",
                    "deployment_teacher_free": True})

                checkpoint_due = step % args.checkpoint_every == 0
                eval_due = step % args.eval_every == 0
                best_improved = False
                evaluation = None
                caption_eval = ppl = None
                if eval_due:
                    if worker_controls is not None:
                        worker_controls.evaluation(step, reject_live_controls)
                    _atomic_json(status_path, {"state": "evaluating", "step": step,
                                               "updated": time.time()})
                    evaluation = _evaluate(eval_loader, eval_rows, student=student,
                        compressor=compressor, rwkv=rwkv, identity=identity,
                        engram=engram, deep_vision=deep_vision,
                        baseline_args=baseline_args, args=args)
                    caption_eval = float(evaluation["caption_loss"])
                    ppl = math.exp(min(caption_eval, 20.0))
                    _append_json(log, {"kind": "eval", "step": step,
                        "loss": caption_eval, "ppl": ppl,
                        **{f"eval_{name}": value for name, value in evaluation.items()}})
                    if worker_observability is not None:
                        worker_observability.publish_if_declared(
                            "eval.loss", caption_eval, step=step
                        )
                        worker_observability.publish_if_declared(
                            "eval.perplexity", ppl, step=step
                        )
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_improved = True
                if checkpoint_due or eval_due:
                    _atomic_json(status_path, {"state": "checkpointing", "step": step,
                                               "updated": time.time()})
                    save_current_checkpoint()
                    _append_json(log, {"kind": "checkpoint", "step": step,
                                       "path": str(last_path),
                                       "reason": "eval" if eval_due else "periodic"})
                    if best_improved:
                        destination = best_dir / f"student_step_{step:08d}.pt"
                        temporary = best_dir / f".{destination.name}.tmp"
                        temporary.unlink(missing_ok=True)
                        os.link(last_path, temporary)
                        os.replace(temporary, destination)
                        _atomic_json(best_dir / "best.json", {
                            "step": step, "loss": caption_eval, "ppl": ppl,
                            "representation_loss": evaluation.get("representation_loss"),
                            "checkpoint": destination.name})
                        for old in best_dir.glob("student_step_*.pt"):
                            if old != destination:
                                old.unlink(missing_ok=True)
                if eval_due:
                    _atomic_json(status_path, {"state": "training", "step": step,
                                               "updated": time.time(),
                                               "trainable_scope": "raw_pixel_vision_rwkv_student",
                                               "deployment_teacher_free": True})
                checkpoint_requested = bool(
                    worker_controls is not None
                    and worker_controls.checkpoint_boundary_requested
                )
                if checkpoint_requested:
                    worker_controls.checkpoint(step, reject_live_controls)
                    save_current_checkpoint()
                    if worker_controls.checkpoint_completion_requested:
                        worker_controls.publish_requested_checkpoint_directory(
                            str(checkpoint_directory),
                            optimizer_step=step,
                            resume_grade="compatible",
                            state_components=(
                                "component_composition",
                                "control_revision",
                                "data_cursor",
                                "model",
                                "optimizer",
                                "rng_accelerator",
                                "rng_python",
                                "rng_torch",
                            ),
                        )
            # A fully consumed sampler advances itself on the next iterator.
        if step and (not last_path.is_file() or stop or step % args.checkpoint_every):
            _atomic_json(status_path, {"state": "checkpointing", "step": step,
                                       "updated": time.time()})
            save_current_checkpoint()
            _append_json(log, {"kind": "checkpoint", "step": step,
                               "path": str(last_path),
                               "reason": "stop" if stop else "final"})
    _atomic_json(status_path, {"state": "stopped" if stop else "complete",
                               "step": step, "updated": time.time(),
                               "deployment_teacher_free": True})
    if not checkpoint_state.is_file():
        save_current_checkpoint()
    return {
        "status": "interrupted" if stop else "complete",
        "step": step,
        "checkpoint": str(checkpoint_directory.resolve()),
        "best_ppl": best_ppl,
    }


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
