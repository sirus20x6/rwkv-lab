"""Train a compact canonical latent codec from frozen vision-teacher caches.

This is Phase 1 of the multi-teacher vision distillation plan.  It consumes the
three staged MoonViT tensors and the aligned SigLIP2/DINOv2/SAM tensor without
loading any teacher model.  Its deployment-independent output contract is
``[batch, 128, 1024]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from rwkv_lab.moonvit import checkpoint_fingerprint, feature_cache_key
from rwkv_lab.vision_fusion import VisionTowerConfig, aligned_feature_cache_key


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
STREAM_WIDTHS = (4608, 4608, 4608, 1152, 768, 256)
STREAM_NAMES = ("moon_8", "moon_17", "moon_26", "siglip2", "dinov2", "sam")


@dataclass(frozen=True)
class CompressorConfig:
    tokens: int = 128
    latent_width: int = 1024
    stem_width: int = 192
    layers: int = 4
    heads: int = 16
    ff_mult: int = 2
    dropout: float = 0.0


def normalized_target(value: Tensor) -> Tensor:
    """Per-token normalization without trainable teacher-side parameters."""
    return F.layer_norm(value.float(), (value.shape[-1],)).to(value.dtype)


class CanonicalTeacherCompressor(nn.Module):
    """Reconcile six aligned teacher streams into one fixed latent array."""

    def __init__(self, config: CompressorConfig = CompressorConfig()):
        super().__init__()
        if config.tokens < 1 or config.latent_width % config.heads:
            raise ValueError("tokens must be positive and latent_width divisible by heads")
        self.config = config
        self.stems = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.stem_width))
            for width in STREAM_WIDTHS
        ])
        self.stream_identity = nn.Parameter(
            torch.empty(len(STREAM_WIDTHS), 1, config.stem_width))
        self.position = nn.Parameter(torch.empty(1, config.tokens, config.latent_width))
        self.fuse = nn.Linear(config.stem_width * len(STREAM_WIDTHS),
                              config.latent_width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_width, nhead=config.heads,
            dim_feedforward=config.latent_width * config.ff_mult,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, config.layers,
                                           enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.latent_width)
        self.reconstruction = nn.ModuleList(
            nn.Linear(config.latent_width, width) for width in STREAM_WIDTHS)
        nn.init.normal_(self.stream_identity, std=0.02)
        nn.init.normal_(self.position, std=0.01)

    def forward(self, streams: Sequence[Tensor],
                keep_mask: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
        if len(streams) != len(STREAM_WIDTHS):
            raise ValueError(f"expected {len(STREAM_WIDTHS)} streams")
        batch = streams[0].shape[0]
        if keep_mask is None:
            keep_mask = torch.ones(batch, len(streams), device=streams[0].device,
                                   dtype=torch.bool)
        if tuple(keep_mask.shape) != (batch, len(streams)):
            raise ValueError("invalid teacher keep mask")
        encoded = []
        for index, (value, stem, width) in enumerate(
                zip(streams, self.stems, STREAM_WIDTHS)):
            if tuple(value.shape[1:]) != (self.config.tokens, width):
                raise ValueError(f"invalid {STREAM_NAMES[index]} shape {tuple(value.shape)}")
            item = stem(value) + self.stream_identity[index]
            item = item * keep_mask[:, index, None, None].to(item.dtype)
            encoded.append(item)
        latent = self.fuse(torch.cat(encoded, dim=-1)) + self.position
        latent = self.final_norm(self.mixer(latent))
        return latent, [head(latent) for head in self.reconstruction]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def split_cached_features(moon: Tensor, fusion: Tensor) -> list[Tensor]:
    """Convert cache payloads into six token-aligned streams."""
    if moon.ndim != 5 or tuple(moon.shape[1:]) != (3, 128, 4, 1152):
        raise ValueError(f"invalid staged MoonViT batch {tuple(moon.shape)}")
    if fusion.ndim != 3 or tuple(fusion.shape[1:]) != (128, 2176):
        raise ValueError(f"invalid aligned fusion batch {tuple(fusion.shape)}")
    moon_streams = list(moon.flatten(3).unbind(1))
    sam = fusion[..., 1920:2176]
    if not torch.isfinite(sam).all() or not bool(sam.abs().max() > 0):
        raise ValueError("aligned fusion payload has no valid SAM feature stream")
    return moon_streams + [fusion[..., :1152], fusion[..., 1152:1920], sam]


def teacher_keep_mask(batch: int, streams: int, probability: float,
                      device: torch.device) -> Tensor:
    if not 0 <= probability < 1:
        raise ValueError("teacher dropout must be in [0, 1)")
    keep = torch.rand(batch, streams, device=device) >= probability
    empty = ~keep.any(dim=1)
    if empty.any():
        replacement = torch.randint(streams, (int(empty.sum()),), device=device)
        keep[empty] = False
        keep[empty, replacement] = True
    return keep


def relational_loss(prediction: Tensor, target: Tensor, stride: int = 4) -> Tensor:
    pred = F.normalize(prediction[:, ::stride].float(), dim=-1)
    truth = F.normalize(target[:, ::stride].float(), dim=-1)
    return F.mse_loss(pred @ pred.transpose(1, 2),
                      truth @ truth.transpose(1, 2))


def collapse_loss(latent: Tensor, covariance_dims: int = 256) -> tuple[Tensor, Tensor, Tensor]:
    flat = latent.float().reshape(-1, latent.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    width = min(covariance_dims, flat.shape[-1])
    centered = flat[:, :width] - flat[:, :width].mean(dim=0)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    off_diagonal = covariance.square().sum() - covariance.diag().square().sum()
    covariance_penalty = off_diagonal / max(width * (width - 1), 1)
    token_unit = F.normalize(latent.float(), dim=-1)
    adjacent_similarity = (token_unit[:, 1:] * token_unit[:, :-1]).sum(-1).mean()
    diversity = F.relu(adjacent_similarity - 0.8)
    return variance, covariance_penalty, diversity


def compressor_loss(latent: Tensor, predictions: Sequence[Tensor],
                    targets: Sequence[Tensor], *, relational_weight: float,
                    variance_weight: float, covariance_weight: float,
                    diversity_weight: float) -> tuple[Tensor, dict[str, Tensor]]:
    metrics: dict[str, Tensor] = {}
    reconstruction = latent.new_zeros((), dtype=torch.float32)
    relations = latent.new_zeros((), dtype=torch.float32)
    for name, prediction, target in zip(STREAM_NAMES, predictions, targets):
        truth = normalized_target(target)
        pred = prediction.float()
        smooth = F.smooth_l1_loss(pred, truth.float())
        cosine = (1 - F.cosine_similarity(pred, truth.float(), dim=-1)).mean()
        item = smooth + cosine
        relation = relational_loss(pred, truth)
        metrics[f"recon/{name}"] = item.detach()
        metrics[f"relation/{name}"] = relation.detach()
        reconstruction = reconstruction + item
        relations = relations + relation
    reconstruction = reconstruction / len(targets)
    relations = relations / len(targets)
    variance, covariance, diversity = collapse_loss(latent)
    total = (reconstruction + relational_weight * relations
             + variance_weight * variance + covariance_weight * covariance
             + diversity_weight * diversity)
    metrics.update({"loss": total.detach(), "reconstruction": reconstruction.detach(),
                    "relational": relations.detach(), "variance": variance.detach(),
                    "covariance": covariance.detach(), "diversity": diversity.detach(),
                    "latent_std": latent.float().std(unbiased=False).detach()})
    return total, metrics


def _resolve_image(value: str) -> Path:
    image = Path(value)
    return (image if image.is_absolute() else ROOT / image).resolve()


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class TeacherCacheDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, manifest: Path, moon_cache: Path, fusion_cache: Path,
                 *, moon_checkpoint: Path, siglip2: str, dinov2: str, sam: str):
        self.manifest = manifest.resolve()
        rows = [json.loads(line) for line in manifest.open() if line.strip()]
        moon_fingerprint = checkpoint_fingerprint(moon_checkpoint)
        fusion_fingerprint = VisionTowerConfig(
            siglip2=siglip2, dinov2=dinov2, sam=sam,
            siglip_width=1152).fingerprint()
        moon_names = {entry.name for entry in moon_cache.iterdir()
                      if entry.is_file() and entry.suffix == ".pt"}
        fusion_names = {entry.name for entry in fusion_cache.iterdir()
                        if entry.is_file() and entry.suffix == ".pt"}
        self.entries: list[tuple[Path, Path]] = []
        missing: list[str] = []
        for row in rows:
            image = _resolve_image(row["image"])
            stat = image.stat()
            moon_name = feature_cache_key(
                image, max_input_patches=1024, prefix_tokens=128,
                vision_fingerprint=moon_fingerprint, source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns, tap_layers=(8, 17, 26),
                view_mode="full-quadrants")
            fusion_name = aligned_feature_cache_key(
                image, tokens=128, tower_fingerprint=fusion_fingerprint,
                source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
            if moon_name not in moon_names or fusion_name not in fusion_names:
                missing.append(str(image))
            else:
                self.entries.append((moon_cache / moon_name,
                                     fusion_cache / fusion_name))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest rows lack paired teacher caches; first={missing[0]}")
        if not self.entries:
            raise ValueError(f"empty manifest {manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        moon_path, fusion_path = self.entries[index]
        moon = torch.load(moon_path, map_location="cpu", weights_only=True)
        fusion = torch.load(fusion_path, map_location="cpu", weights_only=True)
        if tuple(moon.shape) != (3, 128, 4, 1152) or not torch.isfinite(moon).all():
            raise ValueError(f"invalid MoonViT cache payload {moon_path}")
        if tuple(fusion.shape) != (128, 2176) or not torch.isfinite(fusion).all():
            raise ValueError(f"invalid fusion cache payload {fusion_path}")
        return moon, fusion


class EpochBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches with an exactly resumable sample cursor."""
    def __init__(self, size: int, batch_size: int, seed: int,
                 epoch: int = 0, cursor: int = 0):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.cursor = epoch, cursor

    def order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        order = self.order()
        for start in range(self.cursor, self.size, self.batch_size):
            yield order[start:min(start + self.batch_size, self.size)]

    def __len__(self) -> int:
        return math.ceil(max(self.size - self.cursor, 0) / self.batch_size)

    def consumed(self, count: int) -> None:
        self.cursor += count
        if self.cursor >= self.size:
            self.epoch += 1
            self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor, "size": self.size,
                "batch_size": self.batch_size, "seed": self.seed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        expected = {"size": self.size, "batch_size": self.batch_size,
                    "seed": self.seed}
        if any(int(state[name]) != value for name, value in expected.items()):
            raise ValueError("sampler contract differs from checkpoint")
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])


def _cpu_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _durable_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checkpoint_payload(*, model: nn.Module, optimizer: torch.optim.Optimizer,
                       sampler: EpochBatchSampler, step: int, best_eval: float,
                       args: argparse.Namespace, config: CompressorConfig) -> dict:
    return {"schema": SCHEMA, "step": step, "best_eval": best_eval,
            "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(), "config": asdict(config),
            "manifests": {"train": _manifest_sha256(args.data),
                          "eval": _manifest_sha256(args.eval_data)},
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()}, "args": vars(args)}


@torch.no_grad()
def evaluate(model: CanonicalTeacherCompressor, loader: DataLoader,
             args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for moon, fusion in loader:
        moon, fusion = moon.to(device, non_blocking=True), fusion.to(device, non_blocking=True)
        streams = split_cached_features(moon, fusion)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            latent, predictions = model(streams)
            _, metrics = compressor_loss(
                latent, predictions, streams,
                relational_weight=args.relational_weight,
                variance_weight=args.variance_weight,
                covariance_weight=args.covariance_weight,
                diversity_weight=args.diversity_weight)
        count = moon.shape[0]
        examples += count
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
    model.train()
    return {name: value / examples for name, value in totals.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--moon-cache", type=Path, required=True)
    parser.add_argument("--fusion-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--moonvit", type=Path, default=ROOT / "models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    parser.add_argument("--siglip2", default=str(ROOT / "models/vision/siglip2-so400m-patch16-512"))
    parser.add_argument("--dinov2", default=str(ROOT / "models/vision/dinov2-base"))
    parser.add_argument("--sam", default=str(ROOT / "models/vision/sam-vit-base"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--teacher-dropout", type=float, default=0.15)
    parser.add_argument("--relational-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--init-from", type=Path,
                        help="transfer model/optimizer/global step to a new manifest")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("steps", "batch", "eval_every", "checkpoint_every", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    config = CompressorConfig()
    print({"kind": "compressor_preflight", "phase": "mapping_train_cache"}, flush=True)
    train = TeacherCacheDataset(args.data, args.moon_cache, args.fusion_cache,
                                moon_checkpoint=args.moonvit, siglip2=args.siglip2,
                                dinov2=args.dinov2, sam=args.sam)
    evaluation = TeacherCacheDataset(
        args.eval_data, args.moon_cache, args.fusion_cache,
        moon_checkpoint=args.moonvit, siglip2=args.siglip2,
        dinov2=args.dinov2, sam=args.sam)
    sample_moon, sample_fusion = train[0]
    sample_streams = split_cached_features(
        sample_moon.unsqueeze(0), sample_fusion.unsqueeze(0))
    print({"kind": "compressor_preflight", "ready": True, "train": len(train),
           "eval": len(evaluation), "moon_shape": list(sample_moon.shape),
           "fusion_shape": list(sample_fusion.shape), "sam_dense": False,
           "sam_input": "pooled_128x256",
           "sam_abs_mean": float(sample_streams[-1].float().abs().mean())},
          flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compressor training")
    device = torch.device("cuda")
    model = CanonicalTeacherCompressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, fused=True)
    sampler = EpochBatchSampler(len(train), args.batch, args.seed)
    step, best_eval = 0, math.inf
    last = args.out / "last.pt"
    if args.resume == "auto" and last.is_file():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved.get("schema") != SCHEMA or saved.get("config") != asdict(config):
            raise ValueError("checkpoint schema/config does not match this run")
        expected = {"train": _manifest_sha256(args.data),
                    "eval": _manifest_sha256(args.eval_data)}
        if saved.get("manifests") != expected:
            raise ValueError("checkpoint manifest fingerprints do not match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"])
        step, best_eval = int(saved["step"]), float(saved["best_eval"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        print({"kind": "compressor_resume", "step": step,
               "epoch": sampler.epoch, "cursor": sampler.cursor}, flush=True)
    elif args.init_from is not None:
        saved = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if saved.get("schema") != SCHEMA or saved.get("config") != asdict(config):
            raise ValueError("initial checkpoint schema/config does not match this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # Evaluation losses are only comparable within one manifest. Carry the
        # optimizer and global step forward, but let the new slice establish
        # its own best checkpoint.
        step, best_eval = int(saved["step"]), math.inf
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        print({"kind": "compressor_transfer", "step": step,
               "source": str(args.init_from), "sampler_reset": True}, flush=True)
    eval_loader = DataLoader(evaluation, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0)
    # trainboard discovers runs by this canonical filename. Preserve the early
    # prototype name as a hard-link alias when upgrading an in-flight run.
    metrics_path = args.out / "train.jsonl"
    legacy_metrics_path = args.out / "metrics.jsonl"
    if not metrics_path.exists() and legacy_metrics_path.exists():
        os.link(legacy_metrics_path, metrics_path)
    elif not legacy_metrics_path.exists() and metrics_path.exists():
        os.link(metrics_path, legacy_metrics_path)
    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    started = time.perf_counter()
    session_examples = 0
    previous_batch_finished = started
    while step < args.steps and not stop:
        train_loader = DataLoader(
            train, batch_sampler=sampler, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None)
        for moon, fusion in train_loader:
            batch_started = time.perf_counter()
            moon = moon.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            streams = split_cached_features(moon, fusion)
            keep = teacher_keep_mask(moon.shape[0], len(STREAM_WIDTHS),
                                     args.teacher_dropout, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent, predictions = model(streams, keep)
                loss, metrics = compressor_loss(
                    latent, predictions, streams,
                    relational_weight=args.relational_weight,
                    variance_weight=args.variance_weight,
                    covariance_weight=args.covariance_weight,
                    diversity_weight=args.diversity_weight)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sampler.consumed(moon.shape[0])
            session_examples += moon.shape[0]
            step += 1
            batch_finished = time.perf_counter()
            if step % args.log_every == 0 or step == 1:
                elapsed = batch_finished - started
                row = {"kind": "train", "step": step, "epoch": sampler.epoch,
                       "cursor": sampler.cursor,
                       "examples": sampler.epoch * len(train) + sampler.cursor,
                       "session_examples_per_s": session_examples / max(elapsed, 1e-6),
                       "step_s": batch_finished - previous_batch_finished,
                       "compute_s": batch_finished - batch_started,
                       "gradient_norm": float(gradient_norm),
                       **{name: float(value) for name, value in metrics.items()},
                       "time": time.time()}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                _write_json(args.out / "status.json", row)
                print(row, flush=True)
            previous_batch_finished = batch_finished
            if step % args.eval_every == 0:
                values = evaluate(model, eval_loader, args, device)
                row = {"kind": "eval", "step": step, **values, "time": time.time()}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(row, flush=True)
                if values["loss"] < best_eval:
                    best_eval = values["loss"]
                    payload = checkpoint_payload(model=model, optimizer=optimizer,
                        sampler=sampler, step=step, best_eval=best_eval,
                        args=args, config=config)
                    _durable_save(payload, args.out / "best.pt")
                    _write_json(args.out / "best.json",
                                {"step": step, "eval_loss": best_eval})
            if step % args.checkpoint_every == 0 or step >= args.steps or stop:
                payload = checkpoint_payload(model=model, optimizer=optimizer,
                    sampler=sampler, step=step, best_eval=best_eval,
                    args=args, config=config)
                _durable_save(payload, last)
                print({"kind": "checkpoint", "step": step, "path": str(last)},
                      flush=True)
            if step >= args.steps or stop:
                break
        # DataLoader reached the end of the epoch. consumed() has already
        # advanced epoch/cursor exactly when the final short batch was used.
    if not last.is_file() or torch.load(last, map_location="cpu", weights_only=False)["step"] != step:
        _durable_save(checkpoint_payload(model=model, optimizer=optimizer,
            sampler=sampler, step=step, best_eval=best_eval, args=args,
            config=config), last)
    print({"kind": "compressor_exit", "step": step, "stopped": stop}, flush=True)


if __name__ == "__main__":
    main()
