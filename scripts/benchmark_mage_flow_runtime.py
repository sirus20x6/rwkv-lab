#!/usr/bin/env python3
"""Benchmark full Mage-Flow terminal-expert training runtime on a fixed image.

Each invocation runs one isolated variant so compiled/quantized module
mutations and allocator state cannot leak between measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MAGE_SOURCE = REPO_ROOT / ".external" / "Mage"
for source in (REPO_ROOT / "src", MAGE_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from huggingface_hub import snapshot_download
from mage_flow import MageFlowPipeline

from rwkv_lab.mage_flow_adaptation import (
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
    load_domain_manifest,
)
from rwkv_lab.mage_flow_expert_train import (
    _forward_transformer,
    encode_domain_batch,
)
from rwkv_lab.mage_flow_optimizations import (
    FP32MasterAdamW,
    FrozenEncoderCache,
    compile_transformer_blocks,
    configure_activation_checkpointing,
    convert_trainable_image_ffns_to_float8,
)
from rwkv_lab.mage_flow_pretrain import _load_image_tensor, rectified_flow_loss
from rwkv_lab.mage_flow_terminal_experts import (
    configure_terminal_training_scope,
    install_terminal_expert,
    load_terminal_expert,
    load_terminal_shared_backbone,
    terminal_optimizer_parameter_groups,
)
from rwkv_lab.mage_flow_terminal_train import TerminalExpertTrainConfig


VARIANTS = {
    "baseline": {
        "attention": "flash2",
        "checkpoint": "full",
        "compile": False,
        "float8": False,
        "cache": False,
    },
    "fa4": {
        "attention": "flash4",
        "checkpoint": "full",
        "compile": False,
        "float8": False,
        "cache": False,
    },
    "fa4_trainable_checkpoint": {
        "attention": "flash4",
        "checkpoint": "trainable",
        "compile": False,
        "float8": False,
        "cache": False,
    },
    "fa4_no_checkpoint": {
        "attention": "flash4",
        "checkpoint": "none",
        "compile": False,
        "float8": False,
        "cache": False,
    },
    "fa4_compile": {
        "attention": "flash4",
        "checkpoint": "trainable",
        "compile": True,
        "float8": False,
        "cache": False,
    },
    "fa4_compile_fp8": {
        "attention": "flash4",
        "checkpoint": "trainable",
        "compile": True,
        "float8": True,
        "cache": False,
    },
    "fa4_compile_fp8_cache": {
        "attention": "flash4",
        "checkpoint": "trainable",
        "compile": True,
        "float8": True,
        "cache": True,
    },
    "fa2_cache": {
        "attention": "flash2",
        "checkpoint": "full",
        "compile": False,
        "float8": False,
        "cache": True,
    },
    "fa2_no_checkpoint": {
        "attention": "flash2",
        "checkpoint": "none",
        "compile": False,
        "float8": False,
        "cache": False,
    },
    "fa2_no_checkpoint_cache": {
        "attention": "flash2",
        "checkpoint": "none",
        "compile": False,
        "float8": False,
        "cache": True,
    },
}


def _fixed_row(manifest: Path, domain: str) -> dict:
    rows = [
        row
        for row in load_domain_manifest(manifest)
        if row["domain"] == domain
    ]
    if not rows:
        raise ValueError(f"no {domain} rows in {manifest}")
    # Deterministic worst common rung: largest token count, then image ID.
    return max(
        rows,
        key=lambda row: (
            int(row["latent_tokens"]),
            str(row["image_id"]),
        ),
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def benchmark(args: argparse.Namespace) -> dict:
    settings = VARIANTS[args.variant]
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    row = _fixed_row(args.train_manifest, args.domain)
    model_dir = (
        str(args.model_dir.expanduser().resolve())
        if args.model_dir is not None
        else snapshot_download(
            repo_id=MAGE_FLOW_BASE_ID,
            revision=MAGE_FLOW_BASE_REVISION,
            local_files_only=True,
        )
    )
    load_started = time.perf_counter()
    pipeline = MageFlowPipeline.from_pretrained(
        model_dir,
        device=str(device),
        attn_type=settings["attention"],
    )
    model = pipeline.model
    model.vae.sample_posterior = True
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    controller = install_terminal_expert(
        transformer,
        args.domain,
        dtype=next(transformer.parameters()).dtype,
        device=device,
    )
    load_terminal_expert(controller, args.domain, args.expert_checkpoint)
    if args.shared_backbone_checkpoint:
        load_terminal_shared_backbone(
            transformer,
            args.shared_backbone_checkpoint,
        )
    scope = configure_terminal_training_scope(
        transformer,
        controller,
        train_backbone_final_fraction=1 / 3,
    )
    float8_report = convert_trainable_image_ffns_to_float8(
        transformer,
        enabled=settings["float8"],
        recipe="tensorwise",
    )
    if settings["float8"]:
        scope = configure_terminal_training_scope(
            transformer,
            controller,
            train_backbone_final_fraction=1 / 3,
        )
    checkpoint_report = configure_activation_checkpointing(
        transformer,
        settings["checkpoint"],
    )
    compile_report = compile_transformer_blocks(
        transformer,
        enabled=settings["compile"],
        mode="default",
        dynamic=False,
    )
    groups = terminal_optimizer_parameter_groups(
        transformer,
        controller,
        # Exercise the complete AdamW path while keeping independent variant
        # runs numerically anchored to the same checkpoint.
        expert_learning_rate=1e-12,
        backbone_learning_rate_multiplier=0.5,
    )
    optimizer = FP32MasterAdamW(
        groups,
        lr=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=True,
    )
    trainable = [
        parameter
        for parameter in transformer.parameters()
        if parameter.requires_grad
    ]
    config = TerminalExpertTrainConfig(
        domain=args.domain,
        train_manifest=str(args.train_manifest),
        expert_checkpoint=str(args.expert_checkpoint),
        shared_backbone_checkpoint=(
            str(args.shared_backbone_checkpoint)
            if args.shared_backbone_checkpoint
            else None
        ),
        output_dir=str(args.output.parent),
        microbatch_size=1,
        gradient_accumulation_steps=args.accumulation,
        caption_dropout=0.0,
        attention_backend=settings["attention"],
        activation_checkpointing_mode=settings["checkpoint"],
        compile_transformer_blocks=settings["compile"],
        float8_training=settings["float8"],
    )
    image = _load_image_tensor(row).pin_memory()
    if settings["cache"]:
        cache_dir = args.output.parent / f"{args.variant}-encoder-cache"
        cache = FrozenEncoderCache(
            cache_dir,
            mode="read_write",
            model_id=MAGE_FLOW_BASE_ID,
            model_revision=MAGE_FLOW_BASE_REVISION,
        )
        model._training_encoder_cache = cache
        # Populate both entries once, outside all measured windows.
        encode_domain_batch(
            model,
            [row],
            [image],
            config,
            device,
            caption_dropout=0.0,
        )
        cache.mode = "read_only"
        model.txt_enc.to("cpu")
        model.vae.to("cpu")
        source_images = [None]
    else:
        source_images = [image]

    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    load_seconds = time.perf_counter() - load_started

    def update() -> tuple[float, float]:
        started = time.perf_counter()
        last_loss = 0.0
        for _ in range(args.accumulation):
            flow = encode_domain_batch(
                model,
                [row],
                source_images,
                config,
                device,
                caption_dropout=0.0,
            )
            with (
                controller.route(args.domain),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                prediction = _forward_transformer(transformer, flow)
                loss, observed = rectified_flow_loss(
                    prediction,
                    flow["velocity"],
                )
                scaled = loss / args.accumulation
            scaled.backward()
            last_loss = float(observed.item())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        return time.perf_counter() - started, last_loss

    warmup_seconds = []
    for _ in range(args.warmup_updates):
        elapsed, _loss = update()
        warmup_seconds.append(elapsed)
    torch.cuda.reset_peak_memory_stats(device)
    measured = []
    losses = []
    for _ in range(args.measure_updates):
        elapsed, loss = update()
        measured.append(elapsed)
        losses.append(loss)
    samples = args.accumulation * len(measured)
    total_seconds = sum(measured)
    result = {
        "schema": "rwkv-lab.mage-flow-runtime-benchmark.v1",
        "variant": args.variant,
        "settings": settings,
        "domain": args.domain,
        "image_id": str(row["image_id"]),
        "geometry": {
            "width": int(row["train_width"]),
            "height": int(row["train_height"]),
            "latent_tokens": int(row["latent_tokens"]),
        },
        "accumulation": args.accumulation,
        "warmup_updates": args.warmup_updates,
        "measure_updates": args.measure_updates,
        "load_and_first_setup_seconds": load_seconds,
        "warmup_update_seconds": warmup_seconds,
        "measured_update_seconds": measured,
        "median_update_seconds": statistics.median(measured),
        "images_per_second": samples / total_seconds,
        "median_images_per_second": (
            args.accumulation / statistics.median(measured)
        ),
        "last_loss": losses[-1],
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "attention": settings["attention"],
        "activation_checkpointing": checkpoint_report,
        "regional_compile": compile_report,
        "float8": float8_report,
        "optimizer_precision": optimizer.precision_report(),
        "training_scope": scope,
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Local Mage-Flow-Base snapshot; otherwise use the HF cache.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--shared-backbone-checkpoint", type=Path)
    parser.add_argument("--domain", choices=("photo", "animation"), default="photo")
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measure-updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = benchmark(args)
    except torch.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        failure = {
            "schema": "rwkv-lab.mage-flow-runtime-benchmark.v1",
            "variant": args.variant,
            "error": "cuda_out_of_memory",
            "message": str(error),
        }
        _atomic_json(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
