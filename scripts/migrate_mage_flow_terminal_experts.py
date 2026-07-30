#!/usr/bin/env python3
"""Build the two terminal Mage-Flow experts from the residual beta checkpoints.

This is intentionally a CPU-capable migration.  It loads only the released
Mage-Flow transformer, never the VAE or Qwen text encoder, and materializes one
terminal expert at a time.  The resulting photo and animation checkpoints are
independent; inference loads exactly one beside the shared backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
MAGE_SOURCE = REPO_ROOT / ".external" / "Mage"
if str(MAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MAGE_SOURCE))

from mage_flow.models.mage_flow import MageFlow, MageFlowParams

from rwkv_lab.mage_flow_adaptation import (
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
)
from rwkv_lab.mage_flow_terminal_experts import (
    initialize_from_residual_expert,
    install_terminal_expert,
    reset_terminal_expert_from_backbone,
    save_terminal_expert,
    terminal_architecture_report,
)

EXPECTED_BASE_PARAMETERS = 4_115_745_408
EXPECTED_EXPERT_PARAMETERS = 1_019_493_888


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_released_transformer(
    *,
    cache_dir: Path,
    dtype: torch.dtype,
) -> tuple[MageFlow, Path]:
    repository_cache = "models--" + MAGE_FLOW_BASE_ID.replace("/", "--")
    candidates = (
        cache_dir / "hub" / repository_cache / "snapshots" / MAGE_FLOW_BASE_REVISION,
        cache_dir / repository_cache / "snapshots" / MAGE_FLOW_BASE_REVISION,
    )
    snapshot = next((path for path in candidates if path.is_dir()), None)
    if snapshot is None:
        snapshot = Path(
            snapshot_download(
                repo_id=MAGE_FLOW_BASE_ID,
                revision=MAGE_FLOW_BASE_REVISION,
                cache_dir=str(cache_dir / "hub"),
                local_files_only=True,
            )
        )
    transformer_dir = snapshot / "transformer"
    config = json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))
    params = MageFlowParams(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        context_in_dim=int(config["context_in_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_heads=int(config["num_heads"]),
        depth=int(config["depth"]),
        axes_dim=[int(value) for value in config["axes_dim"]],
        checkpoint=False,
        patch_size=int(config.get("patch_size", 1)),
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        transformer = MageFlow(params)
    finally:
        torch.set_default_dtype(previous_dtype)
    weights = transformer_dir / "diffusion_pytorch_model.safetensors"
    missing, unexpected = load_model(
        transformer,
        weights,
        strict=True,
        device="cpu",
    )
    if missing or unexpected:
        raise RuntimeError(
            f"released transformer load mismatch: missing={missing} "
            f"unexpected={unexpected}"
        )
    return transformer, weights


def migrate(args: argparse.Namespace) -> dict:
    residual_dir = args.residual_checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(args.dtype)
    transformer, released_weights = _load_released_transformer(
        cache_dir=args.cache_dir.expanduser().resolve(),
        dtype=dtype,
    )
    controller = install_terminal_expert(
        transformer,
        EXPERT_DOMAINS[0],
        depth=3,
        dtype=dtype,
        device="cpu",
    )
    architecture = terminal_architecture_report(controller)
    if architecture["base_parameter_count"] != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            "pinned Mage-Flow base parameter count changed: "
            f"{architecture['base_parameter_count']} != {EXPECTED_BASE_PARAMETERS}"
        )
    if architecture["expert_parameter_count"] != EXPECTED_EXPERT_PARAMETERS:
        raise RuntimeError(
            "three-block terminal expert parameter count changed: "
            f"{architecture['expert_parameter_count']} "
            f"!= {EXPECTED_EXPERT_PARAMETERS}"
        )
    if not architecture["passed"]:
        raise RuntimeError(f"terminal architecture preflight failed: {architecture}")

    migrations = {}
    experts = {}
    for domain in EXPERT_DOMAINS:
        reset_terminal_expert_from_backbone(controller, domain)
        residual_path = residual_dir / f"mageflow-{domain}-expert.safetensors"
        if not residual_path.is_file():
            raise FileNotFoundError(residual_path)
        migrations[domain] = initialize_from_residual_expert(
            controller,
            residual_path,
        )
        destination = output_dir / f"mageflow-{domain}-terminal-expert.safetensors"
        experts[domain] = save_terminal_expert(
            controller,
            destination,
            dtype=dtype,
        )

    receipt = {
        "schema": "rwkv-lab.mage-flow-terminal-migration-receipt.v1",
        "base_model": MAGE_FLOW_BASE_ID,
        "base_revision": MAGE_FLOW_BASE_REVISION,
        "released_transformer": str(released_weights),
        "residual_checkpoint_dir": str(residual_dir),
        "architecture": architecture,
        "initialization": {
            "full_blocks": "released Mage-Flow blocks 9-11",
            "image_mlps": (
                "top-importance neurons from trained residual blocks 9-11, "
                "with learned scale folded into the output projection"
            ),
            "exact_function_preservation": False,
        },
        "migrations": migrations,
        "experts": experts,
    }
    receipt_path = output_dir / "migration_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".hf_cache",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def main() -> None:
    receipt = migrate(parse_args())
    print(
        json.dumps(
            {
                "architecture": receipt["architecture"],
                "experts": {
                    domain: item["path"]
                    for domain, item in receipt["experts"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
