#!/usr/bin/env python3
"""Convert the local one-terminal-expert MageFlow into a factored loop core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rwkv_lab.mage_flow_adaptation import (
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
)
from rwkv_lab.mage_flow_terminal_experts import (
    install_terminal_expert,
    load_terminal_expert,
    load_terminal_shared_backbone,
)
from rwkv_lab.mage_flow_tread_looping import (
    TreadLoopConfig,
    install_tread_factored_looping,
    save_tread_loop_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("photo", "animation"), required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--shared-backbone", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/mageflow_tread_factored_looping.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-id", default=MAGE_FLOW_BASE_ID)
    parser.add_argument("--revision", default=MAGE_FLOW_BASE_REVISION)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/workspace/models/Mage-Flow-Base"
        ),
        help="Local mirrored Mage-Flow weights; no separate source checkout is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("conversion requires a BF16 CUDA device")
    from huggingface_hub import snapshot_download
    from mage_flow import MageFlowPipeline

    device = torch.device("cuda")
    model_dir = (
        str(args.model_path.expanduser().resolve())
        if args.model_path.is_dir()
        else snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_files_only=True,
        )
    )
    pipeline = MageFlowPipeline.from_pretrained(
        model_dir,
        device=str(device),
        attn_type="flash2",
    )
    transformer = pipeline.model.transformer
    terminal = install_terminal_expert(
        transformer,
        args.domain,
        device=device,
        dtype=next(transformer.parameters()).dtype,
    )
    config = TreadLoopConfig.from_dict(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    controller = install_tread_factored_looping(
        transformer,
        config,
    )
    load_terminal_expert(terminal, args.domain, args.expert)
    if args.shared_backbone:
        load_terminal_shared_backbone(transformer, args.shared_backbone)
    report = save_tread_loop_controller(controller, args.output)
    report.update(
        {
            "domain": args.domain,
            "terminal_expert": str(args.expert.expanduser().resolve()),
            "shared_backbone": (
                str(args.shared_backbone.expanduser().resolve())
                if args.shared_backbone
                else None
            ),
            "base_model": args.model_id,
            "base_revision": args.revision,
            "functional_equivalence": {
                "status": "exact_zero_gate_initialization",
                "reason": (
                    "all original backbone and expert blocks remain intact; "
                    "zero-initialized refinement gates make added passes exact "
                    "no-ops until learned"
                ),
            },
        }
    )
    report_path = (
        args.report
        if args.report is not None
        else args.output.with_suffix(args.output.suffix + ".json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
