"""Adapter-first recursive post-training with immutable confirmed parents.

The bounded proposer/solver lineage follows Absolute Zero
(https://arxiv.org/abs/2505.03335) and Self-Rewarding Language Models
(https://arxiv.org/abs/2401.10020), but with a stricter mechanism: Adamaton proposes train-only
data, each proposal trains an isolated LoRA/QLoRA adapter, and only an independently confirmed
promotion receipt can materialize a new parent. Rejected adapters and receipts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import torch

from rwkv_lab.adapters import load_adapter, unload_adapter
from rwkv_lab.posttrain_data import PostTrainingExample, SCHEMA as DATA_SCHEMA


REQUEST_SCHEMA = "rwkv-lab.adapter-proposal-request.v1"
RESPONSE_SCHEMA = "rwkv-lab.adapter-proposal-response.v1"
LOOP_SCHEMA = "rwkv-lab.adapter-recursive-loop.v1"
KINDS = {"sft": "sft", "dpo": "preference", "kto": "feedback", "orpo": "preference",
         "simpo": "preference", "reward": "preference", "prm": "prm"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def request_proposal(command: list[str], request: dict[str, Any], *, timeout: float,
                     maximum_bytes: int) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(command, input=json.dumps(request, sort_keys=True), text=True,
                               capture_output=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError(f"proposal command exited {completed.returncode}: {completed.stderr[-2000:]}")
    if len(completed.stdout.encode()) > maximum_bytes:
        raise ValueError("proposal response exceeds maximum bytes")
    return json.loads(completed.stdout), hashlib.sha256(completed.stdout.encode()).hexdigest()


def validate_proposal(value: dict[str, Any], args) -> dict[str, Any]:
    if value.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("invalid adapter proposal response schema")
    objective = str(value.get("objective") or "")
    if objective not in KINDS or objective not in set(args.allowed_objectives.split(",")):
        raise ValueError("proposal objective is not allowlisted")
    raw = value.get("train_examples")
    if not isinstance(raw, list) or not 1 <= len(raw) <= args.max_examples:
        raise ValueError("proposal contains an invalid number of examples")
    examples = [PostTrainingExample.from_dict(item, line=index + 1) for index, item in enumerate(raw)]
    if any(row.split != "train" or row.kind != KINDS[objective] for row in examples):
        raise ValueError("proposal may contain only train records matching its objective")
    if len({row.id for row in examples}) != len(examples):
        raise ValueError("proposal example ids must be unique")
    controls = dict(value.get("training") or {})
    unknown = set(controls) - {"rank", "learning_rate", "token_budget"}
    if unknown:
        raise ValueError(f"unsupported proposal training controls: {sorted(unknown)}")
    rank = int(controls.get("rank", args.rank))
    learning_rate = float(controls.get("learning_rate", args.learning_rate))
    token_budget = int(controls.get("token_budget", args.token_budget))
    if not 1 <= rank <= args.max_rank or not 0 < learning_rate <= args.max_learning_rate:
        raise ValueError("proposal rank or learning rate exceeds operator caps")
    if not 1 <= token_budget <= args.token_budget:
        raise ValueError("proposal token budget exceeds the round cap")
    return {"proposal_id": str(value.get("proposal_id") or "unnamed"),
            "rationale": str(value.get("rationale") or "")[:2000], "objective": objective,
            "examples": examples, "training": {"rank": rank, "learning_rate": learning_rate,
                                                  "token_budget": token_budget}}


def _example_dict(row: PostTrainingExample) -> dict[str, Any]:
    value: dict[str, Any] = {"schema": DATA_SCHEMA, "id": row.id, "kind": row.kind,
                             "split": "train", "metadata": row.metadata}
    if row.messages:
        value["messages"] = [message.__dict__ for message in row.messages]
    for name in ("text", "chosen", "rejected", "response"):
        if getattr(row, name):
            value[name] = getattr(row, name)
    if row.label is not None:
        value["label"] = row.label
    if row.steps:
        value["steps"] = [step.__dict__ for step in row.steps]
    if row.adversarial_steps:
        value["adversarial_steps"] = [step.__dict__ for step in row.adversarial_steps]
    return value


def materialize_parent(parent: str, adapter: str, output: Path, *, iteration: int,
                       receipt: str, device: str = "cpu") -> dict[str, Any]:
    from rwkv_lab.generate import build_from_ckpt

    model, blob = build_from_ckpt(parent, device=device)
    if device == "cpu":
        model = model.float()
    raw_manifest = json.loads((Path(adapter) / "adapter.json").read_text())
    if raw_manifest.get("quantized_frozen_base"):
        from rwkv_lab.quantization import quantize_model_nf4
        metadata = raw_manifest.get("metadata") or {}
        block_size = int(metadata.get("quant_block_size", 64))
        backend = str(metadata.get("quant_backend") or "portable")
        quantize_model_nf4(model, block_size=block_size, exclude=("head", "emb"), backend=backend)
    manifest = load_adapter(model, adapter, name="candidate", verify_base=True)
    replaced = unload_adapter(model, "candidate", merge=True)
    from rwkv_lab.quantization import dequantize_model_nf4
    dequantized = dequantize_model_nf4(model)
    child = dict(blob)
    child["model"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    child.pop("optimizer", None)
    child.pop("ema", None)
    child["lineage"] = {"schema": "rwkv-lab.adapter-parent-lineage.v1", "iteration": iteration,
                        "parent": str(Path(parent).resolve()), "parent_sha256": _sha256(parent),
                        "adapter": str(Path(adapter).resolve()),
                        "adapter_sha256": manifest["weights_sha256"],
                        "promotion_receipt": str(Path(receipt).resolve()),
                        "materialized_modules": replaced, "dequantized_modules": dequantized}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(child, temporary)
    os.replace(temporary, output)
    return {**child["lineage"], "checkpoint": str(output.resolve()), "sha256": _sha256(output)}


def run_loop(args) -> dict[str, Any]:
    if not args.proposal_command:
        raise ValueError("--proposal-command is required; Adamaton remains the proposal boundary")
    if not Path(args.checkpoint).is_file() or not Path(args.eval_data).is_file():
        raise ValueError("initial checkpoint and held-out eval data must exist")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "adapter-loop.json"
    configuration = {key: value for key, value in vars(args).items()
                     if key not in {"output", "proposal_command", "resume"}}
    configuration["proposal_command_sha256"] = hashlib.sha256(args.proposal_command.encode()).hexdigest()
    if state_path.exists():
        if not args.resume:
            raise ValueError("adapter loop exists; pass --resume or choose another output")
        state = json.loads(state_path.read_text())
        if state["configuration"] != configuration:
            raise ValueError("saved adapter-loop configuration differs")
    else:
        state = {"schema": LOOP_SCHEMA, "status": "running", "configuration": configuration,
                 "initial_checkpoint": str(Path(args.checkpoint).resolve()),
                 "current_checkpoint": str(Path(args.checkpoint).resolve()),
                 "iterations": [], "created_ts": time.time()}
        _atomic_json(state_path, state)
    command = shlex.split(args.proposal_command)
    for iteration in range(len(state["iterations"]), args.rounds):
        directory = root / f"iteration-{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        request = {"schema": REQUEST_SCHEMA, "iteration": iteration,
                   "parent_sha256": _sha256(state["current_checkpoint"]),
                   "history": [{"iteration": row["iteration"], "accepted": row["accepted"],
                                "failed_gates": row.get("failed_gates", [])}
                               for row in state["iterations"]],
                   "constraints": {"maximum_examples": args.max_examples,
                                   "allowed_objectives": args.allowed_objectives.split(","),
                                   "maximum_rank": args.max_rank,
                                   "maximum_token_budget": args.token_budget}}
        raw, response_hash = request_proposal(command, request, timeout=args.proposal_timeout,
                                              maximum_bytes=args.max_proposal_bytes)
        proposal = validate_proposal(raw, args)
        data_path = directory / "proposal-train.jsonl"
        data_path.write_text("\n".join(json.dumps(_example_dict(row), sort_keys=True, default=list)
                                       for row in proposal["examples"]) + "\n")
        receipt = {"schema": RESPONSE_SCHEMA, "proposal_id": proposal["proposal_id"],
                   "rationale": proposal["rationale"], "response_sha256": response_hash,
                   "data_sha256": _sha256(data_path), "objective": proposal["objective"],
                   "training": proposal["training"]}
        _atomic_json(directory / "proposal-receipt.json", receipt)
        campaign_dir = directory / "campaign"
        train = proposal["training"]
        campaign_command = [sys.executable, "-m", "rwkv_lab.posttrain_campaign",
                            "--checkpoint", state["current_checkpoint"], "--data", str(data_path),
                            "--eval-data", args.eval_data, "--output", str(campaign_dir),
                            "--objectives", proposal["objective"], "--seeds", args.seeds,
                            "--confirm-seeds", args.confirm_seeds,
                            "--token-budget", str(train["token_budget"]),
                            "--rank", str(train["rank"]), "--learning-rate", str(train["learning_rate"]),
                            "--batch-size", str(args.batch_size), "--max-length", str(args.max_length),
                            "--minimum-delta", str(args.minimum_delta),
                            "--maximum-family-regression", str(args.maximum_family_regression),
                            "--device", args.device, "--base-quantization", args.base_quantization,
                            "--quant-backend", args.quant_backend, "--packing",
                            (args.packing if len(proposal["examples"]) > 1 else "audit")]
        with (directory / "campaign.log").open("w", buffering=1) as log:
            completed = subprocess.run(campaign_command, cwd=Path.cwd(),
                                       env={**os.environ, "PYTHONPATH": "src"},
                                       stdout=log, stderr=subprocess.STDOUT, check=False,
                                       timeout=args.round_timeout)
        promotion_path = campaign_dir / f"promotion-{proposal['objective']}.json"
        promotion = json.loads(promotion_path.read_text()) if promotion_path.is_file() else {}
        accepted = completed.returncode == 0 and bool(promotion.get("eligible"))
        lineage = None
        if accepted:
            parent_path = directory / "promoted-parent.pt"
            lineage = materialize_parent(state["current_checkpoint"], promotion["selected_adapter"],
                                         parent_path, iteration=iteration,
                                         receipt=str(promotion_path), device=args.materialize_device)
            state["current_checkpoint"] = str(parent_path.resolve())
        failed_gates = sorted({name for run in promotion.get("confirmation_runs") or []
                               for name, passed in (run.get("promotion") or {}).get("gates", {}).items()
                               if not passed})
        state["iterations"].append({"iteration": iteration, "proposal": receipt,
                                    "accepted": accepted, "failed_gates": failed_gates,
                                    "campaign": str(campaign_dir / "posttrain-campaign.json"),
                                    "promotion_receipt": str(promotion_path), "lineage": lineage,
                                    "preserved_adapter": promotion.get("selected_adapter", "")})
        _atomic_json(state_path, state)
    state["status"] = "complete"
    state["completed_ts"] = time.time()
    _atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapter-first recursive post-training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposal-command", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--allowed-objectives", default="sft,dpo,kto,orpo,simpo,prm")
    parser.add_argument("--max-examples", type=int, default=4096)
    parser.add_argument("--max-proposal-bytes", type=int, default=8 << 20)
    parser.add_argument("--proposal-timeout", type=float, default=60.0)
    parser.add_argument("--round-timeout", type=float, default=86400.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-learning-rate", type=float, default=1e-3)
    parser.add_argument("--token-budget", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--maximum-family-regression", type=float, default=0.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--confirm-seeds", default="100,101,102")
    parser.add_argument("--base-quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--quant-backend", choices=["auto", "portable", "torchao"], default="auto")
    parser.add_argument("--packing", choices=["off", "audit", "reset"], default="reset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--materialize-device", default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    print(json.dumps(run_loop(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
