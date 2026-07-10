"""Bounded propose → train → independently evaluate → promote loop.

The architecture follows Absolute Zero's proposer/solver curriculum loop
(https://arxiv.org/abs/2505.03335) and the iterative lineage idea explored by
Self-Rewarding Language Models (https://arxiv.org/abs/2401.10020), with a
deliberately stricter trust boundary: Adamaton proposes training tasks, but
never sees the held-out task file or private verifier results, and it cannot
promote a checkpoint. ``rlvr_train`` owns independent statistical, regression,
contamination, and budget gates. Generated code is executed only by the trusted
external verifier command, never by this controller or the trainer.
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

from rwkv_lab.rlvr_train import RLVRTask, TASK_SCHEMA


PROPOSAL_REQUEST_SCHEMA = "rwkv-lab.recursive-proposal-request.v1"
PROPOSAL_RESPONSE_SCHEMA = "rwkv-lab.recursive-proposal-response.v1"
LOOP_SCHEMA = "rwkv-lab.recursive-loop.v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_int(
    value: Any, default: int, maximum: int, name: str, *, minimum: int = 0
) -> int:
    parsed = int(default if value is None else value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"proposal {name}={parsed} outside [{minimum},{maximum}]")
    return parsed


def _bounded_float(
    value: Any, default: float, maximum: float, name: str, *, minimum: float = 0.0
) -> float:
    parsed = float(default if value is None else value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"proposal {name}={parsed} outside [{minimum},{maximum}]")
    return parsed


def validate_proposal(value: dict[str, Any], args) -> dict[str, Any]:
    """Validate an Adamaton response against the operator's immutable caps."""

    if value.get("schema") != PROPOSAL_RESPONSE_SCHEMA:
        raise ValueError("proposal returned an invalid schema")
    raw_tasks = value.get("train_tasks")
    if (
        not isinstance(raw_tasks, list)
        or not 1 <= len(raw_tasks) <= args.max_proposal_tasks
    ):
        raise ValueError(
            f"proposal must contain 1–{args.max_proposal_tasks} training tasks"
        )
    tasks = [
        RLVRTask.from_dict(item, line=index + 1) for index, item in enumerate(raw_tasks)
    ]
    if any(task.split != "train" for task in tasks):
        raise ValueError("proposal may contain training tasks only")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("proposal task ids must be unique")
    allowed_keys = {
        "algorithm",
        "steps",
        "prompts_per_step",
        "group_size",
        "max_new",
        "lr",
        "sft_steps",
        "preflight_prompts",
    }
    training = dict(value.get("training") or {})
    unknown = set(training) - allowed_keys
    if unknown:
        raise ValueError(
            f"proposal contains unsupported training controls: {sorted(unknown)}"
        )
    algorithm = str(training.get("algorithm") or args.algorithm)
    if algorithm not in ("gspo", "dr_grpo", "dapo"):
        raise ValueError("proposal algorithm is not allowlisted")
    bounded = {
        "algorithm": algorithm,
        "steps": _bounded_int(
            training.get("steps"), args.steps, args.steps, "steps", minimum=1
        ),
        "prompts_per_step": _bounded_int(
            training.get("prompts_per_step"),
            args.prompts_per_step,
            args.prompts_per_step,
            "prompts_per_step",
            minimum=1,
        ),
        "group_size": _bounded_int(
            training.get("group_size"),
            args.group_size,
            args.group_size,
            "group_size",
            minimum=2,
        ),
        "max_new": _bounded_int(
            training.get("max_new"), args.max_new, args.max_new, "max_new", minimum=1
        ),
        "lr": _bounded_float(training.get("lr"), args.lr, args.max_lr, "lr"),
        "sft_steps": _bounded_int(
            training.get("sft_steps"), args.sft_steps, args.sft_steps, "sft_steps"
        ),
        "preflight_prompts": _bounded_int(
            training.get("preflight_prompts"),
            args.preflight_prompts,
            args.preflight_prompts,
            "preflight_prompts",
        ),
    }
    return {
        "proposal_id": str(value.get("proposal_id") or "unnamed"),
        "rationale": str(value.get("rationale") or "")[:2000],
        "tasks": tasks,
        "training": bounded,
    }


def request_proposal(
    command: list[str], request: dict[str, Any], *, timeout: float, maximum_bytes: int
) -> tuple[dict[str, Any], str]:
    payload = json.dumps(request, sort_keys=True)
    proc = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(
            f"proposal command exited {proc.returncode}: {proc.stderr[-2000:]}"
        )
    if len(proc.stdout.encode()) > maximum_bytes:
        raise ValueError("proposal response exceeds the configured byte limit")
    return json.loads(proc.stdout), hashlib.sha256(proc.stdout.encode()).hexdigest()


def _write_tasks(path: Path, tasks: list[RLVRTask]) -> None:
    lines = []
    for task in tasks:
        lines.append(
            json.dumps(
                {
                    "schema": TASK_SCHEMA,
                    "id": task.id,
                    "split": task.split,
                    "prompt": task.prompt,
                    "verifier": task.verifier,
                    "metadata": task.metadata or {},
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_train_command(
    args,
    proposal: dict[str, Any],
    *,
    parent: str,
    out: Path,
    task_path: Path,
    seed: int,
    rollout_budget: int,
) -> list[str]:
    cfg = proposal["training"]
    command = [
        sys.executable,
        "-m",
        "rwkv_lab.rlvr_train",
        "--ckpt",
        parent,
        "--out",
        str(out),
        "--tasks",
        str(task_path),
        "--heldout-tasks",
        args.heldout_tasks,
        "--algorithm",
        cfg["algorithm"],
        "--steps",
        str(cfg["steps"]),
        "--prompts-per-step",
        str(cfg["prompts_per_step"]),
        "--group-size",
        str(cfg["group_size"]),
        "--max-new",
        str(cfg["max_new"]),
        "--lr",
        str(cfg["lr"]),
        "--reference",
        args.reference,
        "--sft-steps",
        str(cfg["sft_steps"]),
        "--sft-batch-size",
        str(args.sft_batch_size),
        "--sft-lr",
        str(args.sft_lr),
        "--preflight-prompts",
        str(cfg["preflight_prompts"]),
        "--min-preflight-reward",
        str(args.min_preflight_reward),
        "--max-preflight-reward",
        str(args.max_preflight_reward),
        "--min-preflight-active-groups",
        str(args.min_preflight_active_groups),
        "--eval-prompts",
        str(args.eval_prompts),
        "--eval-group-size",
        str(args.eval_group_size),
        "--eval-every",
        str(cfg["steps"]),
        "--min-heldout-delta",
        str(args.min_heldout_delta),
        "--confidence",
        str(args.confidence),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--max-family-regression",
        str(args.max_family_regression),
        "--max-rollout-tokens",
        str(rollout_budget),
        "--max-train-seconds",
        str(args.max_round_seconds),
        "--rollout-engine",
        args.rollout_engine,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--save-every",
        str(cfg["steps"]),
        "--log-samples",
        str(args.log_samples),
    ]
    command.append(
        "--require-confidence" if args.require_confidence else "--no-require-confidence"
    )
    if args.verifier_command:
        command += [
            "--verifier-command",
            args.verifier_command,
            "--verifier-timeout",
            str(args.verifier_timeout),
        ]
    return command


def run_loop(args) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.rounds < 1 or args.max_consecutive_rejections < 1:
        raise ValueError("rounds and rejection limit must be positive")
    if args.max_total_rollout_tokens < 1 or args.max_round_seconds <= 0:
        raise ValueError("recursive rollout and wall-clock budgets must be positive")
    if (
        args.max_proposal_tasks < 1
        or args.max_proposal_bytes < 1
        or args.max_lr < args.lr
    ):
        raise ValueError(
            "proposal caps must be positive and max-lr must cover the default lr"
        )
    if not Path(args.ckpt).is_file() or not Path(args.heldout_tasks).is_file():
        raise ValueError("initial checkpoint and held-out task file must exist")
    if not args.proposal_command:
        raise ValueError(
            "--proposal-command is required; proposal stays an Adamaton boundary"
        )
    proposal_command = shlex.split(args.proposal_command)
    configuration = {
        key: value for key, value in vars(args).items() if key not in {"out", "resume"}
    }
    for key in ("proposal_command", "verifier_command"):
        configuration[key] = (
            hashlib.sha256(str(configuration[key]).encode()).hexdigest()
            if configuration.get(key)
            else ""
        )
    state_path = out / "loop.json"
    if state_path.exists():
        if not args.resume:
            raise ValueError(
                "recursive loop already exists; pass --resume or choose a new --out"
            )
        state = json.loads(state_path.read_text())
        if state.get("configuration") != configuration:
            raise ValueError("recursive-loop configuration differs from the saved run")
    else:
        state = {
            "schema": LOOP_SCHEMA,
            "status": "running",
            "configuration": configuration,
            "initial_checkpoint": str(Path(args.ckpt).resolve()),
            "initial_sha256": _sha256(args.ckpt),
            "current_checkpoint": str(Path(args.ckpt).resolve()),
            "iterations": [],
            "total_rollout_tokens": 0,
            "created_ts": time.time(),
        }
        _atomic_json(state_path, state)

    parent = state["current_checkpoint"]
    rejections = 0
    for previous in reversed(state["iterations"]):
        if previous["accepted"]:
            break
        rejections += 1
    if rejections >= args.max_consecutive_rejections:
        state["status"] = "rejection_limit"
        _atomic_json(state_path, state)
        return state
    start_iteration = len(state["iterations"])
    for iteration in range(start_iteration, args.rounds):
        remaining = args.max_total_rollout_tokens - int(state["total_rollout_tokens"])
        if remaining <= 0:
            state["status"] = "rollout_budget_exhausted"
            break
        iteration_dir = out / f"iteration-{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        parent_hash = _sha256(parent)
        request = {
            "schema": PROPOSAL_REQUEST_SCHEMA,
            "iteration": iteration,
            "parent_sha256": parent_hash,
            "outcome_history": [
                {
                    "iteration": row["iteration"],
                    "accepted": row["accepted"],
                    "failed_gates": row.get("failed_gates", []),
                }
                for row in state["iterations"]
            ],
            "constraints": {
                "maximum_tasks": args.max_proposal_tasks,
                "algorithms": ["gspo", "dr_grpo", "dapo"],
                "maximum_steps": args.steps,
                "maximum_group_size": args.group_size,
                "maximum_response_tokens": args.max_new,
                "remaining_rollout_tokens": remaining,
            },
        }
        raw, response_hash = request_proposal(
            proposal_command,
            request,
            timeout=args.proposal_timeout,
            maximum_bytes=args.max_proposal_bytes,
        )
        proposal = validate_proposal(raw, args)
        task_path = iteration_dir / "train_tasks.jsonl"
        _write_tasks(task_path, proposal["tasks"])
        receipt = {
            "schema": PROPOSAL_RESPONSE_SCHEMA,
            "proposal_id": proposal["proposal_id"],
            "rationale": proposal["rationale"],
            "response_sha256": response_hash,
            "task_sha256": _sha256(task_path),
            "task_count": len(proposal["tasks"]),
            "training": proposal["training"],
        }
        _atomic_json(iteration_dir / "proposal_receipt.json", receipt)

        candidate_dir = iteration_dir / "candidate"
        command = build_train_command(
            args,
            proposal,
            parent=parent,
            out=candidate_dir,
            task_path=task_path,
            seed=args.seed + iteration * 1_000_003,
            rollout_budget=remaining,
        )
        with open(iteration_dir / "trainer.log", "w", buffering=1) as log:
            try:
                proc = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env={**os.environ, "PYTHONPATH": "src"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.max_round_seconds,
                    check=False,
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                log.write(f"trainer exceeded hard {args.max_round_seconds}s controller timeout\n")
                returncode = 124
        result_path = candidate_dir / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "status": "failed",
                "error": f"trainer exited {returncode} without result.json",
            }
        )
        accepted = bool(
            result.get("status") == "complete"
            and result.get("promotion", {}).get("eligible")
        )
        candidate = str(result.get("checkpoint") or "")
        if accepted and result.get("checkpoint_parent_sha256") != parent_hash:
            raise ValueError(
                "promotion candidate does not descend from the current parent hash"
            )
        if accepted and not Path(candidate).is_file():
            raise ValueError(
                "trainer marked a missing candidate checkpoint promotion-eligible"
            )
        if accepted:
            parent, rejections = candidate, 0
        else:
            rejections += 1
        failed_gates = [
            name
            for name, passed in result.get("promotion", {}).get("gates", {}).items()
            if not passed
        ]
        record = {
            "iteration": iteration,
            "proposal_id": proposal["proposal_id"],
            "parent_before": state["current_checkpoint"],
            "candidate": candidate,
            "accepted": accepted,
            "failed_gates": failed_gates,
            "training_status": result.get("training_status"),
            "heldout_delta": result.get("promotion", {}).get("heldout_delta"),
            "rollout_tokens": int(result.get("total_rollout_tokens", 0)),
            "result_path": str(result_path.resolve()),
            "returncode": returncode,
        }
        state["iterations"].append(record)
        state["total_rollout_tokens"] += record["rollout_tokens"]
        state["current_checkpoint"] = parent
        state["current_sha256"] = _sha256(parent)
        state["completed_rounds"] = len(state["iterations"])
        state["promotions"] = sum(row["accepted"] for row in state["iterations"])
        _atomic_json(state_path, state)
        if rejections >= args.max_consecutive_rejections:
            state["status"] = "rejection_limit"
            break
    else:
        state["status"] = "complete"
    state["completed_ts"] = time.time()
    state["promotions"] = sum(row["accepted"] for row in state["iterations"])
    _atomic_json(state_path, state)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bounded Adamaton→RLVR recursive improvement loop"
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-tasks", required=True)
    ap.add_argument("--proposal-command", required=True)
    ap.add_argument("--proposal-timeout", type=float, default=30.0)
    ap.add_argument("--max-proposal-bytes", type=int, default=8 << 20)
    ap.add_argument("--max-proposal-tasks", type=int, default=4096)
    ap.add_argument("--verifier-command", default="")
    ap.add_argument("--verifier-timeout", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-consecutive-rejections", type=int, default=2)
    ap.add_argument("--max-total-rollout-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-round-seconds", type=float, default=3600)
    ap.add_argument("--algorithm", choices=["gspo", "dr_grpo", "dapo"], default="gspo")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-lr", type=float, default=1e-5)
    ap.add_argument(
        "--reference", choices=["initial", "rollout", "none"], default="rollout"
    )
    ap.add_argument(
        "--rollout-engine", choices=["auto", "recurrent", "batched"], default="auto"
    )
    ap.add_argument("--sft-steps", type=int, default=16)
    ap.add_argument("--sft-batch-size", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--preflight-prompts", type=int, default=8)
    ap.add_argument("--min-preflight-reward", type=float, default=0.01)
    ap.add_argument("--max-preflight-reward", type=float, default=0.99)
    ap.add_argument("--min-preflight-active-groups", type=int, default=1)
    ap.add_argument("--eval-prompts", type=int, default=64)
    ap.add_argument("--eval-group-size", type=int, default=4)
    ap.add_argument("--min-heldout-delta", type=float, default=0.01)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument(
        "--require-confidence", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--max-family-regression", type=float, default=0.0)
    ap.add_argument("--log-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    result = run_loop(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
