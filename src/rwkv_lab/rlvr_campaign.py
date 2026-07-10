"""Run a reproducible multi-algorithm, multi-seed RLVR comparison.

The arms invoke :mod:`rwkv_lab.rlvr_train` in isolated subprocesses and
aggregate its versioned result contract. Algorithm sources are Dr.GRPO
(https://arxiv.org/abs/2503.20783), DAPO
(https://arxiv.org/abs/2503.14476), and GSPO
(https://arxiv.org/abs/2507.18071); adopted mechanisms are documented in
``rlvr_train.py`` and README.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


CAMPAIGN_SCHEMA = "rwkv-lab.rlvr-campaign.v1"


@dataclass(frozen=True)
class Arm:
    algorithm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.algorithm}/seed-{self.seed:04d}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), std


def _campaign_config(args) -> dict[str, Any]:
    """Canonical budget identity used to make resume/skip behavior safe."""

    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"out", "resume_existing"}
    }
    for key in ("ckpt", "tasks", "reference_ckpt"):
        if config.get(key):
            config[key] = str(Path(config[key]).resolve())
    return config


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_algorithm: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        if item.get("status") == "complete":
            by_algorithm.setdefault(str(item["algorithm"]), []).append(item)
    summary = {}
    for algorithm, rows in sorted(by_algorithm.items()):
        rewards = [float(r.get("heldout", {}).get("reward", 0)) for r in rows]
        baselines = [
            float(r.get("baseline_heldout", {}).get("reward", 0)) for r in rows
        ]
        deltas = [float(r.get("promotion", {}).get("heldout_delta", 0)) for r in rows]
        mean, std = _mean_std(rewards)
        delta_mean, delta_std = _mean_std(deltas)
        summary[algorithm] = {
            "runs": len(rows),
            "heldout_mean": mean,
            "heldout_std": std,
            "baseline_mean": _mean_std(baselines)[0],
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "promotions": sum(
                bool(r.get("promotion", {}).get("eligible")) for r in rows
            ),
            "updates_applied": sum(
                int(r.get("promotion", {}).get("updates_applied", 0)) for r in rows
            ),
        }
    return summary


def build_command(args, arm: Arm, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rwkv_lab.rlvr_train",
        "--ckpt",
        args.ckpt,
        "--out",
        str(run_dir),
        "--algorithm",
        arm.algorithm,
        "--seed",
        str(arm.seed),
        "--steps",
        str(args.steps),
        "--prompts-per-step",
        str(args.prompts_per_step),
        "--group-size",
        str(args.group_size),
        "--epochs",
        str(args.epochs),
        "--max-new",
        str(args.max_new),
        "--temperature",
        str(args.temperature),
        "--eval-temperature",
        str(args.eval_temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup",
        str(args.warmup),
        "--grad-clip",
        str(args.grad_clip),
        "--kl-coef",
        str(args.kl_coef),
        "--reference",
        args.reference,
        "--eval-every",
        str(args.eval_every),
        "--eval-prompts",
        str(args.eval_prompts),
        "--eval-group-size",
        str(args.eval_group_size),
        "--min-heldout-delta",
        str(args.min_heldout_delta),
        "--difficulty",
        str(args.difficulty),
        "--device",
        args.device,
        "--save-every",
        str(args.save_every),
        "--log-samples",
        str(args.log_samples),
    ]
    if args.tasks:
        command += ["--tasks", args.tasks]
    else:
        command += [
            "--train-tasks",
            str(args.train_tasks),
            "--eval-tasks",
            str(args.eval_tasks),
        ]
    if args.reference_ckpt:
        command += ["--reference-ckpt", args.reference_ckpt]
    if args.verifier_command:
        command += [
            "--verifier-command",
            args.verifier_command,
            "--verifier-timeout",
            str(args.verifier_timeout),
        ]
    return command


def run_campaign(args) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {args.ckpt}")
    if args.tasks and not Path(args.tasks).is_file():
        raise ValueError(f"task JSONL does not exist: {args.tasks}")
    algorithms = _csv(args.algorithms)
    invalid = set(algorithms) - {"gspo", "dr_grpo", "dapo"}
    if not algorithms or invalid:
        raise ValueError(f"invalid RLVR algorithms: {sorted(invalid)}")
    seeds = [int(x) for x in _csv(args.seeds)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(algorithms) != len(set(algorithms)) or len(seeds) != len(set(seeds)):
        raise ValueError("algorithms and seeds must not contain duplicates")
    arms = [Arm(algorithm, seed) for algorithm in algorithms for seed in seeds]
    started = time.time()
    configuration = _campaign_config(args)
    campaign_path = out / "campaign.json"
    if args.resume_existing and campaign_path.exists():
        existing = json.loads(campaign_path.read_text())
        if existing.get("configuration") != configuration:
            raise ValueError(
                "existing campaign configuration differs; choose a new --out directory"
            )
    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "running",
        "created_ts": started,
        "checkpoint": str(checkpoint.resolve()),
        "tasks": args.tasks or "generated:arithmetic",
        "algorithms": algorithms,
        "seeds": seeds,
        "steps": args.steps,
        "group_size": args.group_size,
        "prompts_per_step": args.prompts_per_step,
        "max_new": args.max_new,
        "configuration": configuration,
        "arms": [a.name for a in arms],
    }
    _atomic_json(campaign_path, manifest)
    results = []
    for index, arm in enumerate(arms, 1):
        run_dir = out / arm.algorithm / f"seed-{arm.seed:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"
        if args.resume_existing and result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("status") == "complete":
                results.append(
                    {
                        "algorithm": arm.algorithm,
                        "seed": arm.seed,
                        "run_dir": str(run_dir.resolve()),
                        **result,
                    }
                )
                manifest.update(
                    {
                        "results": results,
                        "summary": summarize(results),
                        "completed_arms": len(results),
                    }
                )
                _atomic_json(campaign_path, manifest)
                continue
        command = build_command(args, arm, run_dir)
        print(f"[{index}/{len(arms)}] {arm.name}", flush=True)
        with open(run_dir / "stdout.log", "w", buffering=1) as log:
            proc = subprocess.run(
                command,
                cwd=Path.cwd(),
                env={**os.environ, "PYTHONPATH": "src"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "schema": "rwkv-lab.rlvr-result.v1",
                "status": "failed",
                "error": f"trainer exited {proc.returncode} without result.json",
            }
        results.append(
            {
                "algorithm": arm.algorithm,
                "seed": arm.seed,
                "run_dir": str(run_dir.resolve()),
                "returncode": proc.returncode,
                **result,
            }
        )
        manifest.update(
            {
                "results": results,
                "summary": summarize(results),
                "completed_arms": len(results),
            }
        )
        _atomic_json(campaign_path, manifest)

    failed = sum(r.get("status") != "complete" for r in results)
    manifest.update(
        {
            "status": "complete" if failed == 0 else "failed",
            "completed_ts": time.time(),
            "elapsed_seconds": time.time() - started,
            "failed_arms": failed,
            "results": results,
            "summary": summarize(results),
        }
    )
    _atomic_json(campaign_path, manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare RLVR algorithms and seeds")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--algorithms", default="gspo,dr_grpo,dapo")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.01)
    ap.add_argument(
        "--reference", choices=["initial", "rollout", "none"], default="rollout"
    )
    ap.add_argument("--reference-ckpt", default="")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-prompts", type=int, default=16)
    ap.add_argument("--eval-group-size", type=int, default=4)
    ap.add_argument("--min-heldout-delta", type=float, default=0.01)
    ap.add_argument("--train-tasks", type=int, default=4096)
    ap.add_argument("--eval-tasks", type=int, default=256)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--log-samples", type=int, default=2)
    ap.add_argument("--verifier-command", default="")
    ap.add_argument("--verifier-timeout", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--resume-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = ap.parse_args()
    result = run_campaign(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
