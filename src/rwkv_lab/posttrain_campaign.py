"""Equal-budget, paired-seed post-training campaigns with confirmation receipts.

Objectives retain their primary implementations and citations in ``preference.py``.  This layer
adds the experimental discipline: frozen-parent comparisons on identical held-out examples,
paired bootstrap intervals, family-regression gates, fresh confirmation seeds, registry records,
and promotion receipts that never equate training completion with promotion. The paired confidence
gate uses Efron's bootstrap, https://doi.org/10.1214/aos/1176344552.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from rwkv_lab.experiment_analysis import paired_stats
from rwkv_lab.registry import (create_campaign, ensure_arm, finish_campaign, record_artifact,
                               record_comparison, record_trial)
from rwkv_lab.rlvr_evaluation import promotion_gates
from rwkv_lab.posttrain_data import load_jsonl


SCHEMA = "rwkv-lab.posttrain-campaign.v1"
PROMOTION_SCHEMA = "rwkv-lab.posttrain-promotion.v1"
OBJECTIVES = frozenset({"sft", "dpo", "kto", "orpo", "simpo", "reward", "prm"})


def _csv(value: str, convert=str) -> list:
    return [convert(item.strip()) for item in value.split(",") if item.strip()]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _content(row) -> str:
    value = {"kind": row.kind, "messages": [message.__dict__ for message in row.messages],
             "text": row.text, "chosen": row.chosen, "rejected": row.rejected,
             "response": row.response, "label": row.label,
             "steps": [step.__dict__ for step in row.steps]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def split_audit(data: str, eval_data: str = "") -> dict[str, Any]:
    rows, _ = load_jsonl(data)
    train = {row.id: _content(row) for row in rows if row.split == "train"}
    if eval_data:
        eval_rows, _ = load_jsonl(eval_data)
        heldout = {row.id: _content(row) for row in eval_rows}
    else:
        heldout = {row.id: _content(row) for row in rows if row.split in ("eval", "test")}
    overlap = sorted(set(train.values()) & set(heldout.values()))
    return {"schema": "rwkv-lab.posttrain-split-audit.v1", "train": len(train),
            "heldout": len(heldout), "content_overlaps": len(overlap),
            "id_overlaps": sorted(set(train) & set(heldout))[:20],
            "passed": bool(heldout) and not overlap and not (set(train) & set(heldout))}


def _summaries(result: dict[str, Any], key: str) -> dict[str, Any]:
    details = result[key]
    rows = details.get("per_example") or []
    task_rewards = {str(row["id"]): -float(row["loss"]) for row in rows}
    families: dict[str, list[float]] = {}
    for row in rows:
        families.setdefault(str(row.get("family") or "default"), []).append(-float(row["loss"]))
    return {"task_rewards": task_rewards,
            "family_rewards": {name: statistics.fmean(values) for name, values in families.items()}}


def assess(result: dict[str, Any], *, audit: dict[str, Any], minimum_delta: float,
           maximum_family_regression: float, confidence: float, bootstrap_samples: int,
           seed: int, token_budget: int, budget_slack: int) -> dict[str, Any]:
    baseline = _summaries(result, "initial_eval")
    candidate = _summaries(result, "eval")
    gates = promotion_gates(
        baseline, candidate, minimum_delta=minimum_delta,
        updates_applied=int(result.get("steps", 0)),
        maximum_family_regression=maximum_family_regression, require_confidence=True,
        bootstrap_samples=bootstrap_samples, confidence=confidence, seed=seed,
        split_audit=audit, rollout_tokens=int(result.get("train_tokens", 0)), elapsed_seconds=0.0,
        maximum_rollout_tokens=0,
    )
    used = int(result.get("train_tokens", 0))
    gates["gates"]["equal_token_budget"] = token_budget <= used <= token_budget + budget_slack
    prm = result.get("eval", {})
    if result.get("objective") == "prm":
        calibration = prm.get("calibration") or {}
        adversarial = prm.get("adversarial") or {}
        gates["gates"]["prm_calibration"] = float(calibration.get("ece", 1.0)) <= 0.15
        gates["gates"]["prm_adversarial"] = (
            bool(adversarial) and float(adversarial.get("adversarial_false_positive_rate", 1.0)) <= 0.25)
        gates["eligible"] = all(gates["gates"].values())
    else:
        gates["eligible"] = all(gates["gates"].values())
    return gates


def build_command(args, objective: str, seed: int, run_dir: Path, *, device: str | None = None) -> list[str]:
    command = [sys.executable, "-m", "rwkv_lab.posttrain_train",
               "--checkpoint", args.checkpoint, "--data", args.data,
               "--output", str(run_dir), "--objective", objective,
               "--adapter-name", "candidate", "--rank", str(args.rank),
               "--alpha", str(args.adapter_alpha), "--steps", str(args.steps),
               "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
               "--beta", str(args.beta), "--gamma", str(args.gamma),
               "--max-length", str(args.max_length), "--seed", str(seed),
               "--device", device or args.device, "--max-train-tokens", str(args.token_budget),
               "--packing", args.packing, "--base-quantization", args.base_quantization,
               "--quant-block-size", str(args.quant_block_size),
               "--quant-backend", args.quant_backend,
               "--log-every", str(getattr(args, "log_every", 10))]
    if args.eval_data:
        command += ["--eval-data", args.eval_data]
    if args.token_cache:
        command += ["--token-cache", args.token_cache]
    if args.targets:
        command += ["--targets", args.targets]
    if args.activation_offload:
        command.append("--activation-offload")
    if getattr(args, "template", ""):
        command += ["--template", args.template]
    return command


def _execute_arm(args, objective: str, seed: int, phase: str, run_dir: Path,
                 device: str) -> dict[str, Any]:
    """Run/recover one deterministic arm and persist attempt state atomically."""
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, objective, seed, run_dir, device=device)
    command_hash = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    state_path = run_dir / "arm-state.json"
    result_path = run_dir / "posttrain-result.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "schema": "rwkv-lab.posttrain-arm-state.v1", "objective": objective,
        "seed": seed, "phase": phase, "command_sha256": command_hash,
        "device": device, "attempts": [], "status": "pending",
    }
    if state.get("command_sha256") != command_hash:
        raise ValueError(f"saved arm command differs for {phase}/{objective}/seed-{seed}")
    if args.resume and state.get("status") == "complete" and result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
            cached_valid = (result.get("schema") == "rwkv-lab.posttrain-result.v1" and
                            result.get("objective") == objective and
                            int(result.get("seed")) == seed)
        except (ValueError, TypeError, json.JSONDecodeError):
            cached_valid = False
        if cached_valid:
            return {"result": result, "returncode": 0, "attempts": len(state["attempts"]),
                    "resumed": True, "device": device}
        print(f"resume: cached posttrain-result.json for {phase}/{objective}/seed-{seed} "
              f"failed objective/seed validation; ignoring cache and rerunning", flush=True)
    last_returncode = 1
    for retry in range(args.retries + 1):
        attempt = len(state["attempts"]) + 1
        result_path.unlink(missing_ok=True)
        state.update({"status": "running", "device": device, "started_ts": time.time()})
        state["attempts"].append({"attempt": attempt, "started_ts": time.time(),
                                  "status": "running"})
        _atomic_json(state_path, state)
        log_path = run_dir / f"stdout-attempt-{attempt:02d}.log"
        started = time.time()
        timed_out = False
        with log_path.open("w", buffering=1) as log:
            try:
                completed = subprocess.run(command, cwd=Path.cwd(),
                                           env={**os.environ, "PYTHONPATH": "src"},
                                           stdout=log, stderr=subprocess.STDOUT, check=False,
                                           timeout=(args.arm_timeout if args.arm_timeout > 0 else None))
                last_returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                last_returncode = 124
                log.write(f"arm exceeded {args.arm_timeout}s timeout\n")
        valid = False
        result = None
        if last_returncode == 0 and result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                valid = (result.get("schema") == "rwkv-lab.posttrain-result.v1" and
                         result.get("objective") == objective and int(result.get("seed")) == seed)
            except (ValueError, TypeError, json.JSONDecodeError):
                valid = False
        attempt_row = state["attempts"][-1]
        attempt_row.update({"finished_ts": time.time(), "elapsed_seconds": time.time() - started,
                            "returncode": last_returncode, "timed_out": timed_out,
                            "status": "complete" if valid else "failed", "log": str(log_path)})
        state["status"] = "complete" if valid else "retrying" if retry < args.retries else "failed"
        _atomic_json(state_path, state)
        if valid:
            return {"result": result, "returncode": 0, "attempts": len(state["attempts"]),
                    "resumed": False, "device": device}
        if retry < args.retries and args.retry_delay > 0:
            time.sleep(args.retry_delay)
    return {"result": None, "returncode": last_returncode, "attempts": len(state["attempts"]),
            "resumed": False, "device": device}


def _upsert_result(results: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["objective"], row["phase"], int(row["seed"]))
    for index, existing in enumerate(results):
        if (existing["objective"], existing["phase"], int(existing["seed"])) == key:
            results[index] = row
            return
    results.append(row)


def run_campaign(args) -> dict[str, Any]:
    objectives = _csv(args.objectives)
    seeds = _csv(args.seeds, int)
    confirm_seeds = _csv(args.confirm_seeds, int)
    invalid = set(objectives) - OBJECTIVES
    if not objectives or invalid or not seeds or not confirm_seeds:
        raise ValueError(f"invalid objectives or empty seed sets: {sorted(invalid)}")
    if set(seeds) & set(confirm_seeds):
        raise ValueError("exploration and confirmation seeds must be disjoint")
    if args.retries < 0 or args.retry_delay < 0 or args.arm_timeout < 0 or args.max_parallel < 0:
        raise ValueError("retry, timeout, and parallelism controls must be non-negative")
    for path in (args.checkpoint, args.data, args.eval_data):
        if path and not Path(path).is_file():
            raise ValueError(f"input does not exist: {path}")
    audit = split_audit(args.data, args.eval_data)
    if not audit["passed"]:
        raise ValueError(f"held-out split audit failed: {audit}")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key not in {"output", "resume"}}
    started = time.time()
    manifest_path = root / "posttrain-campaign.json"
    if manifest_path.is_file():
        if not args.resume:
            raise ValueError("campaign output already exists; pass --resume or choose a new output")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("configuration") != config:
            raise ValueError("saved campaign configuration differs; choose a new output")
        if manifest.get("status") == "complete":
            return manifest
        campaign_id = int(manifest["campaign_id"])
        manifest["status"] = "running"
        manifest.pop("completed_ts", None)
    else:
        campaign_id = create_campaign("posttrain", config, name=args.name, db=args.db)
        manifest = {"schema": SCHEMA, "status": "running", "campaign_id": campaign_id,
                    "created_ts": started, "configuration": config,
                    "objectives": objectives, "seeds": seeds,
                    "confirmation_seeds": confirm_seeds, "split_audit": audit,
                    "results": []}
    _atomic_json(manifest_path, manifest)
    failed = False
    devices = _csv(args.devices) if args.devices else [args.device]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("campaign devices must be a unique comma-separated list")
    parallel = min(len(devices), max(1, int(args.max_parallel or len(devices))))
    devices = devices[:parallel]
    arm_ids = {}
    for objective in objectives:
        parent_arm = ensure_arm(campaign_id, f"parent:{objective}", {"frozen": True}, db=args.db)
        candidate_arm = ensure_arm(campaign_id, objective, {"objective": objective}, db=args.db)
        arm_ids[objective] = (parent_arm, candidate_arm)
    try:
        for phase, phase_seeds in (("explore", seeds), ("confirm", confirm_seeds)):
            jobs = [(objective, seed, root / phase / objective / f"seed-{seed:04d}")
                    for objective in objectives for seed in phase_seeds]
            for start in range(0, len(jobs), len(devices)):
                batch = jobs[start:start + len(devices)]
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    futures = [executor.submit(_execute_arm, args, objective, seed, phase, run_dir,
                                               devices[index])
                               for index, (objective, seed, run_dir) in enumerate(batch)]
                    outcomes = [future.result() for future in futures]
                for (objective, seed, run_dir), outcome in zip(batch, outcomes):
                    parent_arm, candidate_arm = arm_ids[objective]
                    result_path = run_dir / "posttrain-result.json"
                    result = outcome["result"]
                    if result is None:
                        failed = True
                        row = {"objective": objective, "seed": seed, "phase": phase,
                               "status": "failed", "returncode": outcome["returncode"],
                               "attempts": outcome["attempts"], "device": outcome["device"],
                               "run_dir": str(run_dir)}
                        record_trial(campaign_id, candidate_arm, seed, 0, args.token_budget, None,
                                     phase=phase, status="failed", error="trainer failed after retries",
                                     db=args.db)
                        _upsert_result(manifest["results"], row)
                        _atomic_json(manifest_path, manifest)
                        continue
                    promotion = assess(result, audit=audit, minimum_delta=args.minimum_delta,
                                       maximum_family_regression=args.maximum_family_regression,
                                       confidence=args.confidence,
                                       bootstrap_samples=args.bootstrap_samples,
                                       seed=seed, token_budget=args.token_budget,
                                       budget_slack=2 * args.batch_size * args.max_length)
                    score0 = -float(result["initial_eval_loss"])
                    score1 = -float(result["final_eval_loss"])
                    record_trial(campaign_id, parent_arm, seed, 0, args.token_budget,
                                 {"score": score0}, phase=phase, db=args.db)
                    trial_id = record_trial(
                        campaign_id, candidate_arm, seed, 0, args.token_budget,
                        {"score": score1, "delta": score1 - score0,
                         "eligible": promotion["eligible"],
                         "train_tokens": result["train_tokens"]},
                        profile=result.get("quantization"), phase=phase, db=args.db)
                    record_artifact(campaign_id, str(result_path), "posttrain-result",
                                    trial_id=trial_id, db=args.db)
                    row = {"objective": objective, "seed": seed, "phase": phase,
                           "status": "complete", "run_dir": str(run_dir),
                           "initial_score": score0, "final_score": score1,
                           "delta": score1 - score0, "promotion": promotion,
                           "adapter": result["adapter"], "result": str(result_path),
                           "attempts": outcome["attempts"], "resumed": outcome["resumed"],
                           "device": outcome["device"]}
                    _upsert_result(manifest["results"], row)
                    _atomic_json(manifest_path, manifest)
    except (KeyboardInterrupt, SystemExit):
        manifest.update({"status": "interrupted", "interrupted_ts": time.time(),
                         "elapsed_seconds": time.time() - started})
        _atomic_json(manifest_path, manifest)
        finish_campaign(campaign_id, "interrupted", db=args.db)
        raise
    except Exception as exc:
        manifest.update({"status": "failed", "failed_ts": time.time(),
                         "elapsed_seconds": time.time() - started, "error": repr(exc)})
        _atomic_json(manifest_path, manifest)
        finish_campaign(campaign_id, "failed", db=args.db)
        raise

    manifest["comparisons"] = {}
    for objective in objectives:
        parent_arm, candidate_arm = arm_ids[objective]
        for phase in ("explore", "confirm"):
            rows = [row for row in manifest["results"] if row["objective"] == objective
                    and row["phase"] == phase and row["status"] == "complete"]
            if not rows:
                continue
            stats = paired_stats([row["initial_score"] for row in rows],
                                 [row["final_score"] for row in rows], seed=campaign_id + len(rows),
                                 alpha=1.0 - args.confidence)
            stats["p_adjusted"] = stats["p_value"]
            record_comparison(campaign_id, candidate_arm, parent_arm, "score", phase, stats,
                              confirmed=phase == "confirm", db=args.db)
            manifest.setdefault("comparisons", {}).setdefault(objective, {})[phase] = stats
    receipts = []
    for objective in objectives:
        confirmations = [row for row in manifest["results"] if row["objective"] == objective
                         and row["phase"] == "confirm" and row["status"] == "complete"]
        stats = manifest.get("comparisons", {}).get(objective, {}).get("confirm", {})
        eligible = (bool(confirmations) and all(row["promotion"]["eligible"] for row in confirmations)
                    and float(stats.get("ci_low", float("-inf"))) >= args.minimum_delta)
        selected = max(confirmations, key=lambda row: row["final_score"], default=None)
        receipt = {"schema": PROMOTION_SCHEMA, "objective": objective, "eligible": eligible,
                   "reason": "fresh confirmation passed" if eligible else "confirmation gates failed",
                   "parent_checkpoint": str(Path(args.checkpoint).resolve()),
                   "split_audit": audit, "exploration_seeds": seeds,
                   "confirmation_seeds": confirm_seeds, "confirmation": stats,
                   "confirmation_runs": [{"seed": row["seed"],
                                          "promotion": row["promotion"]}
                                         for row in confirmations],
                   "selected_adapter": (str(Path(selected["run_dir"]) / "adapter")
                                        if eligible and selected else ""),
                   "selected_result": selected["result"] if eligible and selected else ""}
        receipt_path = root / f"promotion-{objective}.json"
        _atomic_json(receipt_path, receipt)
        record_artifact(campaign_id, str(receipt_path), "promotion-receipt", db=args.db)
        receipts.append({**receipt, "path": str(receipt_path)})
    manifest.update({"status": "failed" if failed else "complete", "completed_ts": time.time(),
                     "elapsed_seconds": time.time() - started, "promotion_receipts": receipts})
    _atomic_json(manifest_path, manifest)
    finish_campaign(campaign_id, manifest["status"], db=args.db)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired post-training campaign and confirmation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--objectives", default="sft,dpo,kto,orpo,simpo")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--confirm-seeds", default="100,101,102")
    parser.add_argument("--token-budget", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--adapter-alpha", type=float, default=32.0)
    parser.add_argument("--targets", default="")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--maximum-family-regression", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--base-quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--quant-block-size", type=int, default=64)
    parser.add_argument("--quant-backend", choices=["auto", "portable", "torchao"], default="auto")
    parser.add_argument("--packing", choices=["off", "audit", "reset"], default="reset")
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--template", default="")
    parser.add_argument("--token-cache", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--devices", default="",
                        help="comma-separated device slots, e.g. cuda:0,cuda:1")
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="maximum concurrent arms; 0 uses one per listed device")
    parser.add_argument("--arm-timeout", type=float, default=0.0,
                        help="hard seconds per attempt; 0 disables")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--db", default=None)
    print(json.dumps(run_campaign(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
