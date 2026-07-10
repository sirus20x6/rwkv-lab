"""Equal-budget, paired-seed post-training campaigns with confirmation receipts.

Objectives retain their primary implementations and citations in ``preference.py``.  This layer
adds the experimental discipline: frozen-parent comparisons on identical held-out examples,
paired bootstrap intervals, family-regression gates, fresh confirmation seeds, registry records,
and promotion receipts that never equate training completion with promotion. The paired confidence
gate uses Efron's bootstrap, https://doi.org/10.1214/aos/1176344552.
"""
from __future__ import annotations

import argparse
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


def build_command(args, objective: str, seed: int, run_dir: Path) -> list[str]:
    command = [sys.executable, "-m", "rwkv_lab.posttrain_train",
               "--checkpoint", args.checkpoint, "--data", args.data,
               "--output", str(run_dir), "--objective", objective,
               "--adapter-name", "candidate", "--rank", str(args.rank),
               "--alpha", str(args.adapter_alpha), "--steps", str(args.steps),
               "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
               "--beta", str(args.beta), "--gamma", str(args.gamma),
               "--max-length", str(args.max_length), "--seed", str(seed),
               "--device", args.device, "--max-train-tokens", str(args.token_budget),
               "--packing", "audit", "--base-quantization", args.base_quantization,
               "--quant-block-size", str(args.quant_block_size)]
    if args.eval_data:
        command += ["--eval-data", args.eval_data]
    if args.token_cache:
        command += ["--token-cache", args.token_cache]
    if args.targets:
        command += ["--targets", args.targets]
    if args.activation_offload:
        command.append("--activation-offload")
    return command


def run_campaign(args) -> dict[str, Any]:
    objectives = _csv(args.objectives)
    seeds = _csv(args.seeds, int)
    confirm_seeds = _csv(args.confirm_seeds, int)
    invalid = set(objectives) - OBJECTIVES
    if not objectives or invalid or not seeds or not confirm_seeds:
        raise ValueError(f"invalid objectives or empty seed sets: {sorted(invalid)}")
    if set(seeds) & set(confirm_seeds):
        raise ValueError("exploration and confirmation seeds must be disjoint")
    for path in (args.checkpoint, args.data, args.eval_data):
        if path and not Path(path).is_file():
            raise ValueError(f"input does not exist: {path}")
    audit = split_audit(args.data, args.eval_data)
    if not audit["passed"]:
        raise ValueError(f"held-out split audit failed: {audit}")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != "output"}
    campaign_id = create_campaign("posttrain", config, name=args.name, db=args.db)
    started = time.time()
    manifest: dict[str, Any] = {"schema": SCHEMA, "status": "running", "campaign_id": campaign_id,
                                "created_ts": started, "configuration": config,
                                "objectives": objectives, "seeds": seeds,
                                "confirmation_seeds": confirm_seeds, "split_audit": audit,
                                "results": []}
    manifest_path = root / "posttrain-campaign.json"
    _atomic_json(manifest_path, manifest)
    failed = False
    all_seeds = [("explore", seed) for seed in seeds] + [("confirm", seed) for seed in confirm_seeds]
    for objective in objectives:
        parent_arm = ensure_arm(campaign_id, f"parent:{objective}", {"frozen": True}, db=args.db)
        candidate_arm = ensure_arm(campaign_id, objective, {"objective": objective}, db=args.db)
        phase_rows: dict[str, list[dict[str, Any]]] = {"explore": [], "confirm": []}
        for phase, seed in all_seeds:
            run_dir = root / phase / objective / f"seed-{seed:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(args, objective, seed, run_dir)
            with (run_dir / "stdout.log").open("w", buffering=1) as log:
                completed = subprocess.run(command, cwd=Path.cwd(),
                                           env={**os.environ, "PYTHONPATH": "src"},
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            result_path = run_dir / "posttrain-result.json"
            if completed.returncode or not result_path.is_file():
                failed = True
                row = {"objective": objective, "seed": seed, "phase": phase, "status": "failed",
                       "returncode": completed.returncode, "run_dir": str(run_dir)}
                record_trial(campaign_id, candidate_arm, seed, 0, args.token_budget, None,
                             phase=phase, status="failed", error="trainer failed", db=args.db)
            else:
                result = json.loads(result_path.read_text())
                promotion = assess(result, audit=audit, minimum_delta=args.minimum_delta,
                                   maximum_family_regression=args.maximum_family_regression,
                                   confidence=args.confidence, bootstrap_samples=args.bootstrap_samples,
                                   seed=seed, token_budget=args.token_budget,
                                   budget_slack=2 * args.batch_size * args.max_length)
                score0 = -float(result["initial_eval_loss"])
                score1 = -float(result["final_eval_loss"])
                record_trial(campaign_id, parent_arm, seed, 0, args.token_budget,
                             {"score": score0}, phase=phase, db=args.db)
                trial_id = record_trial(campaign_id, candidate_arm, seed, 0, args.token_budget,
                                        {"score": score1, "delta": score1 - score0,
                                         "eligible": promotion["eligible"],
                                         "train_tokens": result["train_tokens"]},
                                        profile=result.get("quantization"), phase=phase, db=args.db)
                record_artifact(campaign_id, str(result_path), "posttrain-result",
                                trial_id=trial_id, db=args.db)
                row = {"objective": objective, "seed": seed, "phase": phase, "status": "complete",
                       "run_dir": str(run_dir), "initial_score": score0, "final_score": score1,
                       "delta": score1 - score0, "promotion": promotion,
                       "adapter": result["adapter"], "result": str(result_path)}
                phase_rows[phase].append(row)
            manifest["results"].append(row)
            _atomic_json(manifest_path, manifest)
        for phase, rows in phase_rows.items():
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
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--token-cache", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="")
    parser.add_argument("--db", default=None)
    print(json.dumps(run_campaign(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
