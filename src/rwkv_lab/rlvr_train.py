"""End-to-end reinforcement learning with verifiable rewards.

Primary algorithm references:

* Dr.GRPO, arXiv:2503.20783, https://arxiv.org/abs/2503.20783.
* DAPO, arXiv:2503.14476, https://arxiv.org/abs/2503.14476.
* GSPO, arXiv:2507.18071, https://arxiv.org/abs/2507.18071.
* Absolute Zero, arXiv:2505.03335, https://arxiv.org/abs/2505.03335.

The trainer closes the model-side loop: grouped policy rollouts, deterministic
verification, clipped RLVR updates, held-out evaluation, and lineage-preserving
checkpoints.  It deliberately does not execute generated code. Code tasks use
the versioned external-verifier protocol below so Adamaton can provide an
isolated sandbox and independent verifier fleet.

Task JSONL (``rwkv-lab.rlvr-task.v1``), one object per line::

    {"id":"t1", "split":"train", "prompt":"Compute 17*9.",
     "verifier":{"kind":"numeric", "expected":153}, "metadata":{}}

For ``kind=external``, ``--verifier-command`` receives one JSON document on
stdin with ``schema=rwkv-lab.rlvr-verify-request.v1`` and returns
``schema=rwkv-lab.rlvr-verify-response.v1`` plus ``[{id,reward,details}]``.
Commands are executed directly (never through a shell) with a timeout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rwkv_lab.rlvr import (ExactAnswerVerifier, NumericAnswerVerifier,
                           PythonExpressionVerifier, group_advantages,
                           policy_loss, token_log_probs)


TASK_SCHEMA = "rwkv-lab.rlvr-task.v1"
VERIFY_REQUEST_SCHEMA = "rwkv-lab.rlvr-verify-request.v1"
VERIFY_RESPONSE_SCHEMA = "rwkv-lab.rlvr-verify-response.v1"
RESULT_SCHEMA = "rwkv-lab.rlvr-result.v1"
MANIFEST_SCHEMA = "rwkv-lab.rlvr-manifest.v1"


@dataclass(frozen=True)
class RLVRTask:
    id: str
    prompt: str
    verifier: dict[str, Any]
    split: str = "train"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line: int = 0) -> "RLVRTask":
        verifier = value.get("verifier", {})
        if isinstance(verifier, str):
            verifier = {"kind": verifier, "expected": value.get("expected")}
        if not isinstance(verifier, dict) or not verifier.get("kind"):
            raise ValueError(f"task line {line}: verifier must contain kind")
        task = cls(str(value.get("id") or f"line-{line}"), str(value.get("prompt") or ""),
                   dict(verifier), str(value.get("split") or "train"),
                   dict(value.get("metadata") or {}))
        if not task.prompt:
            raise ValueError(f"task line {line}: prompt is required")
        if task.split not in ("train", "eval"):
            raise ValueError(f"task line {line}: split must be train or eval")
        return task


@dataclass
class Rollout:
    id: str
    task: RLVRTask
    prompt_ids: list[int]
    response_ids: list[int]
    response: str


def load_task_jsonl(path: str | Path) -> tuple[list[RLVRTask], str]:
    raw = Path(path).read_bytes()
    tasks = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            tasks.append(RLVRTask.from_dict(json.loads(line), line=line_no))
    if not tasks:
        raise ValueError("task JSONL is empty")
    return tasks, hashlib.sha256(raw).hexdigest()


def _apply(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unsupported arithmetic operator {op}")


def arithmetic_curriculum(count: int, *, seed: int, split: str,
                          difficulty: int = 2) -> list[RLVRTask]:
    """Generate a deterministic, leak-free integer arithmetic curriculum."""

    rng = random.Random(seed)
    tasks = []
    bound = 10 ** min(max(int(difficulty), 1), 4)
    ops = ("+", "-") if difficulty <= 1 else ("+", "-", "*")
    for i in range(int(count)):
        a, b, op1 = rng.randint(0, bound), rng.randint(0, bound), rng.choice(ops)
        value = _apply(op1, a, b)
        expression = f"{a} {op1} {b}"
        if difficulty >= 3:
            c, op2 = rng.randint(0, max(10, bound // 2)), rng.choice(ops)
            expression = f"({expression}) {op2} {c}"
            value = _apply(op2, value, c)
        task_id = hashlib.sha256(f"{seed}:{split}:{i}:{expression}".encode()).hexdigest()[:16]
        tasks.append(RLVRTask(
            id=f"arithmetic-{split}-{task_id}", split=split,
            prompt=f"Compute {expression}. Return only the final integer, with no prose.",
            verifier={"kind": "numeric", "expected": value},
            metadata={"family": "arithmetic", "difficulty": difficulty},
        ))
    return tasks


def split_task_pool(tasks: Sequence[RLVRTask]) -> tuple[list[RLVRTask], list[RLVRTask]]:
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("RLVR task ids must be globally unique")
    train, evaluate = [t for t in tasks if t.split == "train"], [t for t in tasks if t.split == "eval"]
    if not evaluate and len(train) >= 10:
        # Stable hidden split when a producer omitted explicit splits.
        ranked = sorted(train, key=lambda t: hashlib.sha256(t.id.encode()).digest())
        evaluate = ranked[:max(1, len(ranked) // 10)]
        eval_ids = {t.id for t in evaluate}
        train = [t for t in train if t.id not in eval_ids]
    if not train or not evaluate:
        raise ValueError("RLVR needs non-empty, disjoint train and eval task splits")
    if {t.id for t in train} & {t.id for t in evaluate}:
        raise ValueError("task ids overlap between train and eval splits")
    return train, evaluate


def _local_verifier(task: RLVRTask):
    kind, spec = str(task.verifier.get("kind")), task.verifier
    if kind == "exact":
        return ExactAnswerVerifier(spec.get("expected", ""),
                                   case_sensitive=bool(spec.get("case_sensitive", False)))
    if kind == "expression":
        return PythonExpressionVerifier(float(spec["expected"]), atol=float(spec.get("atol", 1e-6)))
    if kind == "numeric":
        return NumericAnswerVerifier(float(spec["expected"]), atol=float(spec.get("atol", 1e-6)))
    if kind == "external":
        return None
    raise ValueError(f"task {task.id}: unknown verifier kind {kind!r}")


def verify_rollouts(rollouts: Sequence[Rollout], *, external_command: Sequence[str] = (),
                    timeout: float = 10.0) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Verify local items directly and external items in one bounded RPC."""

    rewards: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    external = []
    for rollout in rollouts:
        verifier = _local_verifier(rollout.task)
        if verifier is None:
            external.append({"id": rollout.id, "task_id": rollout.task.id,
                             "prompt": rollout.task.prompt, "response": rollout.response,
                             "verifier": rollout.task.verifier,
                             "metadata": rollout.task.metadata or {}})
        else:
            rewards[rollout.id] = float(verifier(rollout.response))
            details[rollout.id] = {"source": "local", "kind": rollout.task.verifier["kind"]}

    if external:
        if not external_command:
            raise ValueError("external verifier tasks require --verifier-command")
        payload = json.dumps({"schema": VERIFY_REQUEST_SCHEMA, "items": external})
        proc = subprocess.run(list(external_command), input=payload, text=True, capture_output=True,
                              timeout=float(timeout), check=False)
        if proc.returncode:
            raise RuntimeError(f"external verifier exited {proc.returncode}: {proc.stderr[-2000:]}")
        result = json.loads(proc.stdout)
        if result.get("schema") != VERIFY_RESPONSE_SCHEMA or not isinstance(result.get("rewards"), list):
            raise ValueError("external verifier returned an invalid schema")
        for item in result["rewards"]:
            rid, reward = str(item.get("id")), float(item.get("reward"))
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(f"external verifier reward for {rid!r} is outside [0,1]")
            rewards[rid] = reward
            details[rid] = {"source": "external", **dict(item.get("details") or {})}

    missing = [r.id for r in rollouts if r.id not in rewards]
    if missing:
        raise ValueError(f"verifier omitted rollout ids: {missing[:4]}")
    return (torch.tensor([rewards[r.id] for r in rollouts], dtype=torch.float32),
            [details[r.id] for r in rollouts])


def _logits(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def sample_response(model, prompt_ids: Sequence[int], *, max_new: int, temperature: float,
                    top_p: float, top_k: int, stop_token: int, device: str,
                    seed: int) -> list[int]:
    """Full-prefix reference sampler; includes the stop token in policy tokens."""

    if not prompt_ids:
        raise ValueError("tokenized prompt is empty")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    response = []
    for _ in range(max(1, int(max_new))):
        logits = _logits(model(tokens))[0, -1].float()
        logits[0] = -float("inf")
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = logits.topk(min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            if 0 < top_p < 1:
                probs, order = logits.softmax(-1).sort(descending=True)
                keep = int((probs.cumsum(-1) < top_p).sum()) + 1
                filtered = torch.full_like(logits, -float("inf"))
                filtered[order[:keep]] = logits[order[:keep]]
                logits = filtered
            nxt = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        response.append(nxt)
        if nxt == stop_token:
            break
        tokens = torch.cat((tokens, tokens.new_tensor([[nxt]])), dim=1)
    return response


def generate_rollouts(model, tokenizer, tasks: Sequence[RLVRTask], *, group_size: int,
                      max_new: int, temperature: float, top_p: float, top_k: int,
                      stop_token: int, device: str, seed: int) -> list[Rollout]:
    model.eval()
    out = []
    for group, task in enumerate(tasks):
        prompt = f"User: {task.prompt}\n\nAssistant:"
        prompt_ids = tokenizer.encode(prompt)
        for k in range(int(group_size)):
            response_ids = sample_response(model, prompt_ids, max_new=max_new,
                                           temperature=temperature, top_p=top_p, top_k=top_k,
                                           stop_token=stop_token, device=device,
                                           seed=seed + group * 100_003 + k)
            visible = response_ids[:-1] if response_ids and response_ids[-1] == stop_token else response_ids
            out.append(Rollout(f"{task.id}:{k}", task, list(prompt_ids), response_ids,
                               tokenizer.decode(visible)))
    return out


def response_log_probs(model, rollouts: Sequence[Rollout]) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably score only response tokens, padding to ``[rollouts,max_response]``."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    device = next(model.parameters()).device
    max_response = max(len(r.response_ids) for r in rollouts)
    if max_response <= 0:
        raise ValueError("rollouts must contain at least one policy token")
    rows, masks = [], []
    for rollout in rollouts:
        sequence = rollout.prompt_ids + rollout.response_ids
        ids = torch.tensor(sequence, dtype=torch.long, device=device)
        logits = _logits(model(ids[:-1].unsqueeze(0)))[0]
        targets = ids[1:]
        start = len(rollout.prompt_ids) - 1
        logp = token_log_probs(logits[start:start + len(rollout.response_ids)],
                               targets[start:start + len(rollout.response_ids)])
        pad = max_response - len(rollout.response_ids)
        rows.append(F.pad(logp, (0, pad)))
        masks.append(F.pad(torch.ones_like(logp), (0, pad)))
    return torch.stack(rows), torch.stack(masks)


def grouped_metrics(rewards: torch.Tensor, group_size: int) -> dict[str, float]:
    groups = rewards.view(-1, int(group_size))
    return {"reward": float(rewards.mean()), "pass_at_1": float(groups[:, 0].mean()),
            "pass_at_k": float((groups.max(-1).values > 0).float().mean()),
            "reward_std": float(rewards.std(unbiased=False))}


def promotion_decision(baseline: dict[str, float], candidate: dict[str, float], *,
                       minimum_delta: float, updates_applied: int,
                       candidate_checkpoint: str, rollback_checkpoint: str) -> dict[str, Any]:
    heldout_delta = float(candidate.get("reward", 0.0) - baseline.get("reward", 0.0))
    eligible = updates_applied > 0 and heldout_delta >= minimum_delta
    return {"eligible": eligible, "heldout_delta": heldout_delta,
            "minimum_delta": float(minimum_delta), "updates_applied": int(updates_applied),
            "candidate_checkpoint": candidate_checkpoint, "rollback_checkpoint": rollback_checkpoint,
            "reason": ("heldout improvement passed" if eligible else
                       "insufficient heldout gain or no informative policy update")}


def optimize_rollouts(model, optimizer, rollouts: Sequence[Rollout], rewards: torch.Tensor, *,
                      group_size: int, algorithm: str, epochs: int, clip_low: float,
                      clip_high: float, kl_coef: float, grad_clip: float,
                      reference_model=None, token_normalizer: int | None = None) -> dict[str, float]:
    """Apply one grouped RLVR update and return optimization diagnostics."""

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        old_logp, mask = response_log_probs(model, rollouts)
        if reference_model is None:
            reference_logp = old_logp
        else:
            reference_model.eval()
            reference_logp, _ = response_log_probs(reference_model, rollouts)
    rewards = rewards.to(device)
    group_ids = torch.arange(len(rollouts) // group_size, device=device).repeat_interleave(group_size)
    advantages, active = group_advantages(rewards, group_ids,
                                          standardize=(algorithm != "dr_grpo"),
                                          drop_constant=(algorithm == "dapo"))
    has_signal = bool(active.any() and advantages.abs().sum() > 0)
    diagnostics = {"loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
                   "grad_norm": 0.0, "active_fraction": float(active.float().mean()),
                   "update_applied": float(has_signal)}
    if not has_signal:
        return diagnostics

    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logp, current_mask = response_log_probs(model, rollouts)
        out = policy_loss(logp, old_logp, rewards, group_ids, current_mask,
                          algorithm=algorithm, clip_low=clip_low, clip_high=clip_high,
                          reference_logp=reference_logp, kl_coef=kl_coef,
                          token_normalizer=token_normalizer)
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        diagnostics = {"loss": float(out.loss.detach()), "approx_kl": float(out.approx_kl),
                       "clip_fraction": float(out.clip_fraction), "grad_norm": float(grad_norm),
                       "active_fraction": float(out.active_groups.float().mean()),
                       "update_applied": 1.0}
    return diagnostics


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _checkpoint_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: Path, model, optimizer, source_blob: dict[str, Any], *, step: int,
                    manifest: dict[str, Any], rng: random.Random) -> None:
    blob = {k: v for k, v in source_blob.items() if k not in ("model", "opt", "ema", "rlvr_optimizer")}
    blob.update({"model": model.state_dict(), "rlvr_optimizer": optimizer.state_dict(),
                 "rlvr_step": step, "rlvr": manifest, "rlvr_python_rng": rng.getstate(),
                 "rlvr_torch_rng": torch.get_rng_state()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_policy(model, tokenizer, tasks: Sequence[RLVRTask], *, group_size: int,
                    max_new: int, temperature: float, top_p: float, top_k: int,
                    stop_token: int, device: str, seed: int,
                    external_command: Sequence[str], verifier_timeout: float) -> tuple[dict[str, float], list[Rollout]]:
    rollouts = generate_rollouts(model, tokenizer, tasks, group_size=group_size, max_new=max_new,
                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                 stop_token=stop_token, device=device, seed=seed)
    rewards, _ = verify_rollouts(rollouts, external_command=external_command, timeout=verifier_timeout)
    return grouped_metrics(rewards, group_size), rollouts


def run(args) -> dict[str, Any]:
    from rwkv_lab.generate import WorldVocab, build_from_ckpt
    from rwkv_lab.rwkv_pretrain import build_optimizer

    if args.group_size < 2 or args.prompts_per_step < 1 or args.max_new < 1:
        raise ValueError("group-size must be >=2; prompts-per-step and max-new must be positive")
    if args.eval_group_size < 1 or args.eval_prompts < 1:
        raise ValueError("eval-group-size and eval-prompts must be positive")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.resume or args.ckpt
    model, source_blob = build_from_ckpt(source_path, args.device,
                                         use_ema=(args.use_ema and not args.resume))
    reference_model = None
    reference_path = args.reference_ckpt or args.ckpt
    if args.reference == "initial":
        reference_model, _ = build_from_ckpt(reference_path, args.device, use_ema=args.use_ema)
        reference_model.requires_grad_(False).eval()
    vocab = WorldVocab(args.vocab)

    if args.tasks:
        all_tasks, task_hash = load_task_jsonl(args.tasks)
    else:
        train_generated = arithmetic_curriculum(args.train_tasks, seed=args.seed + 101,
                                                split="train", difficulty=args.difficulty)
        eval_generated = arithmetic_curriculum(args.eval_tasks, seed=args.seed + 1_000_003,
                                               split="eval", difficulty=args.difficulty)
        all_tasks = train_generated + eval_generated
        task_hash = hashlib.sha256("\n".join(json.dumps(asdict(t), sort_keys=True)
                                             for t in all_tasks).encode()).hexdigest()
    train_tasks, eval_tasks = split_task_pool(all_tasks)
    external_command = shlex.split(args.verifier_command) if args.verifier_command else []
    if any(t.verifier.get("kind") == "external" for t in all_tasks) and not external_command:
        raise ValueError("external verifier tasks require --verifier-command")

    arch = source_blob.get("arch") or {}
    u_mup_cfg = None
    if arch.get("u_mup_base_width"):
        from rwkv_lab.u_mup import UMuPConfig
        u_mup_cfg = UMuPConfig(int(arch["u_mup_base_width"]), int(arch["d_model"]),
                               int(arch["n_layers"]), int(arch.get("u_mup_base_depth") or 1))
    optimizer = build_optimizer(model.named_parameters(), args.optimizer, args.lr,
                                args.weight_decay, u_mup_config=u_mup_cfg)
    start_step = int(source_blob.get("rlvr_step", 0)) if args.resume else 0
    if args.steps < start_step:
        raise ValueError(f"--steps {args.steps} precedes resumed RLVR step {start_step}")
    if args.resume and source_blob.get("rlvr_optimizer"):
        optimizer.load_state_dict(source_blob["rlvr_optimizer"])

    parent_hash = _checkpoint_hash(source_path)
    manifest = {"schema": MANIFEST_SCHEMA, "algorithm": args.algorithm,
                "parent_checkpoint": str(Path(source_path).resolve()), "parent_sha256": parent_hash,
                "reference": args.reference, "reference_checkpoint": str(Path(reference_path).resolve()),
                "task_source": str(Path(args.tasks).resolve()) if args.tasks else "generated:arithmetic",
                "task_sha256": task_hash, "seed": args.seed, "group_size": args.group_size,
                "prompts_per_step": args.prompts_per_step, "max_new": args.max_new,
                "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "created_ts": time.time()}
    rng = random.Random(args.seed)
    if args.resume and source_blob.get("rlvr_python_rng"):
        rng.setstate(source_blob["rlvr_python_rng"])
    clip_high = (0.28 if args.algorithm == "dapo" else 0.2) if args.clip_high < 0 else args.clip_high
    log = open(out_dir / "train.jsonl", "a" if args.resume else "w", buffering=1)
    def emit(row):
        return log.write(json.dumps(row, sort_keys=True) + "\n")
    fixed_eval = eval_tasks[:min(args.eval_prompts, len(eval_tasks))]
    stored_baseline = (source_blob.get("rlvr") or {}).get("baseline_heldout") if args.resume else None
    if stored_baseline:
        baseline_eval = {k: float(v) for k, v in stored_baseline.items()}
    else:
        baseline_eval, _ = evaluate_policy(
            model, vocab, fixed_eval, group_size=args.eval_group_size,
            max_new=args.max_new, temperature=args.eval_temperature, top_p=args.top_p,
            top_k=args.top_k, stop_token=args.stop_token, device=args.device,
            seed=args.seed + 9_000_001, external_command=external_command,
            verifier_timeout=args.verifier_timeout)
    manifest["baseline_heldout"] = baseline_eval
    _atomic_json(out_dir / "manifest.json", manifest)
    emit({"kind": "eval", "step": start_step, "split": "heldout",
          "phase": "baseline", **baseline_eval})
    last_eval: dict[str, float] = dict(baseline_eval)
    last_eval_step = start_step
    updates_applied = int((source_blob.get("rlvr") or {}).get("updates_applied", 0)) if args.resume else 0
    checkpoint_path = out_dir / "rlvr.pt"
    started = time.time()

    for step in range(start_step, args.steps):
        chosen = rng.sample(train_tasks, min(args.prompts_per_step, len(train_tasks)))
        rollouts = generate_rollouts(model, vocab, chosen, group_size=args.group_size,
                                     max_new=args.max_new, temperature=args.temperature,
                                     top_p=args.top_p, top_k=args.top_k, stop_token=args.stop_token,
                                     device=args.device, seed=args.seed + step * 1_000_003)
        rewards, verifier_details = verify_rollouts(rollouts, external_command=external_command,
                                                    timeout=args.verifier_timeout)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("u_mup_lr_mult", 1.0)
        ref = reference_model if args.reference == "initial" else None
        diagnostics = optimize_rollouts(model, optimizer, rollouts, rewards,
                                        group_size=args.group_size, algorithm=args.algorithm,
                                        epochs=args.epochs, clip_low=args.clip_low,
                                        clip_high=clip_high, kl_coef=(args.kl_coef if args.reference != "none" else 0),
                                        grad_clip=args.grad_clip, reference_model=ref,
                                        token_normalizer=args.max_new)
        metrics = {**grouped_metrics(rewards, args.group_size), **diagnostics}
        updates_applied += int(diagnostics["update_applied"] > 0)
        row = {"kind": "train", "step": step + 1, "lr": lr,
               "rollout_tokens": sum(len(r.response_ids) for r in rollouts),
               "elapsed_seconds": time.time() - started, **metrics}
        if args.log_samples:
            row["samples"] = [{"task_id": r.task.id, "reward": float(rewards[i]),
                               "response": r.response[:500], "verifier": verifier_details[i]}
                              for i, r in enumerate(rollouts[:args.log_samples])]
        emit(row)
        print(f"[{step + 1}] reward={metrics['reward']:.3f} pass@k={metrics['pass_at_k']:.3f} "
              f"loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.4g}", flush=True)

        if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            last_eval, _ = evaluate_policy(model, vocab, fixed_eval, group_size=args.eval_group_size,
                                           max_new=args.max_new, temperature=args.eval_temperature,
                                           top_p=args.top_p, top_k=args.top_k, stop_token=args.stop_token,
                                           device=args.device, seed=args.seed + 9_000_001,
                                           external_command=external_command,
                                           verifier_timeout=args.verifier_timeout)
            last_eval_step = step + 1
            emit({"kind": "eval", "step": step + 1, "split": "heldout",
                  "phase": "candidate", **last_eval})
            print(f"  heldout reward={last_eval['reward']:.3f} pass@k={last_eval['pass_at_k']:.3f}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            manifest["updates_applied"] = updates_applied
            save_checkpoint(checkpoint_path, model, optimizer, source_blob, step=step + 1,
                            manifest=manifest, rng=rng)

    if last_eval_step != args.steps:
        last_eval, _ = evaluate_policy(model, vocab, fixed_eval, group_size=args.eval_group_size,
                                       max_new=args.max_new, temperature=args.eval_temperature,
                                       top_p=args.top_p, top_k=args.top_k, stop_token=args.stop_token,
                                       device=args.device, seed=args.seed + 9_000_001,
                                       external_command=external_command,
                                       verifier_timeout=args.verifier_timeout)
        emit({"kind": "eval", "step": args.steps, "split": "heldout",
              "phase": "candidate", **last_eval})
    manifest["updates_applied"] = updates_applied
    save_checkpoint(checkpoint_path, model, optimizer, source_blob, step=args.steps,
                    manifest=manifest, rng=rng)
    promotion = promotion_decision(
        baseline_eval, last_eval, minimum_delta=args.min_heldout_delta,
        updates_applied=updates_applied, candidate_checkpoint=str(checkpoint_path.resolve()),
        rollback_checkpoint=str(Path(source_path).resolve()))
    result = {"schema": RESULT_SCHEMA, "status": "complete", "steps": args.steps,
              "checkpoint": str(checkpoint_path.resolve()), "checkpoint_parent_sha256": parent_hash,
              "task_sha256": task_hash, "baseline_heldout": baseline_eval, "heldout": last_eval,
              "promotion": promotion,
              "elapsed_seconds": time.time() - started}
    _atomic_json(out_dir / "result.json", result)
    log.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped RLVR training for RWKV-Lab checkpoints")
    ap.add_argument("--ckpt", required=True, help="self-describing rwkv_pretrain checkpoint")
    ap.add_argument("--resume", default="", help="RLVR checkpoint to resume")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="", help="Adamaton-compatible task JSONL; default generates arithmetic")
    ap.add_argument("--algorithm", choices=["gspo", "dr_grpo", "dapo"], default="gspo")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 is strictly on-policy; other values deliberately temper the behavior policy")
    ap.add_argument("--eval-temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="1.0 is strictly on-policy; truncation deliberately changes the behavior policy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--stop-token", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit", "paged-adamw8bit"], default="adamw")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--clip-low", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=-1.0, help="-1 = 0.28 for DAPO, else 0.2")
    ap.add_argument("--kl-coef", type=float, default=0.01)
    ap.add_argument("--reference", choices=["initial", "rollout", "none"], default="initial")
    ap.add_argument("--reference-ckpt", default="")
    ap.add_argument("--train-tasks", type=int, default=4096)
    ap.add_argument("--eval-tasks", type=int, default=256)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-prompts", type=int, default=32)
    ap.add_argument("--eval-group-size", type=int, default=4)
    ap.add_argument("--min-heldout-delta", type=float, default=0.01,
                    help="absolute held-out reward gain required for promotion eligibility")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--verifier-command", default="",
                    help="trusted argv string for external/code verification; never run through a shell")
    ap.add_argument("--verifier-timeout", type=float, default=10.0)
    ap.add_argument("--log-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-ema", action="store_true")
    from rwkv_lab.generate import VOCAB
    ap.add_argument("--vocab", default=VOCAB)
    args = ap.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / "result.json", {"schema": RESULT_SCHEMA, "status": "failed",
                                           "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
