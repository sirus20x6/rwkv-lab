"""End-to-end reinforcement learning with verifiable rewards.

Primary algorithm references:

* Dr.GRPO, arXiv:2503.20783, https://arxiv.org/abs/2503.20783.
* DAPO, arXiv:2503.14476, https://arxiv.org/abs/2503.14476.
* GSPO, arXiv:2507.18071, https://arxiv.org/abs/2507.18071.
* Absolute Zero, arXiv:2505.03335, https://arxiv.org/abs/2505.03335.
* DeepSeek-R1, arXiv:2501.12948, https://arxiv.org/abs/2501.12948 —
  cold-start supervised data before sparse-reward RL.
* RWKV-7, arXiv:2503.14456, https://arxiv.org/abs/2503.14456 —
  constant-size recurrent state used by the native rollout engine.
* Efron (1979), https://doi.org/10.1214/aos/1176344552 — paired
  bootstrap confidence intervals used by the independent promotion gate.

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
from collections import defaultdict
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
from rwkv_lab.rlvr_evaluation import (audit_task_splits, curriculum_pool,
                                      promotion_gates, reward_diversity,
                                      stratified_tasks, task_reward_summary)


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
            metadata={"family": f"arithmetic/{op1}", "difficulty": difficulty},
        ))
    return tasks


def staged_arithmetic_curriculum(count: int, *, seed: int, split: str,
                                 difficulties: Sequence[int],
                                 exclude_prompts: Sequence[str] = (),
                                 unique: bool = False) -> list[RLVRTask]:
    """Build a balanced deterministic pool across curriculum difficulties."""
    stages = sorted(set(int(value) for value in difficulties)) or [1]
    tasks, seen = [], set()
    blocked = {" ".join(value.split()).casefold() for value in exclude_prompts}
    for index, difficulty in enumerate(stages):
        target = count // len(stages) + int(index < count % len(stages))
        accepted, attempt = 0, 0
        while accepted < target:
            needed = target - accepted
            candidates = arithmetic_curriculum(
                max(needed * 4, 16),
                seed=seed + index * 10_000_019 + attempt * 1_000_003,
                split=split, difficulty=difficulty)
            for task in candidates:
                normalized = " ".join(task.prompt.split()).casefold()
                if normalized in blocked or (unique and normalized in seen):
                    continue
                if unique:
                    seen.add(normalized)
                tasks.append(task)
                accepted += 1
                if accepted == target:
                    break
            attempt += 1
            if attempt > 100:
                raise ValueError("could not generate a unique arithmetic curriculum")
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


def _sample_token(logits: torch.Tensor, *, temperature: float, top_p: float,
                  top_k: int, generator: torch.Generator) -> int:
    logits = logits.float().clone()
    logits[0] = -float("inf")
    if temperature <= 0:
        return int(logits.argmax())
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
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))


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
        logits = _logits(model(tokens))[0, -1]
        nxt = _sample_token(logits, temperature=temperature, top_p=top_p,
                            top_k=top_k, generator=generator)
        response.append(nxt)
        if nxt == stop_token:
            break
        tokens = torch.cat((tokens, tokens.new_tensor([[nxt]])), dim=1)
    return response


def select_rollout_engine(model, requested: str = "auto") -> tuple[str, str]:
    """Select exact recurrent decoding or the semantics-preserving fallback."""
    if requested not in ("auto", "recurrent", "batched"):
        raise ValueError("rollout engine must be auto, recurrent, or batched")
    reason = "model does not expose a recurrent state API"
    if hasattr(model, "recurrent_incompatibility"):
        reason = model.recurrent_incompatibility() or ""
    recurrent = hasattr(model, "forward_recurrent") and not reason
    if requested == "recurrent" and not recurrent:
        raise ValueError(f"recurrent rollout engine unavailable: {reason}")
    if requested == "auto":
        return (("recurrent", "native RWKV constant-size state") if recurrent
                else ("batched", reason))
    return requested, reason if requested == "batched" else "native RWKV constant-size state"


@torch.no_grad()
def sample_response_group(model, prompt_ids: Sequence[int], *, count: int, max_new: int,
                          temperature: float, top_p: float, top_k: int,
                          stop_token: int, device: str, seeds: Sequence[int],
                          engine: str = "auto") -> tuple[list[list[int]], dict[str, Any]]:
    """Decode an equal-prefix rollout group in one model batch.

    Native RWKV checkpoints carry constant-size matrix/shift state across token
    steps (RWKV-7, https://arxiv.org/abs/2503.14456). Non-causal experimental
    levers use batched full-prefix recomputation to preserve exact semantics.
    """
    if not prompt_ids or count < 1 or len(seeds) != count:
        raise ValueError("group sampling needs a prompt and one seed per response")
    chosen, reason = select_rollout_engine(model, engine)
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device).expand(count, -1)
    responses: list[list[int]] = [[] for _ in range(count)]
    finished = [False] * count
    tokens, state = prompt, None
    started = time.perf_counter()
    if chosen == "recurrent":
        logits, state = model.forward_recurrent(tokens)
        next_logits = logits[:, -1]
    for step in range(max(1, int(max_new))):
        if chosen == "batched":
            next_logits = _logits(model(tokens))[:, -1]
        next_tokens = []
        for row in range(count):
            if finished[row]:
                next_tokens.append(stop_token)
                continue
            nxt = _sample_token(next_logits[row], temperature=temperature, top_p=top_p,
                                top_k=top_k, generator=generators[row])
            responses[row].append(nxt)
            next_tokens.append(nxt)
            if nxt == stop_token:
                finished[row] = True
        if all(finished) or step + 1 == max_new:
            break
        token_column = tokens.new_tensor(next_tokens).unsqueeze(1)
        if chosen == "recurrent":
            logits, state = model.forward_recurrent(token_column, state)
            next_logits = logits[:, -1]
        else:
            tokens = torch.cat((tokens, token_column), dim=1)
    elapsed = time.perf_counter() - started
    generated = sum(len(row) for row in responses)
    return responses, {"engine": chosen, "fallback_reason": reason, "seconds": elapsed,
                       "tokens": generated,
                       "tokens_per_second": generated / max(elapsed, 1e-9)}


def generate_rollouts(model, tokenizer, tasks: Sequence[RLVRTask], *, group_size: int,
                      max_new: int, temperature: float, top_p: float, top_k: int,
                      stop_token: int, device: str, seed: int, engine: str = "auto",
                      return_stats: bool = False):
    model.eval()
    out = []
    stats = {"seconds": 0.0, "tokens": 0, "engine": "", "fallback_reason": ""}
    for group, task in enumerate(tasks):
        prompt = f"User: {task.prompt}\n\nAssistant:"
        prompt_ids = tokenizer.encode(prompt)
        seeds = [seed + group * 100_003 + k for k in range(int(group_size))]
        responses, group_stats = sample_response_group(
            model, prompt_ids, count=int(group_size), max_new=max_new,
            temperature=temperature, top_p=top_p, top_k=top_k,
            stop_token=stop_token, device=device, seeds=seeds, engine=engine)
        stats["seconds"] += group_stats["seconds"]
        stats["tokens"] += group_stats["tokens"]
        stats["engine"] = group_stats["engine"]
        stats["fallback_reason"] = group_stats["fallback_reason"]
        for k, response_ids in enumerate(responses):
            visible = response_ids[:-1] if response_ids and response_ids[-1] == stop_token else response_ids
            out.append(Rollout(f"{task.id}:{k}", task, list(prompt_ids), response_ids,
                               tokenizer.decode(visible)))
    stats["tokens_per_second"] = stats["tokens"] / max(stats["seconds"], 1e-9)
    return (out, stats) if return_stats else out


def response_log_probs(model, rollouts: Sequence[Rollout]) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably score only response tokens, padding to ``[rollouts,max_response]``."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    device = next(model.parameters()).device
    max_response = max(len(r.response_ids) for r in rollouts)
    if max_response <= 0:
        raise ValueError("rollouts must contain at least one policy token")
    rows, masks = [None] * len(rollouts), [None] * len(rollouts)
    buckets = defaultdict(list)
    for index, rollout in enumerate(rollouts):
        buckets[len(rollout.prompt_ids) + len(rollout.response_ids)].append((index, rollout))
    for bucket in buckets.values():
        ids = torch.tensor([r.prompt_ids + r.response_ids for _, r in bucket],
                           dtype=torch.long, device=device)
        logits, targets = _logits(model(ids[:, :-1])), ids[:, 1:]
        for row_index, (output_index, rollout) in enumerate(bucket):
            start, length = len(rollout.prompt_ids) - 1, len(rollout.response_ids)
            logp = token_log_probs(logits[row_index, start:start + length],
                                   targets[row_index, start:start + length])
            pad = max_response - length
            rows[output_index] = F.pad(logp, (0, pad))
            masks[output_index] = F.pad(torch.ones_like(logp), (0, pad))
    assert all(row is not None for row in rows) and all(mask is not None for mask in masks)
    return torch.stack(rows), torch.stack(masks)


def grouped_metrics(rewards: torch.Tensor, group_size: int) -> dict[str, float]:
    groups = rewards.view(-1, int(group_size))
    return {"reward": float(rewards.mean()), "pass_at_1": float(groups[:, 0].mean()),
            "pass_at_k": float((groups.max(-1).values > 0).float().mean()),
            "reward_std": float(rewards.std(unbiased=False))}


def supervised_answer(task: RLVRTask) -> str | None:
    """Return trusted cold-start supervision without executing model output."""
    metadata = task.metadata or {}
    if metadata.get("sft_answer") is not None:
        return str(metadata["sft_answer"])
    kind, expected = task.verifier.get("kind"), task.verifier.get("expected")
    if kind in ("numeric", "expression") and expected is not None:
        value = float(expected)
        return str(int(value)) if value.is_integer() else str(value)
    if kind == "exact":
        return str(expected[0] if isinstance(expected, list) and expected else expected)
    return None


def supervised_warm_start(model, tokenizer, tasks: Sequence[RLVRTask], optimizer, *,
                          steps: int, batch_size: int, learning_rate: float,
                          grad_clip: float, stop_token: int, device: str,
                          seed: int) -> dict[str, float]:
    """Short answer-only SFT stage before sparse-reward RLVR.

    DeepSeek-R1 uses cold-start supervised data before its main RL stage
    (https://arxiv.org/abs/2501.12948). Here targets come only from trusted
    deterministic verifier specs or explicit ``metadata.sft_answer`` fields.
    """
    eligible = [(task, supervised_answer(task)) for task in tasks]
    eligible = [(task, answer) for task, answer in eligible if answer is not None]
    if steps <= 0 or not eligible:
        return {"updates": 0, "tokens": 0, "mean_loss": 0.0, "seconds": 0.0}
    rng, cache = random.Random(seed), {}
    old_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group.get("u_mup_lr_mult", 1.0)
    losses_seen, tokens_seen, started = [], 0, time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for task, answer in rng.choices(eligible, k=max(1, batch_size)):
            key = (task.id, answer)
            if key not in cache:
                prompt = tokenizer.encode(f"User: {task.prompt}\n\nAssistant:")
                response = tokenizer.encode(" " + answer) + [stop_token]
                cache[key] = (prompt, response)
            prompt_ids, response_ids = cache[key]
            ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long, device=device)
            logits = _logits(model(ids[:-1].unsqueeze(0)))[0]
            start = len(prompt_ids) - 1
            target = ids[1:]
            losses.append(F.cross_entropy(
                logits[start:start + len(response_ids)].float(),
                target[start:start + len(response_ids)]))
            tokens_seen += len(response_ids)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses_seen.append(float(loss.detach()))
    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr
    return {"updates": steps, "tokens": tokens_seen,
            "mean_loss": sum(losses_seen) / len(losses_seen),
            "seconds": time.perf_counter() - started}


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
                    external_command: Sequence[str], verifier_timeout: float,
                    engine: str = "auto") -> tuple[dict[str, Any], list[Rollout]]:
    rollouts, generation = generate_rollouts(
        model, tokenizer, tasks, group_size=group_size, max_new=max_new,
        temperature=temperature, top_p=top_p, top_k=top_k,
        stop_token=stop_token, device=device, seed=seed, engine=engine,
        return_stats=True)
    rewards, _ = verify_rollouts(rollouts, external_command=external_command, timeout=verifier_timeout)
    summary = task_reward_summary(tasks, rewards.tolist(), group_size)
    return {**grouped_metrics(rewards, group_size), **summary,
            "generation": generation}, rollouts


def run(args) -> dict[str, Any]:
    from rwkv_lab.generate import WorldVocab, build_from_ckpt
    from rwkv_lab.rwkv_pretrain import build_optimizer

    if args.group_size < 2 or args.prompts_per_step < 1 or args.max_new < 1:
        raise ValueError("group-size must be >=2; prompts-per-step and max-new must be positive")
    if args.eval_group_size < 1 or args.eval_prompts < 1:
        raise ValueError("eval-group-size and eval-prompts must be positive")
    if args.sft_steps < 0 or args.sft_batch_size < 1 or args.preflight_prompts < 0:
        raise ValueError("SFT/preflight counts must be non-negative and SFT batch size positive")
    if not 0 < args.confidence < 1 or args.bootstrap_samples < 0:
        raise ValueError("confidence must be in (0,1) and bootstrap samples non-negative")
    if not 0 <= args.min_preflight_reward < args.max_preflight_reward <= 1:
        raise ValueError("preflight reward thresholds must satisfy 0 <= min < max <= 1")
    if args.max_family_regression < 0:
        raise ValueError("max-family-regression must be non-negative")
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

    curriculum_stages = [int(value) for value in args.curriculum_stages.split(",") if value.strip()]
    if args.tasks:
        supplied_tasks, task_hash = load_task_jsonl(args.tasks)
        if args.heldout_tasks:
            heldout_tasks, heldout_hash = load_task_jsonl(args.heldout_tasks)
            train_tasks = [task for task in supplied_tasks if task.split == "train"]
            eval_tasks = [task for task in heldout_tasks if task.split == "eval"]
            if len(train_tasks) != len(supplied_tasks) or len(eval_tasks) != len(heldout_tasks):
                raise ValueError("--tasks must be train-only and --heldout-tasks eval-only when separated")
            task_hash = hashlib.sha256(f"{task_hash}:{heldout_hash}".encode()).hexdigest()
            all_tasks = train_tasks + eval_tasks
        else:
            all_tasks = supplied_tasks
            train_tasks, eval_tasks = split_task_pool(all_tasks)
    else:
        stages = curriculum_stages or [args.difficulty]
        heldout_hash = ""
        if args.heldout_tasks:
            eval_tasks, heldout_hash = load_task_jsonl(args.heldout_tasks)
            if any(task.split != "eval" for task in eval_tasks):
                raise ValueError("--heldout-tasks must contain eval tasks only")
        else:
            eval_tasks = staged_arithmetic_curriculum(
                args.eval_tasks, seed=args.seed + 1_000_003, split="eval",
                difficulties=stages, unique=True)
        train_tasks = staged_arithmetic_curriculum(
            args.train_tasks, seed=args.seed + 101, split="train", difficulties=stages,
            exclude_prompts=[task.prompt for task in eval_tasks])
        all_tasks = train_tasks + eval_tasks
        task_hash = hashlib.sha256(
            ("\n".join(json.dumps(asdict(t), sort_keys=True) for t in all_tasks) +
             (":" + heldout_hash if heldout_hash else "")).encode()).hexdigest()
    if not train_tasks or not eval_tasks:
        raise ValueError("RLVR requires non-empty train and held-out task pools")
    split_audit = audit_task_splits(train_tasks, eval_tasks)
    if not split_audit["passed"]:
        raise ValueError(f"train/held-out contamination audit failed: {split_audit}")
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
                "rollout_engine": args.rollout_engine, "curriculum_stages": curriculum_stages,
                "split_audit": split_audit,
                "created_ts": time.time()}
    rng = random.Random(args.seed)
    if args.resume and source_blob.get("rlvr_python_rng"):
        rng.setstate(source_blob["rlvr_python_rng"])
    clip_high = (0.28 if args.algorithm == "dapo" else 0.2) if args.clip_high < 0 else args.clip_high
    log = open(out_dir / "train.jsonl", "a" if args.resume else "w", buffering=1)
    def emit(row):
        return log.write(json.dumps(row, sort_keys=True) + "\n")
    fixed_eval = stratified_tasks(eval_tasks, min(args.eval_prompts, len(eval_tasks)),
                                  seed=args.seed + 8_000_003)
    final_eval_reserve = len(fixed_eval) * args.eval_group_size * args.max_new
    if args.max_rollout_tokens > 0 and final_eval_reserve > args.max_rollout_tokens:
        raise ValueError("rollout budget is smaller than one fixed held-out evaluation")
    stored_baseline = (source_blob.get("rlvr") or {}).get("baseline_heldout") if args.resume else None
    if stored_baseline and stored_baseline.get("task_rewards"):
        baseline_eval = dict(stored_baseline)
    elif args.resume:
        raise ValueError("resumed checkpoint predates paired held-out evidence; start a fresh RLVR run")
    else:
        baseline_eval, _ = evaluate_policy(
            model, vocab, fixed_eval, group_size=args.eval_group_size,
            max_new=args.max_new, temperature=args.eval_temperature, top_p=args.top_p,
            top_k=args.top_k, stop_token=args.stop_token, device=args.device,
            seed=args.seed + 9_000_001, external_command=external_command,
            verifier_timeout=args.verifier_timeout, engine=args.rollout_engine)
    manifest["baseline_heldout"] = baseline_eval
    _atomic_json(out_dir / "manifest.json", manifest)
    emit({"kind": "eval", "step": start_step, "split": "heldout",
          "phase": "baseline", **baseline_eval})
    prior_manifest = source_blob.get("rlvr") or {}
    prior_elapsed = float(prior_manifest.get("elapsed_seconds", 0)) if args.resume else 0.0
    prior_rollout_tokens = int(prior_manifest.get("total_rollout_tokens", 0)) if args.resume else 0
    if (args.max_rollout_tokens > 0 and
            prior_rollout_tokens + final_eval_reserve > args.max_rollout_tokens):
        raise ValueError("remaining rollout budget cannot reserve one held-out evaluation")
    training_started = time.time()
    if args.resume:
        sft = dict(prior_manifest.get("sft") or
                   {"updates": 0, "tokens": 0, "mean_loss": 0.0, "seconds": 0.0})
    else:
        sft = supervised_warm_start(
            model, vocab, train_tasks, optimizer, steps=args.sft_steps,
            batch_size=args.sft_batch_size, learning_rate=args.sft_lr,
            grad_clip=args.grad_clip, stop_token=args.stop_token,
            device=args.device, seed=args.seed + 7_000_001)
    sft_ran = not args.resume and int(sft["updates"]) > 0
    emit({"kind": "sft", "step": start_step, **sft})

    total_rollout_tokens = prior_rollout_tokens
    preflight = {"passed": True, "disabled": True}
    preflight_budget_exhausted = False
    if not args.resume and args.preflight_prompts > 0:
        pool = curriculum_pool(train_tasks, step=0, total_steps=max(args.steps, 1),
                               stages=curriculum_stages)
        preflight_tasks = stratified_tasks(
            pool, min(args.preflight_prompts, len(pool)), seed=args.seed + 6_000_007)
        preflight_reserve = len(preflight_tasks) * args.group_size * args.max_new
        if (args.max_rollout_tokens > 0 and
                total_rollout_tokens + preflight_reserve + final_eval_reserve >
                args.max_rollout_tokens):
            preflight = {"passed": False, "budget_exhausted": True,
                         "reason": "preflight plus held-out reserve exceeds rollout budget"}
            preflight_budget_exhausted = True
        else:
            preflight_rollouts, preflight_generation = generate_rollouts(
                model, vocab, preflight_tasks, group_size=args.group_size,
                max_new=args.max_new, temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, stop_token=args.stop_token, device=args.device,
                seed=args.seed + 6_000_011, engine=args.rollout_engine, return_stats=True)
            preflight_rewards, _ = verify_rollouts(
                preflight_rollouts, external_command=external_command,
                timeout=args.verifier_timeout)
            total_rollout_tokens += int(preflight_generation["tokens"])
            preflight = reward_diversity(
                preflight_rewards.tolist(), args.group_size,
                minimum_rate=args.min_preflight_reward,
                maximum_rate=args.max_preflight_reward,
                minimum_active_groups=args.min_preflight_active_groups)
            preflight["generation"] = preflight_generation
    manifest["sft"] = sft
    manifest["preflight"] = preflight
    emit({"kind": "preflight", "step": start_step, **preflight})
    _atomic_json(out_dir / "manifest.json", manifest)

    last_eval: dict[str, Any] = dict(baseline_eval)
    last_eval_step = start_step
    updates_applied = int((source_blob.get("rlvr") or {}).get("updates_applied", 0)) if args.resume else 0
    checkpoint_path = out_dir / "rlvr.pt"
    steps_completed = start_step
    training_status = ("rollout_budget_exhausted" if preflight_budget_exhausted else
                       "running" if preflight["passed"] else "preflight_rejected")

    for step in range(start_step, args.steps if preflight["passed"] else start_step):
        elapsed = prior_elapsed + time.time() - training_started
        estimate = args.prompts_per_step * args.group_size * args.max_new
        if (args.max_rollout_tokens > 0 and
                total_rollout_tokens + estimate + final_eval_reserve > args.max_rollout_tokens):
            training_status = "rollout_budget_exhausted"
            break
        if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
            training_status = "time_budget_exhausted"
            break
        pool = curriculum_pool(train_tasks, step=step, total_steps=args.steps,
                               stages=curriculum_stages)
        chosen = rng.sample(pool, min(args.prompts_per_step, len(pool)))
        rollouts, generation = generate_rollouts(
            model, vocab, chosen, group_size=args.group_size,
            max_new=args.max_new, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, stop_token=args.stop_token,
            device=args.device, seed=args.seed + step * 1_000_003,
            engine=args.rollout_engine, return_stats=True)
        total_rollout_tokens += int(generation["tokens"])
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
        steps_completed = step + 1
        row = {"kind": "train", "step": step + 1, "lr": lr,
               "rollout_tokens": sum(len(r.response_ids) for r in rollouts),
               "total_rollout_tokens": total_rollout_tokens,
               "generation": generation,
               "curriculum_pool": len(pool),
               "elapsed_seconds": time.time() - training_started, **metrics}
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
                                           verifier_timeout=args.verifier_timeout,
                                           engine=args.rollout_engine)
            total_rollout_tokens += int(last_eval["generation"]["tokens"])
            last_eval_step = step + 1
            emit({"kind": "eval", "step": step + 1, "split": "heldout",
                  "phase": "candidate", **last_eval})
            print(f"  heldout reward={last_eval['reward']:.3f} pass@k={last_eval['pass_at_k']:.3f}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            manifest["updates_applied"] = updates_applied
            manifest["total_rollout_tokens"] = total_rollout_tokens
            manifest["elapsed_seconds"] = prior_elapsed + time.time() - training_started
            save_checkpoint(checkpoint_path, model, optimizer, source_blob, step=step + 1,
                            manifest=manifest, rng=rng)

    if training_status == "running":
        training_status = "complete"
    if (last_eval_step != steps_completed or
            (sft_ran and steps_completed == start_step) or not preflight["passed"]):
        last_eval, _ = evaluate_policy(model, vocab, fixed_eval, group_size=args.eval_group_size,
                                       max_new=args.max_new, temperature=args.eval_temperature,
                                       top_p=args.top_p, top_k=args.top_k, stop_token=args.stop_token,
                                       device=args.device, seed=args.seed + 9_000_001,
                                       external_command=external_command,
                                       verifier_timeout=args.verifier_timeout,
                                       engine=args.rollout_engine)
        total_rollout_tokens += int(last_eval["generation"]["tokens"])
        emit({"kind": "eval", "step": steps_completed, "split": "heldout",
              "phase": "candidate", **last_eval})
    manifest["updates_applied"] = updates_applied
    manifest["sft_updates"] = int(sft["updates"])
    manifest["training_status"] = training_status
    manifest["total_rollout_tokens"] = total_rollout_tokens
    elapsed_seconds = prior_elapsed + time.time() - training_started
    manifest["elapsed_seconds"] = elapsed_seconds
    save_checkpoint(checkpoint_path, model, optimizer, source_blob, step=steps_completed,
                    manifest=manifest, rng=rng)
    promotion = promotion_gates(
        baseline_eval, last_eval, minimum_delta=args.min_heldout_delta,
        updates_applied=updates_applied + int(sft["updates"]),
        maximum_family_regression=args.max_family_regression,
        require_confidence=args.require_confidence,
        bootstrap_samples=args.bootstrap_samples, confidence=args.confidence,
        seed=args.seed + 5_000_011, split_audit=split_audit,
        rollout_tokens=total_rollout_tokens, elapsed_seconds=elapsed_seconds,
        maximum_rollout_tokens=args.max_rollout_tokens,
        maximum_train_seconds=args.max_train_seconds)
    promotion.update({
        "heldout_delta": float(last_eval["reward"] - baseline_eval["reward"]),
        "minimum_delta": args.min_heldout_delta,
        "updates_applied": updates_applied,
        "sft_updates": int(sft["updates"]),
        "candidate_checkpoint": str(checkpoint_path.resolve()),
        "rollback_checkpoint": str(Path(source_path).resolve()),
    })
    failed_gates = [name for name, passed in promotion["gates"].items() if not passed]
    promotion["reason"] = ("all independent promotion gates passed" if promotion["eligible"]
                           else "failed gates: " + ", ".join(failed_gates))
    result = {"schema": RESULT_SCHEMA, "status": "complete", "steps": args.steps,
              "steps_completed": steps_completed, "training_status": training_status,
              "checkpoint": str(checkpoint_path.resolve()), "checkpoint_parent_sha256": parent_hash,
              "task_sha256": task_hash, "baseline_heldout": baseline_eval, "heldout": last_eval,
              "split_audit": split_audit, "sft": sft, "preflight": preflight,
              "total_rollout_tokens": total_rollout_tokens, "promotion": promotion,
              "elapsed_seconds": elapsed_seconds}
    _atomic_json(out_dir / "result.json", result)
    log.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped RLVR training for RWKV-Lab checkpoints")
    ap.add_argument("--ckpt", required=True, help="self-describing rwkv_pretrain checkpoint")
    ap.add_argument("--resume", default="", help="RLVR checkpoint to resume")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="", help="Adamaton-compatible task JSONL; default generates arithmetic")
    ap.add_argument("--heldout-tasks", default="",
                    help="optional separate eval-only JSONL hidden from the proposal process")
    ap.add_argument("--algorithm", choices=["gspo", "dr_grpo", "dapo"], default="gspo")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--rollout-engine", choices=["auto", "recurrent", "batched"], default="auto")
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
    ap.add_argument("--curriculum-stages", default="",
                    help="comma-separated arithmetic/metadata difficulty stages, e.g. 1,2,3")
    ap.add_argument("--sft-steps", type=int, default=0,
                    help="trusted-answer cold-start updates before RL; 0 disables")
    ap.add_argument("--sft-batch-size", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--preflight-prompts", type=int, default=0,
                    help="sample this many training tasks and require reward diversity before RL")
    ap.add_argument("--min-preflight-reward", type=float, default=0.01)
    ap.add_argument("--max-preflight-reward", type=float, default=0.99)
    ap.add_argument("--min-preflight-active-groups", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-prompts", type=int, default=32)
    ap.add_argument("--eval-group-size", type=int, default=4)
    ap.add_argument("--min-heldout-delta", type=float, default=0.01,
                    help="absolute held-out reward gain required for promotion eligibility")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--require-confidence", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-family-regression", type=float, default=0.0)
    ap.add_argument("--max-rollout-tokens", type=int, default=0,
                    help="hard candidate rollout budget; 0 is unlimited")
    ap.add_argument("--max-train-seconds", type=float, default=0.0,
                    help="candidate wall-clock budget; 0 is unlimited")
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
