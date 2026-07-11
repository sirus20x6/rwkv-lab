"""Native-RWKV SFT, preference, feedback, and outcome-reward training.

This is the executable bridge between ``posttrain_data``, named LoRA adapters, and the losses in
``preference``.  It intentionally accepts only trusted, self-describing rwkv-lab checkpoints and
never executes dataset content.  Adamaton remains responsible for isolated tool execution and
verifier fleets.

Paper references used by the implementation are kept beside their primitives: LoRA/QLoRA in
``adapters.py`` / ``quantization.py``, DPO/KTO/ORPO/SimPO in ``preference.py``, and PRM step
supervision in Lightman et al., https://arxiv.org/abs/2305.20050.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import copy
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from rwkv_lab.adapters import (AdapterConfig, active_adapters, adapter_parameters,
                               inject_lora, save_adapter)
from rwkv_lab.posttrain_data import (IGNORE_INDEX, TokenizedExample, TokenizedVariant,
                                     DEFAULT_TEMPLATE, cache_tokenized, dataset_manifest, load_jsonl,
                                     load_template, pack_variants, render, tokenize)
from rwkv_lab.preference import (dpo_loss, kto_loss, orpo_loss, reward_model_loss,
                                 OutcomeRewardHead, ProcessRewardHead, binary_calibration,
                                 adversarial_reward_audit, process_reward_loss, sequence_logps,
                                 simpo_loss)
from rwkv_lab.quantization import (model_storage_bytes, qualify_accelerated_nf4,
                                   qualify_linear_qlora, quantize_model_nf4)


RESULT_SCHEMA = "rwkv-lab.posttrain-result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_tokens(batch: list[TokenizedExample], objective: str) -> int:
    names = (("chosen", "rejected") if objective in ("dpo", "orpo", "simpo", "reward")
             else ("response",) if objective == "kto" else ("steps",) if objective == "prm"
             else ("sft",))
    return sum(len(row.variants[name].input_ids) for row in batch for name in names)


def _pack_groups(rows: list[TokenizedExample], names: tuple[str, ...], max_length: int
                 ) -> list[list[TokenizedExample]]:
    """Best-fit groups constrained independently for every objective variant."""
    ordered = sorted(rows, key=lambda row: (-max(len(row.variants[name].input_ids)
                                                    for name in names), row.id))
    groups: list[list[TokenizedExample]] = []
    sizes: list[dict[str, int]] = []
    for row in ordered:
        candidates = []
        for index, totals in enumerate(sizes):
            if all(totals[name] + len(row.variants[name].input_ids) <= max_length for name in names):
                waste = sum(max_length - totals[name] - len(row.variants[name].input_ids)
                            for name in names)
                candidates.append((waste, index))
        target = min(candidates)[1] if candidates else len(groups)
        if target == len(groups):
            groups.append([])
            sizes.append({name: 0 for name in names})
        groups[target].append(row)
        for name in names:
            sizes[target][name] += len(row.variants[name].input_ids)
    return groups


def _join_variants(rows: list[TokenizedExample], name: str) -> TokenizedVariant:
    ids: list[int] = []
    labels: list[int] = []
    roles: list[str] = []
    starts: list[int] = []
    step_positions: list[int] = []
    step_labels: list[bool] = []
    truncated = 0
    for row in rows:
        value = row.variants[name]
        offset = len(ids)
        starts.append(offset)
        ids.extend(value.input_ids)
        labels.extend(value.labels)
        # A causal sequence has no logit preceding its first token. Mask the boundary token just
        # as separate padded rows do; otherwise the preceding segment would supply its predictor.
        labels[offset] = IGNORE_INDEX
        roles.extend(value.roles)
        step_positions.extend(offset + position for position in value.step_positions)
        step_labels.extend(value.step_labels)
        truncated += value.truncated
    return TokenizedVariant(tuple(ids), tuple(labels), tuple(roles), truncated,
                            tuple(step_positions), tuple(step_labels), tuple(starts))


def _packed_forward(model: nn.Module, variant: TokenizedVariant, device: str, *,
                    return_hidden: bool = False):
    ids = torch.tensor(variant.input_ids, dtype=torch.long, device=device)[None]
    reset = torch.zeros_like(ids, dtype=torch.bool)
    reset[0, list(variant.sequence_starts)] = True
    return model(ids, reset_mask=reset, return_hidden=return_hidden)


def _segmented_logps(logits: torch.Tensor, variant: TokenizedVariant, *,
                     average: bool = False) -> torch.Tensor:
    labels = torch.tensor(variant.labels, dtype=torch.long, device=logits.device)[None]
    target = labels[:, 1:]
    mask = target != IGNORE_INDEX
    safe = target.masked_fill(~mask, 0)
    values = logits[:, :-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)[0]
    results = []
    starts = variant.sequence_starts
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(variant.input_ids)
        # labels at token position p are predicted by logits p-1.
        selected = mask[0, start:max(0, end - 1)]
        scores = values[start:max(0, end - 1)][selected]
        if not scores.numel():
            raise ValueError("every packed segment needs at least one supervised target")
        results.append(scores.mean() if average else scores.sum())
    return torch.stack(results)


def _packed_outcome_rewards(head: OutcomeRewardHead, hidden: torch.Tensor,
                            variant: TokenizedVariant) -> torch.Tensor:
    labels = torch.tensor(variant.labels, device=hidden.device)
    positions = []
    for index, start in enumerate(variant.sequence_starts):
        end = (variant.sequence_starts[index + 1] if index + 1 < len(variant.sequence_starts)
               else len(variant.input_ids))
        supervised = torch.nonzero(labels[start:end] != IGNORE_INDEX, as_tuple=False).flatten()
        if not supervised.numel():
            raise ValueError("packed reward segment has no response token")
        positions.append(start + int(supervised[-1]))
    chosen = hidden[0, torch.tensor(positions, device=hidden.device)]
    return head.proj(chosen).squeeze(-1)


@lru_cache(maxsize=8192)
def _variant_cpu_tensors(variant: TokenizedVariant) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorize immutable token records once; batching performs one device copy."""
    return (torch.tensor(variant.input_ids, dtype=torch.long),
            torch.tensor(variant.labels, dtype=torch.long))


def _collate(variants: list[TokenizedVariant], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(v.input_ids) for v in variants)
    ids = torch.zeros(len(variants), width, dtype=torch.long)
    labels = torch.full((len(variants), width), IGNORE_INDEX, dtype=torch.long)
    for i, variant in enumerate(variants):
        n = len(variant.input_ids)
        source_ids, source_labels = _variant_cpu_tensors(variant)
        ids[i, :n] = source_ids
        labels[i, :n] = source_labels
    if torch.device(device).type == "cuda":
        ids, labels = ids.pin_memory(), labels.pin_memory()
    return (ids.to(device, non_blocking=True), labels.to(device, non_blocking=True))


def _lm_logps(model: nn.Module, variants: list[TokenizedVariant], device: str, *,
               average: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    ids, labels = _collate(variants, device)
    logits = model(ids)
    return sequence_logps(logits, labels, average=average), labels


def _paired_logps(model: nn.Module, chosen: list[TokenizedVariant], rejected: list[TokenizedVariant],
                  device: str, *, average: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Score preference pairs in one recurrent batch; parity-qualified in posttrain_kernels.py."""
    combined = chosen + rejected
    values, _ = _lm_logps(model, combined, device, average=average)
    return values[:len(chosen)], values[len(chosen):]


def _collate_steps(variants: list[TokenizedVariant], device: str):
    ids, labels = _collate(variants, device)
    width = max(len(value.step_positions) for value in variants)
    positions = torch.zeros(len(variants), width, dtype=torch.long)
    step_labels = torch.zeros(len(variants), width, dtype=torch.float32)
    mask = torch.zeros(len(variants), width, dtype=torch.bool)
    for index, value in enumerate(variants):
        count = len(value.step_positions)
        positions[index, :count] = torch.tensor(value.step_positions)
        step_labels[index, :count] = torch.tensor(value.step_labels, dtype=torch.float32)
        mask[index, :count] = True
    if torch.device(device).type == "cuda":
        positions, step_labels, mask = positions.pin_memory(), step_labels.pin_memory(), mask.pin_memory()
    return (ids, labels, positions.to(device, non_blocking=True),
            step_labels.to(device, non_blocking=True), mask.to(device, non_blocking=True))


def _train_loss(model: nn.Module, reward_head: nn.Module | None, batch: list[TokenizedExample],
                objective: str, adapter_name: str, device: str, beta: float,
                gamma: float, *, reference_cache: dict | None = None,
                collect_metrics: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "sft":
        ids, labels = _collate([row.variants["sft"] for row in batch], device)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return loss, ({"sft_nll": float(loss.detach())} if collect_metrics else {})

    if objective == "kto":
        variants = [row.variants["response"] for row in batch]
        policy, _ = _lm_logps(model, variants, device)
        if reference_cache is None:
            with torch.no_grad(), active_adapters(model, ()):
                reference, _ = _lm_logps(model, variants, device)
        else:
            reference = torch.tensor([reference_cache[row.id][0] for row in batch],
                                     device=device, dtype=policy.dtype)
        desirable = torch.tensor([bool(row.metadata["label"]) for row in batch], device=device)
        losses = kto_loss(policy, reference, desirable, beta=beta)
        return losses.mean(), ({"desirable_frac": float(desirable.float().mean())}
                               if collect_metrics else {})

    if objective == "prm":
        if not isinstance(reward_head, ProcessRewardHead):
            raise ValueError("PRM objective needs a process reward head")
        variants = [row.variants["steps"] for row in batch]
        ids, _, positions, step_labels, step_mask = _collate_steps(variants, device)
        _, hidden = model(ids, return_hidden=True)
        logits = reward_head(hidden, positions)
        loss = process_reward_loss(logits, step_labels, step_mask)
        metrics = (binary_calibration(logits[step_mask], step_labels[step_mask])
                   if collect_metrics else {})
        return loss, {f"prm_{name}": value for name, value in metrics.items()}

    chosen = [row.variants["chosen"] for row in batch]
    rejected = [row.variants["rejected"] for row in batch]
    if objective == "reward":
        if reward_head is None:
            raise ValueError("reward objective needs a reward head")
        ids, labels = _collate(chosen + rejected, device)
        _, hidden = model(ids, return_hidden=True)
        count = len(chosen)
        cr = reward_head(hidden[:count], labels[:count] != IGNORE_INDEX)
        rr = reward_head(hidden[count:], labels[count:] != IGNORE_INDEX)
        losses = reward_model_loss(cr, rr)
        return losses.mean(), ({"reward_margin": float((cr - rr).detach().mean())}
                               if collect_metrics else {})

    average = objective in ("orpo", "simpo")
    policy_chosen, policy_rejected = _paired_logps(model, chosen, rejected, device, average=average)
    if objective == "dpo":
        if reference_cache is None:
            with torch.no_grad(), active_adapters(model, ()):
                reference_chosen, reference_rejected = _paired_logps(model, chosen, rejected, device)
        else:
            reference_chosen = torch.tensor([reference_cache[row.id][0] for row in batch],
                                            device=device, dtype=policy_chosen.dtype)
            reference_rejected = torch.tensor([reference_cache[row.id][1] for row in batch],
                                              device=device, dtype=policy_rejected.dtype)
        result = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                          beta=beta)
    elif objective == "orpo":
        result = orpo_loss(policy_chosen, policy_rejected, -policy_chosen, beta=beta)
    elif objective == "simpo":
        result = simpo_loss(policy_chosen, policy_rejected, beta=beta, gamma=gamma)
    else:
        raise ValueError(f"unsupported objective {objective!r}")
    return result.loss.mean(), ({"preference_margin": float(result.margin.mean())}
                                if collect_metrics else {})


def _packed_train_loss(model: nn.Module, reward_head: nn.Module | None,
                       rows: list[TokenizedExample], objective: str, device: str,
                       beta: float, gamma: float, *, reference_cache: dict | None = None,
                       collect_metrics: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    """Train one reset-isolated multipack without cross-example recurrent state."""
    if objective == "sft":
        variant = _join_variants(rows, "sft")
        logits = _packed_forward(model, variant, device)
        labels = torch.tensor(variant.labels, device=device)[None]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return loss, ({"sft_nll": float(loss.detach()), "packed_examples": len(rows)}
                      if collect_metrics else {})

    if objective == "prm":
        if not isinstance(reward_head, ProcessRewardHead):
            raise ValueError("PRM objective needs a process reward head")
        variant = _join_variants(rows, "steps")
        _, hidden = _packed_forward(model, variant, device, return_hidden=True)
        positions = torch.tensor(variant.step_positions, dtype=torch.long, device=device)[None]
        labels = torch.tensor(variant.step_labels, dtype=torch.float32, device=device)[None]
        logits = reward_head(hidden, positions)
        loss = process_reward_loss(logits, labels)
        metrics = binary_calibration(logits.flatten(), labels.flatten()) if collect_metrics else {}
        return loss, {**{f"prm_{name}": value for name, value in metrics.items()},
                      "packed_examples": len(rows)}

    if objective == "kto":
        variant = _join_variants(rows, "response")
        policy = _segmented_logps(_packed_forward(model, variant, device), variant)
        if reference_cache is None:
            with torch.no_grad(), active_adapters(model, ()):
                reference = _segmented_logps(_packed_forward(model, variant, device), variant)
        else:
            reference = torch.tensor([reference_cache[row.id][0] for row in rows],
                                     device=device, dtype=policy.dtype)
        desirable = torch.tensor([bool(row.metadata["label"]) for row in rows], device=device)
        losses = kto_loss(policy, reference, desirable, beta=beta)
        return losses.mean(), ({"desirable_frac": float(desirable.float().mean()),
                                "packed_examples": len(rows)} if collect_metrics else {})

    chosen = _join_variants(rows, "chosen")
    rejected = _join_variants(rows, "rejected")
    if objective == "reward":
        if not isinstance(reward_head, OutcomeRewardHead):
            raise ValueError("reward objective needs an outcome reward head")
        _, chosen_hidden = _packed_forward(model, chosen, device, return_hidden=True)
        _, rejected_hidden = _packed_forward(model, rejected, device, return_hidden=True)
        chosen_reward = _packed_outcome_rewards(reward_head, chosen_hidden, chosen)
        rejected_reward = _packed_outcome_rewards(reward_head, rejected_hidden, rejected)
        losses = reward_model_loss(chosen_reward, rejected_reward)
        return losses.mean(), ({"reward_margin": float((chosen_reward - rejected_reward).mean().detach()),
                                "packed_examples": len(rows)} if collect_metrics else {})

    average = objective in ("orpo", "simpo")
    policy_chosen = _segmented_logps(_packed_forward(model, chosen, device), chosen, average=average)
    policy_rejected = _segmented_logps(_packed_forward(model, rejected, device), rejected,
                                       average=average)
    if objective == "dpo":
        if reference_cache is None:
            with torch.no_grad(), active_adapters(model, ()):
                reference_chosen = _segmented_logps(_packed_forward(model, chosen, device), chosen)
                reference_rejected = _segmented_logps(_packed_forward(model, rejected, device), rejected)
        else:
            reference_chosen = torch.tensor([reference_cache[row.id][0] for row in rows],
                                            device=device, dtype=policy_chosen.dtype)
            reference_rejected = torch.tensor([reference_cache[row.id][1] for row in rows],
                                              device=device, dtype=policy_rejected.dtype)
        result = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                          beta=beta)
    elif objective == "orpo":
        result = orpo_loss(policy_chosen, policy_rejected, -policy_chosen, beta=beta)
    elif objective == "simpo":
        result = simpo_loss(policy_chosen, policy_rejected, beta=beta, gamma=gamma)
    else:
        raise ValueError(f"unsupported packed objective {objective!r}")
    return result.loss.mean(), ({"preference_margin": float(result.margin.mean()),
                                 "packed_examples": len(rows)} if collect_metrics else {})


@torch.no_grad()
def _build_reference_cache(model: nn.Module, rows: list[TokenizedExample], objective: str,
                           device: str, batch_size: int) -> dict[str, tuple[float, ...]]:
    """Score the immutable frozen base once instead of once per optimizer step."""
    if objective not in ("dpo", "kto"):
        return {}
    cache: dict[str, tuple[float, ...]] = {}
    with active_adapters(model, ()):
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start:start + max(1, int(batch_size))]
            if objective == "kto":
                values, _ = _lm_logps(
                    model, [row.variants["response"] for row in batch], device)
                cpu = values.detach().float().cpu().tolist()
                cache.update((row.id, (float(value),)) for row, value in zip(batch, cpu))
            else:
                chosen, rejected = _paired_logps(
                    model, [row.variants["chosen"] for row in batch],
                    [row.variants["rejected"] for row in batch], device)
                chosen_cpu = chosen.detach().float().cpu().tolist()
                rejected_cpu = rejected.detach().float().cpu().tolist()
                cache.update((row.id, (float(left), float(right)))
                             for row, left, right in zip(batch, chosen_cpu, rejected_cpu))
    return cache


def _qualify_reset_packing(model: nn.Module, reward_head: nn.Module | None,
                           rows: list[TokenizedExample], objective: str, adapter_name: str,
                           device: str, beta: float, gamma: float,
                           parameters: list[nn.Parameter], tolerance: float = 3e-4,
                           reference_cache: dict | None = None) -> dict:
    """Require unpacked/packed loss and trainable-gradient parity before adoption."""
    model.zero_grad(set_to_none=True)
    if reward_head is not None:
        reward_head.zero_grad(set_to_none=True)
    unpacked, _ = _train_loss(model, reward_head, rows, objective, adapter_name,
                              device, beta, gamma, reference_cache=reference_cache,
                              collect_metrics=False)
    unpacked_gradients = torch.autograd.grad(unpacked, parameters, allow_unused=True)
    model.zero_grad(set_to_none=True)
    if reward_head is not None:
        reward_head.zero_grad(set_to_none=True)
    packed, _ = _packed_train_loss(
        model, reward_head, rows, objective, device, beta, gamma,
        reference_cache=reference_cache, collect_metrics=False)
    packed_gradients = torch.autograd.grad(packed, parameters, allow_unused=True)
    gradient_error = 0.0
    for left, right in zip(unpacked_gradients, packed_gradients):
        if left is None and right is None:
            continue
        if left is None or right is None:
            gradient_error = float("inf")
            break
        gradient_error = max(gradient_error, float((left.float() - right.float()).abs().max()))
    loss_error = abs(float(unpacked.detach()) - float(packed.detach()))
    passed = loss_error <= tolerance and gradient_error <= tolerance
    model.zero_grad(set_to_none=True)
    if reward_head is not None:
        reward_head.zero_grad(set_to_none=True)
    return {"schema": "rwkv-lab.reset-pack-qualification.v1", "examples": len(rows),
            "unpacked_loss": float(unpacked.detach()), "packed_loss": float(packed.detach()),
            "loss_max_abs": loss_error, "gradient_max_abs": gradient_error,
            "tolerance": tolerance, "passed": passed}


def train(*, checkpoint: str, data: str, output: str, objective: str = "sft",
          adapter_name: str = "posttrain", rank: int = 16, alpha: float = 32.0,
          targets: tuple[str, ...] = (), steps: int = 100, batch_size: int = 2,
          learning_rate: float = 2e-4, beta: float = 0.1, gamma: float = 1.0,
          max_length: int = 2048, seed: int = 0, device: str = "auto",
          template: str = "", eval_data: str = "", token_cache: str = "",
          max_train_tokens: int = 0, packing: str = "audit",
          base_quantization: str = "none", quant_block_size: int = 64,
          quant_backend: str = "auto", activation_offload: bool = False,
          log_every: int = 10) -> dict:
    from rwkv_lab.generate import WorldVocab, build_from_ckpt

    if objective not in ("sft", "dpo", "kto", "orpo", "simpo", "reward", "prm"):
        raise ValueError("unsupported post-training objective")
    if base_quantization not in ("none", "nf4") or packing not in ("off", "audit", "reset"):
        raise ValueError("base_quantization must be none/nf4 and packing must be off/audit/reset")
    if quant_backend not in ("auto", "portable", "torchao"):
        raise ValueError("quant_backend must be auto, portable, or torchao")
    if steps <= 0 or batch_size <= 0 or max_length < 2 or log_every <= 0:
        raise ValueError("steps/batch_size must be positive and max_length must be at least two")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, blob = build_from_ckpt(checkpoint, device=device)
    if device == "cpu":
        model = model.float()
    dense_storage = model_storage_bytes(model)
    quantized_modules: list[str] = []
    quantization_qualification = None
    selected_quant_backend = "none"
    if base_quantization == "nf4":
        representative = next((module for path, module in model.named_modules()
                               if isinstance(module, nn.Linear) and path and
                               not path.startswith("head") and not path.startswith("emb")), None)
        if representative is None:
            raise ValueError("NF4 qualification found no eligible dense linear")
        sample = torch.randn(2, representative.in_features, device=representative.weight.device,
                             dtype=representative.weight.dtype)
        portable_report = qualify_linear_qlora(
            representative, sample, block_size=quant_block_size,
            parity_tolerance=(2e-2 if representative.weight.dtype in (torch.bfloat16, torch.float16)
                              else 2e-5))
        accelerated_report = qualify_accelerated_nf4(representative, sample,
                                                      block_size=quant_block_size)
        quantization_qualification = {"portable": portable_report.__dict__,
                                      "accelerated": accelerated_report}
        if quant_backend == "auto" and not accelerated_report.get("adopted"):
            raise ValueError("automatic NF4 requires an adopted accelerated backend; "
                             "use --quant-backend portable only for correctness-scale runs: "
                             f"{accelerated_report}")
        selected_quant_backend = ("torchao" if quant_backend == "auto" else quant_backend)
        if selected_quant_backend == "torchao" and not accelerated_report.get("adopted"):
            raise ValueError(f"TorchAO NF4 failed parity/performance qualification: {accelerated_report}")
        if not portable_report.passed:
            raise ValueError(f"portable NF4 failed QLoRA qualification: {portable_report}")
        quantized_modules = quantize_model_nf4(model, block_size=quant_block_size,
                                               exclude=("head", "emb"),
                                               backend=selected_quant_backend)
    base_storage = model_storage_bytes(model)
    config = AdapterConfig(adapter_name, rank, alpha, 0.0,
                           targets or ("receptance", "key", "value", "output", "up", "down"))
    selected = inject_lora(model, config)
    rows, _ = load_jsonl(data)
    required_kind = {"sft": "sft", "kto": "feedback", "reward": "preference",
                     "prm": "prm"}.get(
        objective, "preference")
    train_rows = [row for row in rows if row.split == "train" and row.kind == required_kind]
    if eval_data:
        separate_eval, _ = load_jsonl(eval_data)
        eval_rows = [row for row in separate_eval if row.kind == required_kind]
    else:
        eval_rows = [row for row in rows if row.split in ("eval", "test") and row.kind == required_kind]
    if not train_rows:
        raise ValueError(f"dataset has no train/{required_kind} examples for {objective}")
    tokenizer = WorldVocab()
    chat_template = load_template(template) if template else DEFAULT_TEMPLATE
    cache_manifest = None
    if token_cache:
        all_encoded, cache_manifest = cache_tokenized(data, tokenizer, token_cache,
                                                      max_length=max_length,
                                                      template=chat_template,
                                                      tokenizer_fingerprint="rwkv-world-vocab-v1")
        encoded = [row for row in all_encoded if row.split == "train" and row.kind == required_kind]
    else:
        encoded = [tokenize(render(row, chat_template), tokenizer, max_length=max_length,
                            truncate="left") for row in train_rows]
    encoded_eval = [tokenize(render(row, chat_template), tokenizer, max_length=max_length,
                             truncate="left") for row in eval_rows]
    reference_cache = _build_reference_cache(
        model, encoded + encoded_eval, objective, device, batch_size)
    variant_name = {"sft": "sft", "kto": "response", "prm": "steps"}.get(objective, "chosen")
    packing_audit = None
    if packing in ("audit", "reset"):
        _, packing_audit = pack_variants([(row.id, row.variants[variant_name]) for row in encoded],
                                         max_length, separator_id=0)
    kto_pools = None
    if objective == "kto":
        good = [row for row in encoded if bool(row.metadata["label"])]
        bad = [row for row in encoded if not bool(row.metadata["label"])]
        if not good or not bad or batch_size < 2:
            raise ValueError("KTO requires both feedback labels and batch_size >= 2")
        kto_pools = (good, bad)
    reward_head = (OutcomeRewardHead(int(blob["arch"]["d_model"]), bias=False).to(device)
                   if objective == "reward" else
                   ProcessRewardHead(int(blob["arch"]["d_model"]), bias=False).to(device)
                   if objective == "prm" else None)
    parameters = list(adapter_parameters(model)) + (list(reward_head.parameters()) if reward_head else [])
    packed_groups: list[list[TokenizedExample]] = []
    if packing == "reset":
        reason = model.packing_incompatibility() if hasattr(model, "packing_incompatibility") else None
        if reason:
            raise ValueError(f"reset-mask packing is unavailable: {reason}")
        pack_names = (("chosen", "rejected") if objective in ("dpo", "orpo", "simpo", "reward")
                      else ("response",) if objective == "kto" else
                      ("steps",) if objective == "prm" else ("sft",))
        packed_groups = _pack_groups(encoded, pack_names, max_length)
        qualification_rows = next((group for group in packed_groups if len(group) > 1), [])
        if not qualification_rows:
            raise ValueError("reset packing needs at least two examples that fit one context")
        qualification = _qualify_reset_packing(model, reward_head, qualification_rows,
                                                objective, adapter_name, device, beta, gamma,
                                                parameters, reference_cache=reference_cache)
        packed_tokens = sum(len(row.variants[name].input_ids) for group in packed_groups
                            for row in group for name in pack_names)
        packed_capacity = len(packed_groups) * max_length * len(pack_names)
        packing_audit = {**(packing_audit or {}), "execution": "reset_mask",
                         "groups": len(packed_groups), "qualification": qualification,
                         "execution_utilization": packed_tokens / max(1, packed_capacity),
                         "examples_per_group": [len(group) for group in packed_groups],
                         "qualification_required": False,
                         "recurrent_isolation": bool(qualification["passed"])}
        if not qualification["passed"]:
            raise ValueError(f"reset-mask packing parity failed: {qualification}")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    log = (root / "train.jsonl").open("w", buffering=1)

    eval_details: dict[str, object] = {}

    def evaluate() -> float | None:
        if not encoded_eval:
            return None
        model.eval()
        values = []
        per_example = []
        with torch.no_grad():
            for row in encoded_eval:
                value, _ = _train_loss(model, reward_head, [row],
                                       objective, adapter_name, device, beta, gamma,
                                       reference_cache=reference_cache)
                values.append(float(value))
                per_example.append({"id": row.id, "loss": float(value),
                                    "family": str(row.metadata.get("family") or "default")})
        eval_details["per_example"] = per_example
        families: dict[str, list[float]] = {}
        for row in per_example:
            families.setdefault(str(row["family"]), []).append(float(row["loss"]))
        eval_details["families"] = {name: sum(group) / len(group) for name, group in families.items()}
        if objective == "prm" and isinstance(reward_head, ProcessRewardHead):
            clean_logits, clean_labels = [], []
            adversarial_logits, adversarial_labels = [], []
            for row in encoded_eval:
                for name, output_logits, output_labels in (
                    ("steps", clean_logits, clean_labels),
                    ("adversarial", adversarial_logits, adversarial_labels),
                ):
                    if name not in row.variants:
                        continue
                    ids, _, positions, labels, mask = _collate_steps([row.variants[name]], device)
                    _, hidden = model(ids, return_hidden=True)
                    logits = reward_head(hidden, positions)
                    output_logits.append(logits[mask].detach().cpu())
                    output_labels.append(labels[mask].detach().cpu())
            if clean_logits:
                clean = torch.cat(clean_logits)
                truth = torch.cat(clean_labels)
                eval_details["calibration"] = binary_calibration(clean, truth)
                if adversarial_logits:
                    eval_details["adversarial"] = adversarial_reward_audit(
                        clean, torch.cat(adversarial_logits), torch.cat(adversarial_labels))
        model.train()
        return sum(values) / len(values)

    initial_eval = evaluate()
    initial_eval_details = copy.deepcopy(eval_details)
    if initial_eval is not None:
        log.write(json.dumps({"kind": "eval", "step": 0, "loss": initial_eval,
                              "objective": objective}) + "\n")
    model.train()
    loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    last_loss_t = loss_sum_t
    finite_window_t = torch.ones((), dtype=torch.bool, device=device)
    scalar_losses: list[float] = []
    last_metrics: dict[str, float] = {}
    train_tokens = 0
    completed_steps = 0
    for step in range(int(steps)):
        if packing == "reset" and objective != "kto":
            batch = packed_groups[rng.randrange(len(packed_groups))]
        elif kto_pools is not None:
            good, bad = kto_pools
            batch = [good[rng.randrange(len(good))], bad[rng.randrange(len(bad))]]
            candidates = [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size) - 2)]
            if packing == "reset":
                used = sum(len(row.variants["response"].input_ids) for row in batch)
                for row in candidates:
                    size = len(row.variants["response"].input_ids)
                    if used + size <= max_length:
                        batch.append(row)
                        used += size
                if used > max_length:
                    raise ValueError("one good/bad KTO pair does not fit the packing context")
            else:
                batch += candidates
            rng.shuffle(batch)
        else:
            batch = [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size))]
        optimizer.zero_grad(set_to_none=True)
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if activation_offload and device.startswith("cuda") else nullcontext())
        with offload:
            collect_metrics = (step == 0 or step + 1 == steps or (step + 1) % log_every == 0)
            if packing == "reset":
                loss, last_metrics = _packed_train_loss(model, reward_head, batch, objective,
                                                        device, beta, gamma,
                                                        reference_cache=reference_cache,
                                                        collect_metrics=collect_metrics)
            else:
                loss, last_metrics = _train_loss(model, reward_head, batch, objective, adapter_name,
                                                 device, beta, gamma,
                                                 reference_cache=reference_cache,
                                                 collect_metrics=collect_metrics)
        loss.backward()
        finite = torch.isfinite(loss.detach())
        finite_window_t.logical_and_(finite)
        # Keep a bad update from poisoning adapter/reward-head state while deferring the host
        # status read to telemetry cadence.  Adapter training has few gradient tensors, so these
        # device-side passes cost less than a full CUDA-stream synchronization every step.
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        for gradient in gradients:
            gradient.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        gradient_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for gradient in gradients:
            gradient_groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
        for (_, dtype), group in gradient_groups.items():
            torch._foreach_mul_(group, finite.to(dtype=dtype))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        train_tokens += _batch_tokens(batch, objective)
        completed_steps = step + 1
        safe_loss = torch.where(finite, loss.detach().float(), torch.zeros_like(loss.detach().float()))
        loss_sum_t.add_(safe_loss)
        last_loss_t = safe_loss
        telemetry_due = (completed_steps == steps or completed_steps % log_every == 0)
        if telemetry_due:
            if not bool(finite_window_t):
                raise FloatingPointError(
                    f"non-finite {objective} loss in steps "
                    f"{max(1, completed_steps - log_every + 1)}..{completed_steps}")
            scalar_loss = float(last_loss_t)
            scalar_losses.append(scalar_loss)
            log.write(json.dumps({"kind": "train", "step": completed_steps,
                                  "loss": scalar_loss, "objective": objective,
                                  "telemetry_steps": min(log_every, completed_steps),
                                  **last_metrics}) + "\n")
            finite_window_t.fill_(True)
        if max_train_tokens and train_tokens >= max_train_tokens:
            break
    if completed_steps and completed_steps % log_every:
        if not bool(finite_window_t):
            raise FloatingPointError(f"non-finite {objective} loss before step {completed_steps}")
        scalar_loss = float(last_loss_t)
        scalar_losses.append(scalar_loss)
        log.write(json.dumps({"kind": "train", "step": completed_steps,
                              "loss": scalar_loss, "objective": objective,
                              "telemetry_steps": completed_steps % log_every,
                              **last_metrics}) + "\n")
    final_eval = evaluate()
    if final_eval is not None:
        log.write(json.dumps({"kind": "eval", "step": completed_steps, "loss": final_eval,
                              "objective": objective}) + "\n")
    log.close()
    adapter_manifest = save_adapter(model, root / "adapter", adapter_name,
                                    parent_checkpoint=str(Path(checkpoint).resolve()),
                                    metadata={"objective": objective,
                                              "base_quantization": base_quantization,
                                              "quant_block_size": quant_block_size,
                                              "quant_backend": selected_quant_backend})
    reward_version = None
    if reward_head is not None:
        from safetensors.torch import save_file
        head_path = root / ("process_reward_head.safetensors" if objective == "prm"
                            else "reward_head.safetensors")
        save_file({"weight": reward_head.proj.weight.detach().cpu()}, str(head_path),
                  metadata={"schema": "rwkv-lab.reward-head.v1", "objective": objective})
        reward_version = {"schema": "rwkv-lab.reward-model-version.v1",
                          "kind": "process" if objective == "prm" else "outcome",
                          "base_checkpoint": str(Path(checkpoint).resolve()),
                          "dataset_sha256": dataset_manifest(data, template=chat_template)["sha256"],
                          "head": head_path.name, "head_sha256": _sha256(head_path),
                          "calibration": eval_details.get("calibration", {}),
                          "adversarial_audit": eval_details.get("adversarial", {}),
                          "promotion": "unassessed"}
        (root / "reward-model.json").write_text(json.dumps(reward_version, indent=2,
                                                            sort_keys=True) + "\n")
    final_loss = float(last_loss_t)
    mean_loss = float(loss_sum_t / max(1, completed_steps))
    result = {"schema": RESULT_SCHEMA, "objective": objective, "steps": completed_steps,
              "examples": len(encoded), "final_loss": final_loss,
              "mean_loss": mean_loss, "metrics": last_metrics,
              "eval_examples": len(encoded_eval), "initial_eval_loss": initial_eval,
              "final_eval_loss": final_eval,
              "initial_eval": initial_eval_details, "eval": eval_details,
              "train_tokens": train_tokens,
              "dataset": dataset_manifest(data, template=chat_template), "adapter": adapter_manifest,
              "eval_dataset": (dataset_manifest(eval_data, template=chat_template)
                               if eval_data else None),
              "targets": selected, "seed": seed,
              "token_cache": cache_manifest, "packing": packing_audit,
              "quantization": {"kind": base_quantization, "backend": selected_quant_backend,
                               "modules": quantized_modules,
                               "dense_bytes": dense_storage, "stored_bytes": base_storage,
                               "compression_ratio": dense_storage / max(1, base_storage),
                               "qualification": quantization_qualification},
              "activation_offload": activation_offload, "log_every": log_every,
              "reward_model": reward_version,
              "promotion": {"status": "unassessed", "reason": "training never implies promotion"}}
    (root / "posttrain-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RWKV adapter post-training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", default="sft",
                        choices=["sft", "dpo", "kto", "orpo", "simpo", "reward", "prm"])
    parser.add_argument("--adapter-name", default="posttrain")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--targets", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--template", default="", help="custom ChatTemplate JSON")
    parser.add_argument("--eval-data", default="", help="separate frozen held-out JSONL")
    parser.add_argument("--token-cache", default="", help="content-addressed token-cache root")
    parser.add_argument("--max-train-tokens", type=int, default=0,
                        help="equal-budget stop after at least this many scored input tokens")
    parser.add_argument("--packing", choices=["off", "audit", "reset"], default="audit")
    parser.add_argument("--base-quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--quant-block-size", type=int, default=64)
    parser.add_argument("--quant-backend", choices=["auto", "portable", "torchao"], default="auto")
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--log-every", type=int, default=10,
                        help="materialize train loss/non-finite status every N updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    values = vars(args)
    values["targets"] = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    print(json.dumps(train(**values), sort_keys=True))


if __name__ == "__main__":
    main()
