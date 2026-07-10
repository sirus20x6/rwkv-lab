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
from rwkv_lab.quantization import model_storage_bytes, quantize_model_nf4


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


def _collate(variants: list[TokenizedVariant], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(v.input_ids) for v in variants)
    ids = torch.zeros(len(variants), width, dtype=torch.long, device=device)
    labels = torch.full((len(variants), width), IGNORE_INDEX, dtype=torch.long, device=device)
    for i, variant in enumerate(variants):
        n = len(variant.input_ids)
        ids[i, :n] = torch.tensor(variant.input_ids, device=device)
        labels[i, :n] = torch.tensor(variant.labels, device=device)
    return ids, labels


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
    positions = torch.zeros(len(variants), width, dtype=torch.long, device=device)
    step_labels = torch.zeros(len(variants), width, dtype=torch.float32, device=device)
    mask = torch.zeros(len(variants), width, dtype=torch.bool, device=device)
    for index, value in enumerate(variants):
        count = len(value.step_positions)
        positions[index, :count] = torch.tensor(value.step_positions, device=device)
        step_labels[index, :count] = torch.tensor(value.step_labels, dtype=torch.float32, device=device)
        mask[index, :count] = True
    return ids, labels, positions, step_labels, mask


def _train_loss(model: nn.Module, reward_head: nn.Module | None, batch: list[TokenizedExample],
                objective: str, adapter_name: str, device: str, beta: float,
                gamma: float) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "sft":
        ids, labels = _collate([row.variants["sft"] for row in batch], device)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return loss, {"sft_nll": float(loss.detach())}

    if objective == "kto":
        variants = [row.variants["response"] for row in batch]
        policy, _ = _lm_logps(model, variants, device)
        with torch.no_grad(), active_adapters(model, ()):
            reference, _ = _lm_logps(model, variants, device)
        desirable = torch.tensor([bool(row.metadata["label"]) for row in batch], device=device)
        losses = kto_loss(policy, reference, desirable, beta=beta)
        return losses.mean(), {"desirable_frac": float(desirable.float().mean())}

    if objective == "prm":
        if not isinstance(reward_head, ProcessRewardHead):
            raise ValueError("PRM objective needs a process reward head")
        variants = [row.variants["steps"] for row in batch]
        ids, _, positions, step_labels, step_mask = _collate_steps(variants, device)
        _, hidden = model(ids, return_hidden=True)
        logits = reward_head(hidden, positions)
        loss = process_reward_loss(logits, step_labels, step_mask)
        metrics = binary_calibration(logits[step_mask], step_labels[step_mask])
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
        return losses.mean(), {"reward_margin": float((cr - rr).detach().mean())}

    average = objective in ("orpo", "simpo")
    policy_chosen, policy_rejected = _paired_logps(model, chosen, rejected, device, average=average)
    if objective == "dpo":
        with torch.no_grad(), active_adapters(model, ()):
            reference_chosen, reference_rejected = _paired_logps(model, chosen, rejected, device)
        result = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                          beta=beta)
    elif objective == "orpo":
        result = orpo_loss(policy_chosen, policy_rejected, -policy_chosen, beta=beta)
    elif objective == "simpo":
        result = simpo_loss(policy_chosen, policy_rejected, beta=beta, gamma=gamma)
    else:
        raise ValueError(f"unsupported objective {objective!r}")
    return result.loss.mean(), {"preference_margin": float(result.margin.mean())}


def train(*, checkpoint: str, data: str, output: str, objective: str = "sft",
          adapter_name: str = "posttrain", rank: int = 16, alpha: float = 32.0,
          targets: tuple[str, ...] = (), steps: int = 100, batch_size: int = 2,
          learning_rate: float = 2e-4, beta: float = 0.1, gamma: float = 1.0,
          max_length: int = 2048, seed: int = 0, device: str = "auto",
          template: str = "", eval_data: str = "", token_cache: str = "",
          max_train_tokens: int = 0, packing: str = "audit",
          base_quantization: str = "none", quant_block_size: int = 64,
          activation_offload: bool = False) -> dict:
    from rwkv_lab.generate import WorldVocab, build_from_ckpt

    if objective not in ("sft", "dpo", "kto", "orpo", "simpo", "reward", "prm"):
        raise ValueError("unsupported post-training objective")
    if base_quantization not in ("none", "nf4") or packing not in ("off", "audit"):
        raise ValueError("base_quantization must be none/nf4 and packing must be off/audit")
    if steps <= 0 or batch_size <= 0 or max_length < 2:
        raise ValueError("steps/batch_size must be positive and max_length must be at least two")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, blob = build_from_ckpt(checkpoint, device=device)
    if device == "cpu":
        model = model.float()
    dense_storage = model_storage_bytes(model)
    quantized_modules: list[str] = []
    if base_quantization == "nf4":
        quantized_modules = quantize_model_nf4(model, block_size=quant_block_size,
                                               exclude=("head", "emb"))
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
    variant_name = {"sft": "sft", "kto": "response", "prm": "steps"}.get(objective, "chosen")
    packing_audit = None
    if packing == "audit":
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
                                       objective, adapter_name, device, beta, gamma)
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
    losses = []
    last_metrics: dict[str, float] = {}
    train_tokens = 0
    completed_steps = 0
    for step in range(int(steps)):
        if kto_pools is not None:
            good, bad = kto_pools
            batch = [good[rng.randrange(len(good))], bad[rng.randrange(len(bad))]]
            batch += [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size) - 2)]
            rng.shuffle(batch)
        else:
            batch = [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size))]
        optimizer.zero_grad(set_to_none=True)
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if activation_offload and device.startswith("cuda") else nullcontext())
        with offload:
            loss, last_metrics = _train_loss(model, reward_head, batch, objective, adapter_name,
                                             device, beta, gamma)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {objective} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        train_tokens += _batch_tokens(batch, objective)
        completed_steps = step + 1
        losses.append(float(loss.detach()))
        log.write(json.dumps({"kind": "train", "step": step + 1, "loss": losses[-1],
                              "objective": objective, **last_metrics}) + "\n")
        if max_train_tokens and train_tokens >= max_train_tokens:
            break
    final_eval = evaluate()
    if final_eval is not None:
        log.write(json.dumps({"kind": "eval", "step": completed_steps, "loss": final_eval,
                              "objective": objective}) + "\n")
    log.close()
    adapter_manifest = save_adapter(model, root / "adapter", adapter_name,
                                    parent_checkpoint=str(Path(checkpoint).resolve()),
                                    metadata={"objective": objective,
                                              "base_quantization": base_quantization,
                                              "quant_block_size": quant_block_size})
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
    result = {"schema": RESULT_SCHEMA, "objective": objective, "steps": completed_steps,
              "examples": len(encoded), "final_loss": losses[-1],
              "mean_loss": sum(losses) / len(losses), "metrics": last_metrics,
              "eval_examples": len(encoded_eval), "initial_eval_loss": initial_eval,
              "final_eval_loss": final_eval,
              "initial_eval": initial_eval_details, "eval": eval_details,
              "train_tokens": train_tokens,
              "dataset": dataset_manifest(data, template=chat_template), "adapter": adapter_manifest,
              "eval_dataset": (dataset_manifest(eval_data, template=chat_template)
                               if eval_data else None),
              "targets": selected, "seed": seed,
              "token_cache": cache_manifest, "packing": packing_audit,
              "quantization": {"kind": base_quantization, "modules": quantized_modules,
                               "dense_bytes": dense_storage, "stored_bytes": base_storage,
                               "compression_ratio": dense_storage / max(1, base_storage)},
              "activation_offload": activation_offload, "reward_model": reward_version,
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
    parser.add_argument("--packing", choices=["off", "audit"], default="audit")
    parser.add_argument("--base-quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--quant-block-size", type=int, default=64)
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    values = vars(args)
    values["targets"] = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    print(json.dumps(train(**values), sort_keys=True))


if __name__ == "__main__":
    main()
