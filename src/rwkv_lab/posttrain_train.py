"""Native-RWKV SFT, preference, feedback, and outcome-reward training.

This is the executable bridge between ``posttrain_data``, named LoRA adapters, and the losses in
``preference``.  It intentionally accepts only trusted, self-describing rwkv-lab checkpoints and
never executes dataset content.  Adamaton remains responsible for isolated tool execution and
verifier fleets.

Paper references used by the implementation are kept beside their primitives:
LoRA/QLoRA in ``adapters.py`` and DPO/KTO/ORPO/SimPO in ``preference.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import nn

from rwkv_lab.adapters import (AdapterConfig, active_adapters, adapter_parameters,
                               inject_lora, save_adapter)
from rwkv_lab.posttrain_data import (IGNORE_INDEX, TokenizedExample, TokenizedVariant,
                                     DEFAULT_TEMPLATE, dataset_manifest, load_jsonl, load_template,
                                     render, tokenize)
from rwkv_lab.preference import (dpo_loss, kto_loss, orpo_loss, reward_model_loss,
                                 OutcomeRewardHead, sequence_logps, simpo_loss)


RESULT_SCHEMA = "rwkv-lab.posttrain-result.v1"


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

    chosen = [row.variants["chosen"] for row in batch]
    rejected = [row.variants["rejected"] for row in batch]
    if objective == "reward":
        if reward_head is None:
            raise ValueError("reward objective needs a reward head")
        ci, cl = _collate(chosen, device)
        ri, rl = _collate(rejected, device)
        _, ch = model(ci, return_hidden=True)
        _, rh = model(ri, return_hidden=True)
        cr = reward_head(ch, cl != IGNORE_INDEX)
        rr = reward_head(rh, rl != IGNORE_INDEX)
        losses = reward_model_loss(cr, rr)
        return losses.mean(), {"reward_margin": float((cr - rr).detach().mean())}

    average = objective in ("orpo", "simpo")
    policy_chosen, _ = _lm_logps(model, chosen, device, average=average)
    policy_rejected, _ = _lm_logps(model, rejected, device, average=average)
    if objective == "dpo":
        with torch.no_grad(), active_adapters(model, ()):
            reference_chosen, _ = _lm_logps(model, chosen, device)
            reference_rejected, _ = _lm_logps(model, rejected, device)
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
          template: str = "") -> dict:
    from rwkv_lab.generate import WorldVocab, build_from_ckpt

    if objective not in ("sft", "dpo", "kto", "orpo", "simpo", "reward"):
        raise ValueError("objective must be sft, dpo, kto, orpo, simpo, or reward")
    if steps <= 0 or batch_size <= 0 or max_length < 2:
        raise ValueError("steps/batch_size must be positive and max_length must be at least two")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, blob = build_from_ckpt(checkpoint, device=device)
    if device == "cpu":
        model = model.float()
    config = AdapterConfig(adapter_name, rank, alpha, 0.0,
                           targets or ("receptance", "key", "value", "output", "up", "down"))
    selected = inject_lora(model, config)
    rows, _ = load_jsonl(data)
    required_kind = {"sft": "sft", "kto": "feedback", "reward": "preference"}.get(
        objective, "preference")
    train_rows = [row for row in rows if row.split == "train" and row.kind == required_kind]
    eval_rows = [row for row in rows if row.split in ("eval", "test") and row.kind == required_kind]
    if not train_rows:
        raise ValueError(f"dataset has no train/{required_kind} examples for {objective}")
    tokenizer = WorldVocab()
    chat_template = load_template(template) if template else DEFAULT_TEMPLATE
    encoded = [tokenize(render(row, chat_template), tokenizer, max_length=max_length, truncate="left")
               for row in train_rows]
    encoded_eval = [tokenize(render(row, chat_template), tokenizer, max_length=max_length, truncate="left")
                    for row in eval_rows]
    kto_pools = None
    if objective == "kto":
        good = [row for row in encoded if bool(row.metadata["label"])]
        bad = [row for row in encoded if not bool(row.metadata["label"])]
        if not good or not bad or batch_size < 2:
            raise ValueError("KTO requires both feedback labels and batch_size >= 2")
        kto_pools = (good, bad)
    reward_head = (OutcomeRewardHead(int(blob["arch"]["d_model"]), bias=False).to(device)
                   if objective == "reward" else None)
    parameters = list(adapter_parameters(model)) + (list(reward_head.parameters()) if reward_head else [])
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    log = (root / "train.jsonl").open("w", buffering=1)

    def evaluate() -> float | None:
        if not encoded_eval:
            return None
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(encoded_eval), batch_size):
                value, _ = _train_loss(model, reward_head, encoded_eval[start:start + batch_size],
                                       objective, adapter_name, device, beta, gamma)
                values.append(float(value))
        model.train()
        return sum(values) / len(values)

    initial_eval = evaluate()
    if initial_eval is not None:
        log.write(json.dumps({"kind": "eval", "step": 0, "loss": initial_eval,
                              "objective": objective}) + "\n")
    model.train()
    losses = []
    last_metrics: dict[str, float] = {}
    for step in range(int(steps)):
        if kto_pools is not None:
            good, bad = kto_pools
            batch = [good[rng.randrange(len(good))], bad[rng.randrange(len(bad))]]
            batch += [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size) - 2)]
            rng.shuffle(batch)
        else:
            batch = [encoded[rng.randrange(len(encoded))] for _ in range(int(batch_size))]
        optimizer.zero_grad(set_to_none=True)
        loss, last_metrics = _train_loss(model, reward_head, batch, objective, adapter_name,
                                         device, beta, gamma)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {objective} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        log.write(json.dumps({"kind": "train", "step": step + 1, "loss": losses[-1],
                              "objective": objective, **last_metrics}) + "\n")
    final_eval = evaluate()
    if final_eval is not None:
        log.write(json.dumps({"kind": "eval", "step": int(steps), "loss": final_eval,
                              "objective": objective}) + "\n")
    log.close()
    adapter_manifest = save_adapter(model, root / "adapter", adapter_name,
                                    parent_checkpoint=str(Path(checkpoint).resolve()),
                                    metadata={"objective": objective})
    if reward_head is not None:
        from safetensors.torch import save_file
        save_file({"weight": reward_head.proj.weight.detach().cpu()}, str(root / "reward_head.safetensors"),
                  metadata={"schema": "rwkv-lab.reward-head.v1"})
    result = {"schema": RESULT_SCHEMA, "objective": objective, "steps": int(steps),
              "examples": len(encoded), "final_loss": losses[-1],
              "mean_loss": sum(losses) / len(losses), "metrics": last_metrics,
              "eval_examples": len(encoded_eval), "initial_eval_loss": initial_eval,
              "final_eval_loss": final_eval,
              "dataset": dataset_manifest(data, template=chat_template), "adapter": adapter_manifest,
              "targets": selected, "seed": seed,
              "promotion": {"status": "unassessed", "reason": "training never implies promotion"}}
    (root / "posttrain-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RWKV adapter post-training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", default="sft",
                        choices=["sft", "dpo", "kto", "orpo", "simpo", "reward"])
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    values = vars(args)
    values["targets"] = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    print(json.dumps(train(**values), sort_keys=True))


if __name__ == "__main__":
    main()
