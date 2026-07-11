"""Deterministic decoder evaluation for recurrent language models.

The evaluation matrix follows Shi et al., *A Thorough Examination of Decoding
Methods in the Era of LLMs* (EMNLP 2024), https://arxiv.org/abs/2402.06925.
RWKV-specific state-drift and free-running checks were proposed in the RWKV
Discord decoding channel:
https://discord.com/channels/992359628979568762/1076020543205163118/1236635292665118822

The harness deliberately treats decoding as part of an experiment, rather than
reporting teacher-forced loss and silently choosing one sampler afterward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Iterable, Sequence

import torch


@dataclass(frozen=True)
class DecoderConfig:
    name: str
    method: str = "top_p"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    typical_p: float = 1.0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if self.method not in {"greedy", "top_k", "top_p", "typical", "mirostat"}:
            raise ValueError(f"unsupported decoder method {self.method!r}")
        if self.temperature <= 0 or self.top_k < 0:
            raise ValueError("temperature must be positive and top_k non-negative")
        if not 0 < self.top_p <= 1 or not 0 < self.typical_p <= 1:
            raise ValueError("top_p and typical_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")


def _flatten_state(value: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    def visit(item):
        if torch.is_tensor(item):
            tensors.append(item.detach().float().reshape(-1).cpu())
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            visit(vars(item))
    visit(value)
    return torch.cat(tensors) if tensors else torch.zeros(0)


def state_divergence(left: Any, right: Any) -> float:
    """Normalized L2 divergence between two recurrent-state trees."""
    a, b = _flatten_state(left), _flatten_state(right)
    if a.shape != b.shape:
        raise ValueError("recurrent states have different geometry")
    return float((a - b).norm() / b.norm().clamp_min(1e-12)) if a.numel() else 0.0


def sample_logits(logits: torch.Tensor, config: DecoderConfig, *, generator: torch.Generator,
                  history: Sequence[int] = (), mirostat_mu: float | None = None
                  ) -> tuple[int, float | None, float]:
    """Sample one token and return ``(token, next_mirostat_mu, entropy)``."""
    scores = logits.detach().float().flatten().clone()
    if scores.numel() == 0:
        raise ValueError("cannot sample empty logits")
    if config.repetition_penalty > 1:
        for token in set(int(x) for x in history):
            if 0 <= token < scores.numel():
                scores[token] = (scores[token] / config.repetition_penalty
                                 if scores[token] >= 0 else scores[token] * config.repetition_penalty)
    scores /= config.temperature
    raw_prob = torch.softmax(scores, -1)
    entropy = float(-(raw_prob * raw_prob.clamp_min(1e-20).log()).sum())
    if config.method == "greedy":
        return int(scores.argmax()), mirostat_mu, entropy

    keep = torch.ones_like(scores, dtype=torch.bool)
    if config.method == "top_k" or config.top_k:
        k = min(config.top_k or scores.numel(), scores.numel())
        threshold = torch.topk(scores, k).values[-1]
        keep &= scores >= threshold
    if config.method == "top_p" or config.top_p < 1:
        order = scores.argsort(descending=True)
        cumulative = torch.softmax(scores[order], -1).cumsum(-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone(); remove[0] = False
        keep[order[remove]] = False
    if config.method == "typical":
        surprisal = -raw_prob.clamp_min(1e-20).log()
        order = (surprisal - entropy).abs().argsort()
        cumulative = raw_prob[order].cumsum(-1)
        remove = cumulative > config.typical_p
        remove[1:] = remove[:-1].clone(); remove[0] = False
        keep[order[remove]] = False
    if config.method == "mirostat":
        mu = float(2 * config.mirostat_tau if mirostat_mu is None else mirostat_mu)
        surprise = -torch.log2(raw_prob.clamp_min(1e-20))
        keep &= surprise <= max(mu, 1e-3)
        if not keep.any():
            keep[scores.argmax()] = True
    filtered = scores.masked_fill(~keep, float("-inf"))
    probability = torch.softmax(filtered, -1)
    # Draw on CPU so one RNG tape is reproducible across CPU/CUDA evaluators.
    token = int(torch.multinomial(probability.cpu(), 1, generator=generator))
    if config.method == "mirostat":
        observed = float(-torch.log2(raw_prob[token].clamp_min(1e-20)))
        mirostat_mu = mu - config.mirostat_eta * (observed - config.mirostat_tau)
    return token, mirostat_mu, entropy


def evaluate_decoders(prompts: Iterable[Sequence[int]], configs: Iterable[DecoderConfig], *,
                      step: Callable[[int, Any], tuple[torch.Tensor, Any]], max_new: int,
                      seed: int = 0, stop_tokens: Sequence[int] = (),
                      score: Callable[[Sequence[int], Sequence[int]], float] | None = None,
                      reference_method: str = "greedy") -> dict[str, dict[str, Any]]:
    """Run paired decoding tapes through a token-step callback.

    ``step(token, state)`` consumes one token and returns next-token logits plus
    the new state. Every decoder sees identical prompts and deterministic seeds.
    """
    prompts = [tuple(map(int, prompt)) for prompt in prompts]
    if max_new < 1 or not prompts:
        raise ValueError("decoder evaluation needs prompts and a positive token budget")
    configs = list(configs)
    if not configs or len({cfg.name for cfg in configs}) != len(configs):
        raise ValueError("decoder configurations need unique names")
    traces: dict[str, list[tuple[list[int], Any, list[float], float]]] = {}
    for ci, cfg in enumerate(configs):
        rows = []
        for pi, prompt in enumerate(prompts):
            generator = torch.Generator(device="cpu").manual_seed(seed + pi * 1009 + ci * 9176)
            state = None; logits = None
            started = time.perf_counter()
            for token in prompt:
                logits, state = step(token, state)
            output: list[int] = []; entropies: list[float] = []; mu = None
            for _ in range(max_new):
                token, mu, entropy = sample_logits(logits, cfg, generator=generator,
                                                   history=output, mirostat_mu=mu)
                output.append(token); entropies.append(entropy)
                logits, state = step(token, state)
                if token in stop_tokens:
                    break
            rows.append((output, state, entropies, time.perf_counter() - started))
        traces[cfg.name] = rows
    reference = next((cfg.name for cfg in configs if cfg.method == reference_method), configs[0].name)
    result = {}
    for cfg in configs:
        rows = traces[cfg.name]
        total_tokens = sum(len(row[0]) for row in rows)
        loops = [1.0 if len(out) >= 6 and len(set(out[-6:])) <= 2 else 0.0 for out, *_ in rows]
        divergences = [state_divergence(row[1], ref[1])
                       for row, ref in zip(rows, traces[reference])]
        scores = [score(prompts[i], row[0]) for i, row in enumerate(rows)] if score else []
        result[cfg.name] = {"config": asdict(cfg), "samples": [row[0] for row in rows],
                            "mean_entropy": sum(sum(r[2]) for r in rows) / max(total_tokens, 1),
                            "loop_rate": sum(loops) / len(loops),
                            "state_divergence": sum(divergences) / len(divergences),
                            "tokens_per_second": total_tokens / max(sum(r[3] for r in rows), 1e-9),
                            "mean_score": (sum(scores) / len(scores) if scores else None)}
    return result
