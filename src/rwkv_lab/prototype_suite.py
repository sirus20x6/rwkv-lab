"""Declarative runner for small architecture probes and comparative proofs."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


_MODEL_ARCHITECTURES = {"rwkv8", "kan_rwkv", "blt_rwkv7", "rosa_blt"}
_TASK_KINDS = {"model_probe", "blt_comparison", "standard_benchmark"}


def _positive_int(values: dict[str, Any], key: str, default: int) -> int:
    value = int(values.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be positive")
    return value


def _model_probe(task: dict[str, Any]) -> dict[str, Any]:
    # Probes are CPU deterministic by default. Callers can explicitly set this
    # environment variable to 0 before launch when validating an accelerated path.
    os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
    import torch

    architecture = str(task.get("architecture", ""))
    if architecture not in _MODEL_ARCHITECTURES:
        raise ValueError(
            f"unknown prototype architecture {architecture!r}; "
            f"expected one of {sorted(_MODEL_ARCHITECTURES)}"
        )

    model_cfg = dict(task.get("model", {}))
    vocab_size = _positive_int(model_cfg, "vocab_size", 256)
    d_model = _positive_int(model_cfg, "d_model", 16)
    n_layers = _positive_int(model_cfg, "n_layers", 2)
    head_size = _positive_int(model_cfg, "head_size", 16)
    batch = _positive_int(task, "batch", 1)
    sequence_length = _positive_int(task, "sequence_length", 8)
    seed = int(task.get("seed", 42))
    if d_model % head_size:
        raise ValueError("d_model must be divisible by head_size")
    if n_layers < 2:
        raise ValueError("n_layers must be at least 2 for RWKV time-mix initialization")

    common = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
    }
    if architecture == "rwkv8":
        from rwkv_lab.rwkv8_prototype import RWKV8LanguageModel

        model = RWKV8LanguageModel(
            **common,
            head_size=head_size,
            deepembed=bool(model_cfg.get("deepembed", True)),
        )
    elif architecture == "kan_rwkv":
        from rwkv_lab.kan_rwkv import KANRWKVLanguageModel

        model = KANRWKVLanguageModel(
            **common,
            head_size=head_size,
            deepembed=bool(model_cfg.get("deepembed", True)),
            grid_size=_positive_int(model_cfg, "grid_size", 5),
            spline_order=_positive_int(model_cfg, "spline_order", 3),
        )
    elif architecture == "blt_rwkv7":
        from rwkv_lab.toy_blt_train import BLTRWKV7LanguageModel

        model = BLTRWKV7LanguageModel(
            **common,
            threshold=float(model_cfg.get("threshold", 3.5)),
            max_patch=_positive_int(model_cfg, "max_patch", 16),
        )
    else:
        from rwkv_lab.rosa_blt_prototype import BLT_ROSA_LanguageModel

        model = BLT_ROSA_LanguageModel(
            **common,
            head_size=head_size,
            threshold=float(model_cfg.get("threshold", 3.5)),
            max_patch=_positive_int(model_cfg, "max_patch", 8),
            rosa_M=_positive_int(model_cfg, "rosa_M", 4),
        )

    torch.manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, sequence_length))
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        output = model(ids)
    elapsed = time.perf_counter() - started
    logits = output[0] if isinstance(output, tuple) else output
    expected_shape = (batch, sequence_length, vocab_size)
    if tuple(logits.shape) != expected_shape or not torch.isfinite(logits).all():
        raise RuntimeError(
            f"{architecture} produced invalid logits: shape={tuple(logits.shape)}, "
            f"finite={bool(torch.isfinite(logits).all())}"
        )
    return {
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "logits_shape": list(logits.shape),
        "latency_ms": elapsed * 1000,
        "logits_rms": math.sqrt(float(logits.float().square().mean())),
    }


def _run_task(task: dict[str, Any], report_path: Path) -> dict[str, Any]:
    kind = str(task.get("kind", ""))
    if kind not in _TASK_KINDS:
        raise ValueError(f"unknown prototype task {kind!r}; expected one of {sorted(_TASK_KINDS)}")
    if kind == "model_probe":
        return _model_probe(task)
    args = dict(task.get("args", {}))
    args["report_path"] = report_path
    if kind == "blt_comparison":
        from rwkv_lab.blt_validation import run_comparative_validation

        return run_comparative_validation(**args)
    from rwkv_lab.benchmark_standards import run_benchmark

    return run_benchmark(**args)


def run_prototype_suite(cfg: dict[str, Any]) -> Path:
    """Execute the ``prototype.tasks`` list and write one structured summary."""
    suite = cfg.get("prototype")
    if not isinstance(suite, dict):
        raise ValueError("prototype must be a mapping")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prototype.tasks must be a nonempty list")

    output_dir = Path(suite.get("output_dir", "runs/prototype-suite"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, dict):
            raise ValueError(f"prototype task {index} must be a mapping")
        name = str(raw_task.get("name", f"task-{index + 1}"))
        if not name or name in results or not all(ch.isalnum() or ch in "-_" for ch in name):
            raise ValueError(f"prototype task name must be unique and filesystem-safe: {name!r}")
        started = time.perf_counter()
        result = _run_task(raw_task, output_dir / f"{name}.json")
        results[name] = {
            "kind": raw_task.get("kind"),
            "elapsed_seconds": time.perf_counter() - started,
            "result": result,
        }

    summary_path = output_dir / "summary.json"
    summary = {"name": cfg.get("name", "prototype suite"), "tasks": results}
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[prototype] summary written to {summary_path}", flush=True)
    return summary_path
