#!/usr/bin/env python3
"""Materialize the three real experiments used by the production parity gate.

The production qualification runner intentionally accepts complete declarative
experiments rather than trainer paths or argv.  This authoring helper turns one
strict deployment-input document into the MageFlow, RWKV, and transformer
experiments plus the exact root sets consumed by ``trainvm lock-input-content``.
It does not hash inputs, compile plans, launch workers, or touch an accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

API_VERSION = "trainvm.production-qualification-inputs/v1"
MATERIALIZATION_VERSION = "trainvm.production-qualification-materialization/v1"
FAMILIES = ("mageflow", "rwkv", "transformer")


class MaterializationError(ValueError):
    """Deployment inputs cannot produce an authoritative qualification set."""


def _reject_constant(value: str) -> None:
    raise MaterializationError(f"non-finite JSON number is forbidden: {value}")


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise MaterializationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_inputs(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise MaterializationError("input document must not be a symlink")
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read deployment inputs: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError("deployment inputs must be a JSON object")
    return value


def _closed(mapping: Mapping[str, Any], required: set[str], optional: set[str], label: str) -> None:
    missing = sorted(required - mapping.keys())
    unknown = sorted(mapping.keys() - required - optional)
    if missing:
        raise MaterializationError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise MaterializationError(f"{label} has unknown fields: {', '.join(unknown)}")


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MaterializationError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise MaterializationError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or re.fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", value) is None
    ):
        raise MaterializationError(f"{label} must be a TrainVM identifier")
    return value


def _read_path(value: Any, label: str, kind: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MaterializationError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise MaterializationError(f"{label} must be an absolute normalized path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise MaterializationError(f"{label} is unavailable: {error}") from error
    if kind == "file" and not resolved.is_file():
        raise MaterializationError(f"{label} must be a file")
    if kind == "directory" and not resolved.is_dir():
        raise MaterializationError(f"{label} must be a directory")
    return str(resolved)


def _output_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MaterializationError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)) or path == Path("/"):
        raise MaterializationError(f"{label} must be a specific absolute normalized path")
    ancestor = path
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise MaterializationError(f"{label} has no existing ancestor")
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise MaterializationError(f"{label} ancestor is not a directory")
    return str(path.resolve(strict=False))


def _path_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise MaterializationError(f"{label} must be a nonempty bounded path list")
    paths = [_read_path(item, f"{label}[{index}]", "directory") for index, item in enumerate(value)]
    if len(paths) != len(set(paths)):
        raise MaterializationError(f"{label} contains duplicate canonical paths")
    return paths


def _artifact(kind: str, schema: str, *, fingerprint: str = "manifest_sha256") -> dict[str, Any]:
    return {
        "type": kind,
        "schema": schema,
        "immutability": "immutable" if kind != "image_gallery" else "append_only",
        "fingerprint": fingerprint,
        "required": True,
    }


def _component(category: str, name: str, version: str, configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": {"category": category, "name": name, "version": version},
        "configuration": dict(configuration),
    }


def _metric(name: str, aggregation: str = "last", unit: str = "dimensionless") -> dict[str, Any]:
    return {
        "name": name,
        "type": "gauge",
        "unit": unit,
        "step_domain": "optimizer_step",
        "aggregation": aggregation,
    }


def _resources(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise MaterializationError("resources must be an object")
    _closed(
        raw,
        set(),
        {
            "minimum_accelerator_memory_gib",
            "minimum_host_memory_gib",
            "cpu_threads",
            "omp_threads",
            "preprocessing_workers",
        },
        "resources",
    )
    cpu_threads = _integer(raw.get("cpu_threads", 8), "resources.cpu_threads", 1, 65536)
    omp_threads = _integer(raw.get("omp_threads", min(4, cpu_threads)), "resources.omp_threads", 1, 65536)
    return {
        "accelerators": {
            "vendor": "nvidia",
            "count": 1,
            "minimum_memory_gib": _integer(
                raw.get("minimum_accelerator_memory_gib", 24),
                "resources.minimum_accelerator_memory_gib",
                1,
                1048576,
            ),
            "exclusive": True,
        },
        "minimum_host_memory_gib": _integer(
            raw.get("minimum_host_memory_gib", 32),
            "resources.minimum_host_memory_gib",
            1,
            1048576,
        ),
        "cpu_threads": cpu_threads,
        "lease_timeout_seconds": 30,
        "cpu_io_policy": {
            "cpu_weight": 100,
            "io_weight": 100,
            "omp_threads": omp_threads,
            "preprocessing_workers": _integer(
                raw.get("preprocessing_workers", min(2, cpu_threads)),
                "resources.preprocessing_workers",
                0,
                65536,
            ),
            "nice": 0,
        },
    }


def _read_roots(workspace_root: str, roots: Sequence[str]) -> list[str]:
    candidates = {Path(workspace_root)}
    for value in roots:
        path = Path(value)
        candidates.add(path if path.is_dir() else path.parent)
    ordered = sorted(candidates, key=lambda item: (len(item.parts), str(item)))
    selected: list[Path] = []
    for candidate in ordered:
        if not any(candidate == parent or parent in candidate.parents for parent in selected):
            selected.append(candidate)
    return sorted(str(path) for path in selected)


def _execution() -> dict[str, Any]:
    return {
        "component": "trainer",
        "operation": "train",
        "compile": {"enabled": False},
        "gpu_trace": {
            "enabled": True,
            "backend": "torch",
            "warmup_steps": 1,
            "skip_steps": 0,
            "capture_steps": 2,
            "output_artifact": "gpu_trace",
            "activities": ["cpu", "accelerator"],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
        },
    }


def _base_document(
    *,
    family: str,
    workspace_root: str,
    run_directory: str,
    concurrency_key: str,
    read_roots: Sequence[str],
    resources: Mapping[str, Any],
    adapter: str,
    contract: str,
    config: Mapping[str, Any],
    training_components: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    publishes: Mapping[str, str],
    metrics: Sequence[Mapping[str, Any]],
    gallery_artifact: str | None = None,
) -> dict[str, Any]:
    observability: dict[str, Any] = {
        "heartbeat_seconds": 2,
        "metrics": list(metrics),
        "retain_raw_metrics_days": 30,
    }
    if gallery_artifact is not None:
        observability["eval_gallery_artifact"] = gallery_artifact
    return {
        "api_version": "trainvm.rwkv-lab/v1alpha1",
        "kind": "Experiment",
        "metadata": {
            "name": f"production-qualification-{family}",
            "description": f"Bounded real-{family} pause, release, checkpoint-resume, telemetry, and accelerator-trace qualification.",
            "labels": {"family": family, "purpose": "production-qualification"},
        },
        "spec": {
            "workspace": {
                "root": workspace_root,
                "run_directory": run_directory,
                "concurrency_key": concurrency_key,
                "allowed_read_roots": list(read_roots),
                "allowed_write_roots": [run_directory],
            },
            "resources": dict(resources),
            "execution": _execution(),
            "parameters": {},
            "artifacts": dict(artifacts),
            "components": {
                "core": {
                    "adapter": "trainvm.core",
                    "version": "1.0.0",
                    "runtime": "builtin",
                    "operations": {
                        "acquire_resources": {"contract": "trainvm.v1.AcquireResources"},
                        "release_resources": {"contract": "trainvm.v1.ReleaseResources"},
                    },
                },
                "trainer": {
                    "adapter": adapter,
                    "version": "1.0.0",
                    "runtime": "python_worker",
                    "operations": {"train": {"contract": contract}},
                },
            },
            "workflow": {
                "entrypoint": "acquire_gpu",
                "nodes": {
                    "acquire_gpu": {
                        "invoke": {
                            "component": "core",
                            "operation": "acquire_resources",
                            "inputs": {"concurrency_key": {"literal": concurrency_key}},
                        },
                        "idempotency": "receipt_required",
                        "effect": "resource",
                        "transitions": [
                            {"on": "resource.acquired", "target": "train"},
                            {"on": "operation.failed", "target": "$failed"},
                        ],
                    },
                    "train": {
                        "invoke": {
                            "component": "trainer",
                            "operation": "train",
                            "inputs": {"config": {"literal": dict(config)}},
                            "training": {
                                "model_family": family,
                                "components": dict(training_components),
                            },
                        },
                        "publishes": dict(publishes),
                        "idempotency": "receipt_required",
                        "effect": "process",
                        "timeout_seconds": 86400,
                        "transitions": [
                            {"on": "worker.completed", "target": "release_gpu"},
                            {"on": "operation.failed", "target": "$failed"},
                        ],
                    },
                    "release_gpu": {
                        "invoke": {
                            "component": "core",
                            "operation": "release_resources",
                            "inputs": {"concurrency_key": {"literal": concurrency_key}},
                        },
                        "idempotency": "replay_safe",
                        "effect": "resource",
                        "transitions": [
                            {"on": "resource.released", "target": "$completed"},
                            {"on": "operation.failed", "target": "$failed"},
                        ],
                    },
                },
            },
            "controls": {"catalog": {}},
            "observability": observability,
            "recovery": {
                "exact_resume": False,
                "checkpoint_artifact": "checkpoint",
                "reconcile": "restart_from_checkpoint",
                "orphan_policy": "leave_and_block",
                "graceful_stop_seconds": 300,
                "release_accelerators_when_paused": True,
            },
        },
    }


def _mageflow(raw: Any, common: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(raw, Mapping):
        raise MaterializationError("mageflow must be an object")
    _closed(
        raw,
        {"model_path", "train_manifest", "eval_manifest", "image_roots"},
        {"max_steps", "packed_sequence_tokens", "eval_gen_samples", "eval_gen_steps", "learning_rate"},
        "mageflow",
    )
    model = _read_path(raw["model_path"], "mageflow.model_path", "directory")
    train = _read_path(raw["train_manifest"], "mageflow.train_manifest", "file")
    evaluate = _read_path(raw["eval_manifest"], "mageflow.eval_manifest", "file")
    images = _path_list(raw["image_roots"], "mageflow.image_roots")
    max_steps = _integer(raw.get("max_steps", 8), "mageflow.max_steps", 5, 10000)
    learning_rate = _number(raw.get("learning_rate", 1.0e-5), "mageflow.learning_rate", 1.0e-12, 1.0)
    config = {
        "train_manifest": train,
        "eval_manifest": evaluate,
        "model_path": model,
        "output_dir": common["run_directory"],
        "max_steps": max_steps,
        "packed_sequence_tokens": _integer(raw.get("packed_sequence_tokens", 4096), "mageflow.packed_sequence_tokens", 1, 1048576),
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "min_learning_rate_ratio": 0.1,
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "caption_dropout": 0.1,
        "checkpoint_every": 1,
        "eval_every": 1,
        "eval_packs": 1,
        "eval_gen_every": 1,
        "eval_gen_samples": _integer(raw.get("eval_gen_samples", 2), "mageflow.eval_gen_samples", 1, 8),
        "eval_gen_steps": _integer(raw.get("eval_gen_steps", 4), "mageflow.eval_gen_steps", 1, 100),
        "eval_gen_cfg": 5.0,
        "eval_gen_step_zero": False,
        "eval_gen_screen_prompts": True,
        "keep_last_n": 3,
        "prefetch_packs": 1,
        "seed": 1729,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "compile_vae_encoder": False,
        "compile_transformer_blocks": False,
    }
    components = {
        "gradient_clipping": _component("gradient_clipping", "global_norm", "1.0.0", {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}),
        "learning_rate": _component("learning_rate_schedule", "linear_warmup_cosine", "1.0.0", {"warmup_steps": 1, "max_steps": max_steps, "minimum_ratio": 0.1}),
        "optimizer": _component("optimizer", "torch_adamw_no_decay", "2.0.0", {"learning_rate": learning_rate, "beta1": 0.9, "beta2": 0.95, "epsilon": 1.0e-8, "foreach": True, "fused": False}),
        "parameter_router": _component("parameter_router", "mageflow_full_backbone", "1.0.0", {}),
        "weight_decay": _component("weight_decay_schedule", "constant", "1.0.0", {"weight_decay": 0.01}),
    }
    document = _base_document(
        family="mageflow",
        adapter="rwkv-lab.mageflow-full-backbone",
        contract="rwkv_lab.mageflow_full_backbone.v1.Train",
        config=config,
        training_components=components,
        artifacts={
            "checkpoint": _artifact("checkpoint", "rwkv-lab.mageflow-checkpoint.v1"),
            "eval_gallery": _artifact("image_gallery", "rwkv-lab.eval-gallery.v2"),
            "gpu_trace": _artifact("opaque", "trainvm.gpu-trace.v1", fingerprint="adapter"),
        },
        publishes={"checkpoint": "checkpoint", "eval_gallery": "eval_gallery"},
        metrics=[
            _metric("train.loss", "weighted_mean"),
            _metric("train.images_per_second", "mean", "image/second"),
            _metric("eval.loss"),
            _metric("system.gpu_memory_used", "max", "byte"),
        ],
        gallery_artifact="eval_gallery",
        **common,
    )
    return document, [model, train, evaluate, *images], 2


def _rwkv(raw: Any, common: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(raw, Mapping):
        raise MaterializationError("rwkv must be an object")
    _closed(
        raw,
        {"model_path", "data_path"},
        {"max_steps", "sequence_length", "batch_size", "validation_windows", "learning_rate"},
        "rwkv",
    )
    model = _read_path(raw["model_path"], "rwkv.model_path", "file")
    data = _read_path(raw["data_path"], "rwkv.data_path", "file")
    max_steps = _integer(raw.get("max_steps", 8), "rwkv.max_steps", 5, 10000)
    sequence_length = _integer(raw.get("sequence_length", 128), "rwkv.sequence_length", 16, 1048576)
    batch_size = _integer(raw.get("batch_size", 1), "rwkv.batch_size", 1, 1024)
    validation_windows = _integer(raw.get("validation_windows", 2), "rwkv.validation_windows", 1, 1024)
    learning_rate = _number(raw.get("learning_rate", 1.0e-4), "rwkv.learning_rate", 1.0e-12, 1.0)
    config = {
        "model_path": model,
        "data_path": data,
        "max_steps": max_steps,
        "subject": "production_qualification",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "validation_windows": validation_windows,
        "eval_every": 1,
        "log_every": 1,
        "checkpoint_every": 1,
        "keep_last_n": 3,
        "seed": 1729,
        "dtype": "bf16",
    }
    components = {
        "gradient_clipping": _component("gradient_clipping", "global_norm", "1.0.0", {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}),
        "learning_rate": _component("learning_rate_schedule", "constant", "1.0.0", {}),
        "optimizer": _component("optimizer", "torch_adamw_no_decay", "2.0.0", {"learning_rate": learning_rate, "beta1": 0.9, "beta2": 0.95, "epsilon": 1.0e-8, "foreach": True, "fused": False}),
        "parameter_router": _component("parameter_router", "rwkv_matrix_optimizer", "1.0.0", {"fallback_multiplier": 0.1}),
        "weight_decay": _component("weight_decay_schedule", "constant", "1.0.0", {"weight_decay": 0.0}),
    }
    document = _base_document(
        family="rwkv",
        adapter="rwkv-lab.rwkv-optimizer-finetune",
        contract="rwkv_lab.rwkv_optimizer_finetune.v1.Train",
        config=config,
        training_components=components,
        artifacts={
            "checkpoint": _artifact("checkpoint", "rwkv-lab.rwkv-optimizer-finetune-checkpoint.v1"),
            "result": _artifact("report", "rwkv-lab.scalar-metric-result.v1"),
            "gpu_trace": _artifact("opaque", "trainvm.gpu-trace.v1", fingerprint="adapter"),
        },
        publishes={"checkpoint": "checkpoint", "result": "result"},
        metrics=[
            _metric("train.loss", "mean"),
            _metric("train.tokens_per_second", "mean", "token/second"),
            _metric("train.learning_rate", "last", "ratio"),
            _metric("eval.loss"),
            _metric("eval.perplexity"),
            _metric("system.gpu_memory_used", "max", "byte"),
        ],
        **common,
    )
    return document, [model, data], 2


def _transformer(raw: Any, common: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(raw, Mapping):
        raise MaterializationError("transformer must be an object")
    _closed(
        raw,
        {"model_dir", "patch_dir", "tokens_bin", "total_tokens_in_bin"},
        {"max_steps", "sequence_length", "eval_tokens", "learning_rate", "minimum_learning_rate"},
        "transformer",
    )
    model = _read_path(raw["model_dir"], "transformer.model_dir", "directory")
    patch = _read_path(raw["patch_dir"], "transformer.patch_dir", "directory")
    tokens = _read_path(raw["tokens_bin"], "transformer.tokens_bin", "file")
    total_tokens = _integer(raw["total_tokens_in_bin"], "transformer.total_tokens_in_bin", 2, (1 << 63) - 1)
    max_steps = _integer(raw.get("max_steps", 8), "transformer.max_steps", 5, 10000)
    sequence_length = _integer(raw.get("sequence_length", 128), "transformer.sequence_length", 16, 1048576)
    eval_tokens = _integer(raw.get("eval_tokens", max(256, sequence_length * 2)), "transformer.eval_tokens", 1, total_tokens - 1)
    learning_rate = _number(raw.get("learning_rate", 1.0e-4), "transformer.learning_rate", 1.0e-12, 1.0)
    minimum_lr = _number(raw.get("minimum_learning_rate", learning_rate * 0.1), "transformer.minimum_learning_rate", 0.0, learning_rate)
    minimum_ratio = minimum_lr / learning_rate
    config = {
        "profile": "mla",
        "model_dir": model,
        "patch_dir": patch,
        "tokens_bin": tokens,
        "output_dir": common["run_directory"],
        "total_tokens_in_bin": total_tokens,
        "eval_tokens": eval_tokens,
        "max_steps": max_steps,
        "sequence_length": sequence_length,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "minimum_learning_rate": minimum_lr,
        "warmup_steps": 1,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
        "log_every_steps": 1,
        "eval_every_steps": 1,
        "eval_batches": 1,
        "save_every_steps": 1,
        "seed": 1729,
    }
    components = {
        "gradient_accumulation": _component("gradient_accumulation", "fixed", "1.0.0", {"microbatches_per_optimizer_step": 1}),
        "gradient_clipping": _component("gradient_clipping", "global_norm", "1.0.0", {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}),
        "learning_rate": _component("learning_rate_schedule", "linear_warmup_cosine", "1.0.0", {"warmup_steps": 1, "max_steps": max_steps, "minimum_ratio": minimum_ratio}),
        "objective": _component("objective", "linear_head_cross_entropy", "1.0.0", {"chunk_size": 2048, "prefer_fused": True}),
        "optimizer": _component("optimizer", "torch_adamw", "1.0.0", {"learning_rate": learning_rate, "beta1": 0.9, "beta2": 0.95, "epsilon": 1.0e-8, "weight_decay": 0.0, "foreach": True, "fused": False}),
        "precision": _component("precision", "bf16_parameters_fp32_reductions", "1.0.0", {}),
        "weight_decay": _component("weight_decay_schedule", "constant", "1.0.0", {"weight_decay": 0.0}),
    }
    document = _base_document(
        family="transformer",
        adapter="rwkv-lab.transformer-mla",
        contract="rwkv_lab.transformer_mla.v1.Train",
        config=config,
        training_components=components,
        artifacts={
            "checkpoint": _artifact("checkpoint", "rwkv-lab.transformer-mla-checkpoint.v1"),
            "gpu_trace": _artifact("opaque", "trainvm.gpu-trace.v1", fingerprint="adapter"),
        },
        publishes={"checkpoint": "checkpoint"},
        metrics=[
            _metric("train.loss", "mean"),
            _metric("train.tokens_per_second", "mean", "token/second"),
            _metric("train.learning_rate", "last", "ratio"),
            _metric("eval.loss"),
            _metric("eval.perplexity"),
            _metric("system.gpu_memory_used", "max", "byte"),
        ],
        **common,
    )
    return document, [model, patch, tokens], 2


def build_documents(inputs: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], list[str], int]]:
    _closed(
        inputs,
        {"api_version", "workspace_root", "run_root", "concurrency_key", *FAMILIES},
        {"resources"},
        "deployment inputs",
    )
    if inputs["api_version"] != API_VERSION:
        raise MaterializationError("unsupported deployment-input API version")
    workspace_root = _read_path(inputs["workspace_root"], "workspace_root", "directory")
    run_root = _output_path(inputs["run_root"], "run_root")
    if Path(run_root) == Path(workspace_root):
        raise MaterializationError("run_root must not replace workspace_root")
    concurrency_key = _identifier(inputs["concurrency_key"], "concurrency_key")
    resources = _resources(inputs.get("resources"))
    builders = {"mageflow": _mageflow, "rwkv": _rwkv, "transformer": _transformer}
    provisional: dict[str, tuple[dict[str, Any], list[str], int]] = {}
    for family in FAMILIES:
        run_directory = str(Path(run_root) / family)
        common = {
            "workspace_root": workspace_root,
            "run_directory": run_directory,
            "concurrency_key": concurrency_key,
            "read_roots": [],
            "resources": resources,
        }
        document, roots, pause_step = builders[family](inputs[family], common)
        document["spec"]["workspace"]["allowed_read_roots"] = _read_roots(workspace_root, roots)
        provisional[family] = document, sorted(set(roots)), pause_step
    return provisional


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def materialize(inputs: Mapping[str, Any], destination: Path) -> Path:
    documents = build_documents(inputs)
    if destination.exists() or destination.is_symlink():
        raise MaterializationError(f"destination already exists: {destination}")
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise MaterializationError("destination parent is not a directory")
    staging = parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise MaterializationError(f"staging path already exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        files: dict[str, dict[str, str]] = {}
        pause_steps: dict[str, int] = {}
        for family, (document, roots, pause_step) in documents.items():
            document_name = f"{family}.json"
            roots_name = f"{family}.input-roots.json"
            document_bytes = _json_bytes(document)
            roots_bytes = _json_bytes({"api_version": "trainvm.input-content-root-set/v1", "paths": roots})
            (staging / document_name).write_bytes(document_bytes)
            (staging / roots_name).write_bytes(roots_bytes)
            files[family] = {
                "document": document_name,
                "document_sha256": hashlib.sha256(document_bytes).hexdigest(),
                "input_roots": roots_name,
                "input_roots_sha256": hashlib.sha256(roots_bytes).hexdigest(),
                "locked_document": f"{family}.locked.json",
            }
            pause_steps[family] = pause_step
        manifest = {
            "api_version": MATERIALIZATION_VERSION,
            "files": files,
            "pause_steps": pause_steps,
            "next_actions": [
                "Run trainvm lock-input-content for each document/root-set pair.",
                "Submit only the resulting *.locked.json files to run_trainvm_production_qualification.py.",
            ],
        }
        (staging / "materialization.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "materialization.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="strict deployment-input JSON")
    parser.add_argument("destination", type=Path, help="new output directory")
    arguments = parser.parse_args(argv)
    try:
        inputs = load_inputs(arguments.inputs)
        manifest = materialize(inputs, arguments.destination)
    except MaterializationError as error:
        print(f"qualification materialization rejected: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
