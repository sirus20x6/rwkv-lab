"""TrainVM-native optimizer experiments on a pretrained BlinkDL RWKV-7."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import (
        WorkerControlRuntime,
        WorkerObservability,
        WorkerStepProfiler,
    )


RUN_SCHEMA = "rwkv-lab.rwkv-optimizer-finetune.v1"


@dataclass(frozen=True, slots=True)
class RWKVOptimizerFinetuneConfig:
    model_path: str
    data_path: str
    output_dir: str
    max_steps: int
    input_identity_digest: str
    subject: str
    resume_from: str | None = None
    sequence_length: int = 1024
    batch_size: int = 8
    validation_windows: int = 32
    eval_every: int = 50
    log_every: int = 5
    checkpoint_every: int = 250
    keep_last_n: int = 2
    seed: int = 0
    dtype: str = "bf16"

    def validate(self) -> None:
        if not Path(self.model_path).expanduser().is_file():
            raise ValueError("RWKV optimizer-finetune model path is unavailable")
        if not Path(self.data_path).expanduser().is_file():
            raise ValueError("RWKV optimizer-finetune token path is unavailable")
        for name in (
            "max_steps",
            "sequence_length",
            "batch_size",
            "validation_windows",
            "eval_every",
            "log_every",
            "checkpoint_every",
            "keep_last_n",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dtype != "bf16":
            raise ValueError("RWKV optimizer finetuning is qualified only for bf16")
        if (
            not isinstance(self.subject, str)
            or not self.subject.isascii()
            or not 0 < len(self.subject) <= 128
            or not self.subject[0].isalnum()
            or any(
                not (character.isalnum() or character in "._-")
                for character in self.subject
            )
        ):
            raise ValueError("subject must be a bounded ASCII identifier")
        if (
            len(self.input_identity_digest) != 71
            or not self.input_identity_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in self.input_identity_digest[7:]
            )
        ):
            raise ValueError("input_identity_digest must be a lowercase SHA-256")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _training_identity_digest(config: RWKVOptimizerFinetuneConfig) -> str:
    identity = {
        "input_identity_digest": config.input_identity_digest,
        "model_path": config.model_path,
        "data_path": config.data_path,
        "sequence_length": config.sequence_length,
        "batch_size": config.batch_size,
        "validation_windows": config.validation_windows,
        "seed": config.seed,
        "dtype": config.dtype,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _rng_state(generator: np.random.Generator) -> dict[str, Any]:
    return {
        "numpy": generator.bit_generator.state,
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(generator: np.random.Generator, state: Mapping[str, Any]) -> None:
    generator.bit_generator.state = state["numpy"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    return sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    generator: np.random.Generator,
    output_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    component_digest: str,
    training_identity_digest: str,
    routing_report: Mapping[str, Any],
    keep_last_n: int,
    worker_control_state: Mapping[str, Any] | None = None,
) -> Path:
    final = output_dir / f"checkpoint-{step:08d}"
    if final.is_dir():
        return final
    temporary = output_dir / f".checkpoint-{step:08d}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(model.state_dict(), temporary / "model.pt")
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(generator),
            "component_composition_digest": component_digest,
            "training_identity_digest": training_identity_digest,
            "parameter_routing": dict(routing_report),
            "worker_control_state": dict(worker_control_state or {}),
        },
        temporary / "trainer_state.pt",
    )
    _atomic_json(
        temporary / "checkpoint.json",
        {
            "schema": RUN_SCHEMA,
            "step": step,
            "tokens_seen": tokens_seen,
            "component_composition_digest": component_digest,
            "training_identity_digest": training_identity_digest,
            "created_at": _utc_now(),
        },
    )
    os.replace(temporary, final)
    checkpoints = _checkpoint_directories(output_dir)
    for stale in checkpoints[: max(0, len(checkpoints) - keep_last_n)]:
        shutil.rmtree(stale)
    return final


def _load_checkpoint(
    checkpoint: Path,
    model,
    optimizer,
    scheduler,
    generator: np.random.Generator,
    *,
    component_digest: str,
    training_identity_digest: str,
    routing_report: Mapping[str, Any],
    worker_controls: WorkerControlRuntime | None = None,
) -> tuple[int, int]:
    state = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    if (
        state.get("schema") != RUN_SCHEMA
        or state.get("component_composition_digest") != component_digest
        or state.get("training_identity_digest") != training_identity_digest
        or state.get("parameter_routing") != dict(routing_report)
    ):
        raise ValueError("RWKV optimizer-finetune checkpoint identity is incompatible")
    if worker_controls is not None:
        control_state = state.get("worker_control_state")
        if not isinstance(control_state, Mapping):
            raise ValueError("RWKV optimizer-finetune control state is missing")
        worker_controls.verify_checkpoint_state(control_state)
    model.load_state_dict(
        torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(generator, state["rng"])
    return int(state["step"]), int(state["tokens_seen"])


def _token_stream(path: Path, *, sequence_length: int, validation_windows: int):
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    sample = np.asarray(tokens[: min(len(tokens), 1 << 20)])
    if len(sample) >= 4 and not sample[1::2].any():
        raise ValueError("token stream looks like uint32 data read as uint16")
    validation_count = validation_windows * sequence_length
    if len(tokens) <= validation_count + sequence_length + 1:
        raise ValueError("token stream is too short for the declared split")
    validation = tokens[:validation_count]
    training = tokens[validation_count:]
    offsets = np.linspace(
        0,
        len(validation) - (sequence_length + 1),
        validation_windows,
    ).astype(np.int64)
    return validation, training, offsets


def train(
    config: RWKVOptimizerFinetuneConfig,
    *,
    worker_components: WorkerTrainingComponents,
    worker_step_profiler: WorkerStepProfiler | None = None,
    worker_observability: WorkerObservability | None = None,
    worker_controls: WorkerControlRuntime | None = None,
) -> Mapping[str, Any]:
    config.validate()
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    generator = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from rwkv_lab.rwkv_finetune import load_g1g_fla

    model = load_g1g_fla(
        Path(config.model_path).expanduser().resolve(),
        device=device,
        dtype=torch.bfloat16,
    )
    model.train()
    named_parameters = list(model.named_parameters())
    muon_ids = frozenset(
        id(parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
        and parameter.ndim == 2
        and "embeddings" not in name
        and "lm_head" not in name
        and "norm" not in name
    )
    optimizer_configuration = worker_components.configuration(
        "optimizer", category="optimizer"
    )
    base_learning_rate = float(optimizer_configuration["learning_rate"])
    routing = worker_components.parameter_routing(
        named_parameters,
        {"muon": muon_ids},
        base_learning_rate=base_learning_rate,
    )
    groups = [
        {
            **group,
            "use_muon": group["group_name"] == "muon",
        }
        for group in routing.groups
    ]
    optimizer = worker_components.optimizer(groups)
    scheduler = worker_components.learning_rate_schedule(optimizer)
    weight_decay = worker_components.weight_decay_schedule(optimizer)
    component_digest = worker_components.composition.composition_digest
    training_identity_digest = _training_identity_digest(config)
    validation, training, validation_offsets = _token_stream(
        Path(config.data_path).expanduser().resolve(),
        sequence_length=config.sequence_length,
        validation_windows=config.validation_windows,
    )
    step, tokens_seen = 0, 0
    if config.resume_from:
        step, tokens_seen = _load_checkpoint(
            Path(config.resume_from).expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            generator,
            component_digest=component_digest,
            training_identity_digest=training_identity_digest,
            routing_report=routing.report,
            worker_controls=worker_controls,
        )
    stopped = {"value": False}

    def handle_stop(_signal, _frame):
        stopped["value"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    sequence_length = config.sequence_length
    started = time.perf_counter()
    started_tokens = tokens_seen

    def reject_live_controls(_effective: object, assignments: object) -> None:
        if assignments:
            raise ValueError("RWKV optimizer-finetune v1 has no live controls")

    def batch(source, offsets):
        values = np.stack(
            [
                np.asarray(
                    source[offset : offset + sequence_length + 1],
                    dtype=np.int64,
                )
                for offset in offsets
            ]
        )
        return torch.from_numpy(values).to(device)

    def evaluate() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, config.validation_windows, config.batch_size):
                offsets = validation_offsets[start : start + config.batch_size]
                values = batch(validation, offsets)
                logits = model(values[:, :sequence_length]).logits.float()
                total += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    values[:, 1 : sequence_length + 1].reshape(-1),
                    reduction="sum",
                ).item()
                count += values.shape[0] * sequence_length
        model.train()
        return total / max(1, count)

    log_path = output_dir / "train.jsonl"
    _atomic_json(
        output_dir / "run_contract.json",
        {
            "schema": RUN_SCHEMA,
            "config": asdict(config),
            "component_composition": worker_components.evidence(),
            "component_composition_digest": component_digest,
            "training_identity_digest": training_identity_digest,
            "parameter_routing": routing.report,
        },
    )
    last_checkpoint = None
    final_validation_loss = None
    while step < config.max_steps and not stopped["value"]:
        if worker_controls is not None:
            worker_controls.microbatch(step + 1, reject_live_controls)
        step_started = time.perf_counter()
        offsets = generator.integers(
            0,
            len(training) - (sequence_length + 1),
            size=config.batch_size,
        )
        values = batch(training, offsets)
        logits = model(values[:, :sequence_length]).logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            values[:, 1 : sequence_length + 1].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = worker_components.gradient_clipping(model.parameters())
        weight_decay.step(step)
        optimizer.step()
        scheduler.step()
        step += 1
        tokens_seen += config.batch_size * sequence_length
        if worker_step_profiler is not None:
            worker_step_profiler.step(step)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        metrics = {
            "kind": "train",
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "step_seconds": time.perf_counter() - step_started,
            "tokens_per_second": (tokens_seen - started_tokens) / elapsed,
        }
        if step % config.log_every == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if worker_observability is not None:
            worker_observability.optimizer_step(step)
            worker_observability.publish_if_declared(
                "train.loss", metrics["loss"], step=step
            )
            worker_observability.publish_if_declared(
                "train.tokens_per_second",
                metrics["tokens_per_second"],
                step=step,
            )
            worker_observability.publish_if_declared(
                "train.gradient_norm", metrics["grad_norm"], step=step
            )
            worker_observability.publish_if_declared(
                "train.learning_rate", metrics["learning_rate"], step=step
            )
            worker_observability.publish_if_declared(
                "train.step_seconds", metrics["step_seconds"], step=step
            )
            worker_observability.publish_if_declared(
                "system.gpu_memory_used",
                int(torch.cuda.memory_allocated(device)),
                step=step,
            )
        if worker_controls is not None:
            worker_controls.optimizer_step(step, reject_live_controls)
        if step % config.eval_every == 0 or step == config.max_steps:
            if worker_controls is not None:
                worker_controls.evaluation(step, reject_live_controls)
            validation_loss = evaluate()
            final_validation_loss = validation_loss
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "eval",
                            "step": step,
                            "loss": validation_loss,
                            "ppl": math.exp(min(validation_loss, 20.0)),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if worker_observability is not None:
                worker_observability.publish_if_declared(
                    "eval.loss", validation_loss, step=step
                )
                worker_observability.publish_if_declared(
                    "eval.perplexity",
                    math.exp(min(validation_loss, 20.0)),
                    step=step,
                )
        checkpoint_requested = bool(
            worker_controls is not None
            and worker_controls.checkpoint_boundary_requested
        )
        if checkpoint_requested:
            worker_controls.checkpoint(step, reject_live_controls)
        if step % config.checkpoint_every == 0 or checkpoint_requested:
            last_checkpoint = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                generator,
                output_dir,
                step=step,
                tokens_seen=tokens_seen,
                component_digest=component_digest,
                training_identity_digest=training_identity_digest,
                routing_report=routing.report,
                keep_last_n=config.keep_last_n,
                worker_control_state=(
                    worker_controls.checkpoint_state()
                    if worker_controls is not None
                    else None
                ),
            )
            if (
                checkpoint_requested
                and worker_controls is not None
                and worker_controls.checkpoint_completion_requested
            ):
                worker_controls.publish_requested_checkpoint_directory(
                    str(last_checkpoint),
                    optimizer_step=step,
                    resume_grade="compatible",
                    state_components=(
                        "component_composition",
                        "control_revision",
                        "data_cursor",
                        "input_content_identity",
                        "learning_rate_schedule",
                        "model",
                        "optimizer",
                        "parameter_routing",
                        "rng_accelerator",
                        "rng_numpy",
                        "rng_python",
                        "rng_torch",
                    ),
                )
    last_checkpoint = _save_checkpoint(
        model,
        optimizer,
        scheduler,
        generator,
        output_dir,
        step=step,
        tokens_seen=tokens_seen,
        component_digest=component_digest,
        training_identity_digest=training_identity_digest,
        routing_report=routing.report,
        keep_last_n=config.keep_last_n,
        worker_control_state=(
            worker_controls.checkpoint_state()
            if worker_controls is not None
            else None
        ),
    )
    if not stopped["value"] and final_validation_loss is None:
        final_validation_loss = evaluate()
    terminal: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "state": "interrupted" if stopped["value"] else "complete",
        "step": step,
        "tokens_seen": tokens_seen,
        "checkpoint": str(last_checkpoint),
        "updated_at": _utc_now(),
    }
    if final_validation_loss is not None:
        terminal["eval_loss"] = float(final_validation_loss)
    _atomic_json(
        output_dir / ("interrupted.json" if stopped["value"] else "complete.json"),
        terminal,
    )
    return terminal


__all__ = ["RUN_SCHEMA", "RWKVOptimizerFinetuneConfig", "train"]
