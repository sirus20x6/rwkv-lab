"""Closed typed configuration for the baseline scratch-RWKV TrainVM adapter."""

from __future__ import annotations

import math
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rwkv_lab.training_runtime.model_loaders import RWKVModelFactory
from rwkv_lab.training_runtime.schedules import (
    LinearWarmupCosineConfiguration,
    PowerCoolConfiguration,
    ScheduleImplementation,
)

from .components import WorkerTrainingComponents


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: object, label: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class RWKVScratchTrainConfig:
    """Intentionally small v1 surface for reproducible native scratch training.

    Research levers remain separate adapter versions until their topology,
    optimizer state, and checkpoint identity are represented declaratively.
    """

    steps: int
    d_model: int = 512
    n_layers: int = 6
    head_size: int = 64
    sequence_length: int = 512
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6.0e-4
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    warmup_steps: int = 100
    minimum_learning_rate: float = 0.0
    cooldown_fraction: float = 0.2
    cooldown_power: float = 2.0
    validation_windows: int = 40
    eval_every_steps: int = 50
    log_every_steps: int = 10
    seed: int = 0
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.resume is not None and (
            not isinstance(self.resume, str) or not self.resume
        ):
            raise ValueError("resume must be a nonempty path when present")
        _integer(self.steps, "steps", 1, 1_000_000_000)
        _integer(self.d_model, "d_model", 64, 65_536)
        _integer(self.n_layers, "n_layers", 1, 4_096)
        _integer(self.head_size, "head_size", 16, 1_024)
        if self.d_model % self.head_size:
            raise ValueError("d_model must be divisible by head_size")
        _integer(self.sequence_length, "sequence_length", 2, 1_048_576)
        _integer(self.batch_size, "batch_size", 1, 1_048_576)
        _integer(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
            1,
            65_536,
        )
        learning_rate = _finite(
            self.learning_rate, "learning_rate", 1.0e-12, 1.0
        )
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e9)
        _integer(self.warmup_steps, "warmup_steps", 0, self.steps)
        minimum = _finite(
            self.minimum_learning_rate,
            "minimum_learning_rate",
            0.0,
            learning_rate,
        )
        if minimum > learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        cooldown = _finite(
            self.cooldown_fraction, "cooldown_fraction", 0.0, 1.0
        )
        if cooldown == 0.0:
            raise ValueError("cooldown_fraction must be positive")
        _finite(self.cooldown_power, "cooldown_power", 1.0e-12, 64.0)
        _integer(self.validation_windows, "validation_windows", 1, 1_000_000)
        _integer(self.eval_every_steps, "eval_every_steps", 1, self.steps)
        _integer(self.log_every_steps, "log_every_steps", 1, self.steps)
        _integer(self.seed, "seed", 0, (1 << 63) - 1)

    @classmethod
    def from_components(
        cls, components: WorkerTrainingComponents
    ) -> RWKVScratchTrainConfig:
        model = components.model_loader()
        if not isinstance(model, RWKVModelFactory):
            raise TypeError("scratch-RWKV requires a registered RWKV model factory")
        if model.configuration.vocabulary_size != 65_536:
            raise ValueError("scratch-RWKV currently requires the 65536-token vocabulary")
        _, schedule = components.learning_rate_configuration()
        if not isinstance(
            schedule, (LinearWarmupCosineConfiguration, PowerCoolConfiguration)
        ):
            raise TypeError(
                "scratch-RWKV requires a finite-horizon cosine or PowerCool schedule"
            )
        optimizer = dict(components.configuration("optimizer", category="optimizer"))
        weight_decay = dict(
            components.configuration("weight_decay", category="weight_decay_schedule")
        )
        clipping = dict(
            components.configuration("gradient_clipping", category="gradient_clipping")
        )
        accumulation = components.gradient_accumulation()
        curriculum = components.curriculum()
        evaluator = components.evaluator()
        evaluation = components.evaluation_schedule()
        if evaluator.configuration.maximum_examples < 1:
            raise ValueError("scratch-RWKV requires a bounded validation window count")
        if evaluation.configuration.full_every_steps < 1:
            raise ValueError("scratch-RWKV requires periodic full scalar evaluation")
        if evaluation.configuration.qualitative_every_steps < 1:
            raise ValueError("scratch-RWKV requires periodic qualitative evaluation")
        if evaluation.configuration.full_every_steps != (
            evaluation.configuration.qualitative_every_steps
        ):
            raise ValueError(
                "scratch-RWKV currently evaluates scalar and text evidence together"
            )
        renderer = components.artifact_renderer()
        if renderer.configuration.modality != "text":
            raise ValueError("scratch-RWKV evaluation evidence must use text modality")
        components.generation_policy()
        checkpoint_policy = components.checkpoint_policy()
        if not checkpoint_policy.configuration.publish_final:
            raise ValueError("scratch-RWKV requires a final checkpoint publication")
        sampler = dict(components.configuration("sampler", category="sampler"))
        batching = dict(components.configuration("batching", category="batching"))
        collation = dict(components.configuration("collation", category="collator"))
        processor = dict(
            components.configuration("processor", category="sample_processor")
        )
        if sampler["seed"] != model.configuration.seed:
            raise ValueError("RWKV model and data sampler seeds must agree")
        if batching["batch_size"] != curriculum.configuration.base_batch_size:
            raise ValueError("RWKV batching and curriculum batch sizes must agree")
        if (
            collation["maximum_sequence_length"]
            != curriculum.configuration.maximum_sequence_length
        ):
            raise ValueError("RWKV collation and curriculum context limits must agree")
        if processor["vocabulary_size"] != model.configuration.vocabulary_size:
            raise ValueError("RWKV processor and model vocabularies must agree")
        minimum_ratio = schedule.minimum_ratio
        return cls(
            steps=schedule.max_steps,
            d_model=model.configuration.d_model,
            n_layers=model.configuration.n_layers,
            head_size=model.configuration.head_size,
            sequence_length=curriculum.configuration.maximum_sequence_length,
            batch_size=curriculum.configuration.base_batch_size,
            gradient_accumulation_steps=(
                accumulation.microbatches_per_optimizer_step
            ),
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(weight_decay["weight_decay"]),
            max_gradient_norm=float(clipping["max_norm"]),
            warmup_steps=schedule.warmup_steps,
            minimum_learning_rate=float(optimizer["learning_rate"])
            * minimum_ratio,
            cooldown_fraction=(
                schedule.cooldown_fraction
                if isinstance(schedule, PowerCoolConfiguration)
                else 1.0
            ),
            cooldown_power=(
                schedule.power if isinstance(schedule, PowerCoolConfiguration) else 1.0
            ),
            validation_windows=evaluator.configuration.maximum_examples,
            eval_every_steps=evaluation.configuration.full_every_steps,
            log_every_steps=min(10, evaluation.configuration.full_every_steps),
            seed=model.configuration.seed,
            resume=(
                model.configuration.checkpoint_path if model.continuation else None
            ),
        )

    def trainer_arguments(
        self,
        *,
        data: str,
        output_dir: str,
        checkpoint: str,
        resume: str | None,
        schedule: ScheduleImplementation = ScheduleImplementation.POWERCOOL_V1,
    ) -> list[str]:
        values = {
            "--data": data,
            "--out": output_dir,
            "--save": checkpoint,
            "--steps": self.steps,
            "--d-model": self.d_model,
            "--n-layers": self.n_layers,
            "--head-size": self.head_size,
            "--seq-len": self.sequence_length,
            "--batch": self.batch_size,
            "--grad-accum": self.gradient_accumulation_steps,
            "--lr": self.learning_rate,
            "--weight-decay": self.weight_decay,
            "--grad-clip": self.max_gradient_norm,
            "--warmup": self.warmup_steps,
            "--powercool-min-lr": self.minimum_learning_rate,
            "--cosine-min-ratio": (
                self.minimum_learning_rate / self.learning_rate
            ),
            "--powercool-cooldown-fraction": self.cooldown_fraction,
            "--powercool-power": self.cooldown_power,
            "--val-windows": self.validation_windows,
            "--eval-every": self.eval_every_steps,
            "--log-every": self.log_every_steps,
            "--seed": self.seed,
        }
        arguments: list[str] = []
        for flag, value in values.items():
            arguments.extend((flag, str(value)))
        schedule_name = {
            ScheduleImplementation.LINEAR_WARMUP_COSINE_V1: "cosine",
            ScheduleImplementation.POWERCOOL_V1: "powercool",
        }.get(schedule)
        if schedule_name is None:
            raise ValueError("scratch-RWKV schedule is not lowerable")
        arguments.extend(("--optimizer", "adamw"))
        arguments.extend(("--lr-schedule", schedule_name))
        arguments.extend(("--distributed", "none"))
        if resume is not None:
            arguments.extend(("--resume", resume))
        return arguments


@dataclass(frozen=True, slots=True)
class PreparedRWKVCorpus:
    path: Path
    heldout_tokens: Mapping[str, tuple[int, ...]]
    identities_digest: str
    selector_digest: str


def prepare_registered_corpus(
    components: WorkerTrainingComponents,
    destination: Path,
    *,
    authority_content_fingerprint: str,
    validation_windows: int,
    sequence_length: int,
) -> PreparedRWKVCorpus:
    training = components.data_pipeline(split_slot="split")
    evaluation = components.data_pipeline(split_slot="evaluation_split")
    training.validate_schema()
    evaluation.validate_schema()
    training.source.verify_content(
        authority_content_fingerprint=authority_content_fingerprint
    )
    validation_rows = evaluation.source.records_for_split("validation")
    training_rows = training.source.records_for_split("train")
    qualitative = components.qualitative_samples()
    sample_count = qualitative.configuration.sample_count
    if len(validation_rows) < sample_count:
        raise ValueError("validation split is smaller than the qualitative policy")
    selection = evaluation.split_selector.select(
        tuple(row.sample_id for row in validation_rows)
    )
    identities = selection.selected_ids[:sample_count]
    binding = qualitative.bind(identities, selector_digest=selection.membership_digest)

    def mapped_tokens(row) -> tuple[int, ...]:
        processed = evaluation.processor.process(row)
        mapped = evaluation.mapper.map(processed)
        return tuple(mapped.input_ids)

    heldout = {row.sample_id: mapped_tokens(row) for row in validation_rows[:sample_count]}
    validation_tokens = array("H")
    for row in validation_rows:
        validation_tokens.extend(mapped_tokens(row))
    required_validation = validation_windows * (sequence_length + 1)
    if len(validation_tokens) < required_validation:
        raise ValueError("validation split has too few tokens for declared evaluation")
    training_tokens = array("H")
    for row in training_rows:
        processed = training.processor.process(row)
        mapped = training.mapper.map(processed)
        training_tokens.extend(mapped.input_ids)
    if len(training_tokens) <= sequence_length + 1:
        raise ValueError("training split has too few tokens for declared context")
    try:
        with destination.open("xb") as output:
            validation_tokens[:required_validation].tofile(output)
            training_tokens.tofile(output)
    except OverflowError as error:
        raise ValueError("RWKV token IDs must fit the uint16 corpus format") from error
    return PreparedRWKVCorpus(
        path=destination,
        heldout_tokens=MappingProxyType(heldout),
        identities_digest=binding.identities_digest,
        selector_digest=binding.selector_digest,
    )


__all__ = [
    "PreparedRWKVCorpus",
    "RWKVScratchTrainConfig",
    "prepare_registered_corpus",
]
