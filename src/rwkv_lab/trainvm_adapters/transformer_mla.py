"""Closed TrainVM profiles for the canonical MLA-family trainer.

The legacy :mod:`rwkv_lab.train_mla` command exposes dozens of independently
composable research switches.  TrainVM deliberately does not forward that
surface.  Each topology below is a distinct adapter contract with an exact
freeze policy and optimizer-group meaning; a new combination requires a new
profile version instead of an unreviewed boolean cocktail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PROFILE_ADAPTERS = {
    "mla": "rwkv-lab.transformer-mla",
    "mtp": "rwkv-lab.transformer-mla-mtp",
    "mutor": "rwkv-lab.transformer-mla-mutor",
    "fsp": "rwkv-lab.transformer-mla-fsp",
    "parallel": "rwkv-lab.transformer-mla-parallel",
    "rwkv8": "rwkv-lab.transformer-mla-rwkv8",
    "engram": "rwkv-lab.transformer-mla-engram",
    "full_backbone": "rwkv-lab.transformer-mla-full-backbone",
}


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
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _path(value: object, label: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty path string")
    return value


@dataclass(frozen=True, slots=True)
class TransformerMLATrainConfig:
    profile: str
    model_dir: str
    patch_dir: str
    tokens_bin: str
    output_dir: str
    total_tokens_in_bin: int
    eval_tokens: int
    max_steps: int
    sequence_length: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-5
    warmup_steps: int = 200
    weight_decay: float = 0.0
    max_gradient_norm: float = 1.0
    log_every_steps: int = 10
    eval_every_steps: int = 500
    eval_batches: int = 32
    save_every_steps: int = 1000
    seed: int = 0
    mtp_loss_weight: float = 0.3
    mutor_weight: float = 0.3
    mutor_num_registers: int = 32
    mutor_min_horizon: int = 2
    mutor_max_horizon: int = 4
    fsp_weight: float = 0.1
    fsp_num_positions: int = 64
    fsp_horizon: int = 12
    fsp_idf_path: str = ""
    parallel_horizons: tuple[int, ...] = (2, 3, 4)
    parallel_weights: tuple[float, ...] = (0.3, 0.2, 0.1)
    parallel_head_expansion: int = 4
    rwkv8_layers: str = ""
    rwkv8_swap_mode: str = "timemix"
    engram_patch_dir: str = ""
    engram_learning_rate_multiplier: float = 5.0

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_ADAPTERS:
            raise ValueError("profile is not a registered Transformer MLA topology")
        for label in ("model_dir", "patch_dir", "tokens_bin", "output_dir"):
            _path(getattr(self, label), label)
        _integer(
            self.total_tokens_in_bin,
            "total_tokens_in_bin",
            2,
            (1 << 63) - 1,
        )
        _integer(self.eval_tokens, "eval_tokens", 1, self.total_tokens_in_bin - 1)
        _integer(self.max_steps, "max_steps", 1, 1_000_000_000)
        _integer(self.sequence_length, "sequence_length", 2, 1_048_576)
        _integer(self.micro_batch_size, "micro_batch_size", 1, 1_048_576)
        _integer(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
            1,
            65_536,
        )
        learning_rate = _finite(
            self.learning_rate, "learning_rate", 1.0e-12, 1.0
        )
        _finite(
            self.minimum_learning_rate,
            "minimum_learning_rate",
            0.0,
            learning_rate,
        )
        _integer(self.warmup_steps, "warmup_steps", 1, self.max_steps)
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e9)
        for label in ("log_every_steps", "eval_every_steps", "eval_batches"):
            _integer(getattr(self, label), label, 1, self.max_steps)
        _integer(self.save_every_steps, "save_every_steps", 0, self.max_steps)
        _integer(self.seed, "seed", 0, (1 << 63) - 1)

        if self.profile == "mtp":
            _finite(self.mtp_loss_weight, "mtp_loss_weight", 1.0e-12, 1000.0)
        elif self.mtp_loss_weight != 0.3:
            raise ValueError("mtp_loss_weight is only configurable for the mtp profile")
        if self.profile == "mutor":
            _finite(self.mutor_weight, "mutor_weight", 1.0e-12, 1000.0)
            _integer(self.mutor_num_registers, "mutor_num_registers", 1, 65_536)
            minimum = _integer(
                self.mutor_min_horizon, "mutor_min_horizon", 2, 1_048_576
            )
            maximum = _integer(
                self.mutor_max_horizon, "mutor_max_horizon", 2, 1_048_576
            )
            if maximum < minimum:
                raise ValueError("mutor_max_horizon cannot precede mutor_min_horizon")
        elif (
            self.mutor_weight != 0.3
            or self.mutor_num_registers != 32
            or self.mutor_min_horizon != 2
            or self.mutor_max_horizon != 4
        ):
            raise ValueError("MuToR fields are only configurable for the mutor profile")
        if self.profile == "fsp":
            _finite(self.fsp_weight, "fsp_weight", 1.0e-12, 1000.0)
            _integer(self.fsp_num_positions, "fsp_num_positions", 1, 65_536)
            _integer(self.fsp_horizon, "fsp_horizon", 1, 1_048_576)
            _path(self.fsp_idf_path, "fsp_idf_path", optional=True)
        elif (
            self.fsp_weight != 0.1
            or self.fsp_num_positions != 64
            or self.fsp_horizon != 12
            or self.fsp_idf_path
        ):
            raise ValueError("FSP fields are only configurable for the fsp profile")
        if self.profile == "parallel":
            if not isinstance(self.parallel_horizons, (list, tuple)):
                raise ValueError("parallel_horizons must be an integer sequence")
            if not isinstance(self.parallel_weights, (list, tuple)):
                raise ValueError("parallel_weights must be a numeric sequence")
            horizons = tuple(
                _integer(value, "parallel_horizons member", 2, 1_048_576)
                for value in self.parallel_horizons
            )
            weights = tuple(
                _finite(value, "parallel_weights member", 1.0e-12, 1000.0)
                for value in self.parallel_weights
            )
            if not horizons or len(horizons) != len(weights):
                raise ValueError(
                    "parallel_horizons and parallel_weights must be nonempty and aligned"
                )
            if horizons != tuple(sorted(set(horizons))):
                raise ValueError("parallel_horizons must be sorted and unique")
            _integer(
                self.parallel_head_expansion,
                "parallel_head_expansion",
                1,
                1024,
            )
            object.__setattr__(self, "parallel_horizons", horizons)
            object.__setattr__(self, "parallel_weights", weights)
        elif (
            self.parallel_horizons != (2, 3, 4)
            or self.parallel_weights != (0.3, 0.2, 0.1)
            or self.parallel_head_expansion != 4
        ):
            raise ValueError(
                "parallel-head fields are only configurable for the parallel profile"
            )
        if self.profile == "rwkv8":
            layers = self._rwkv8_layer_indices()
            if not layers:
                raise ValueError("rwkv8 profile requires at least one replacement layer")
            if self.rwkv8_swap_mode not in {"timemix", "channelmix"}:
                raise ValueError("rwkv8_swap_mode must be timemix or channelmix")
        elif self.rwkv8_layers or self.rwkv8_swap_mode != "timemix":
            raise ValueError("RWKV8 fields are only configurable for the rwkv8 profile")
        if self.profile == "engram":
            _path(self.engram_patch_dir, "engram_patch_dir")
            _finite(
                self.engram_learning_rate_multiplier,
                "engram_learning_rate_multiplier",
                1.0e-12,
                1000.0,
            )
        elif self.engram_patch_dir or self.engram_learning_rate_multiplier != 5.0:
            raise ValueError("Engram fields are only configurable for the engram profile")

    @property
    def adapter(self) -> str:
        return PROFILE_ADAPTERS[self.profile]

    def _rwkv8_layer_indices(self) -> tuple[int, ...]:
        if not self.rwkv8_layers:
            return ()
        values: list[int] = []
        for token in self.rwkv8_layers.split(","):
            if not token or not token.isascii() or not token.isdigit():
                raise ValueError("rwkv8_layers must be canonical comma-separated integers")
            value = int(token)
            if value < 0 or value > 4095:
                raise ValueError("rwkv8 layer index is outside [0, 4095]")
            values.append(value)
        if values != sorted(set(values)):
            raise ValueError("rwkv8_layers must be sorted and unique")
        return tuple(values)

    def trainer_configuration(self) -> Any:
        """Lower this exact profile into the canonical trainer's config type."""

        from rwkv_lab.train_mla import TrainConfig

        topology: dict[str, Any] = {
            "install_mtp": 0,
            "train_mtp_only": 0,
            "mutor_enabled": 0,
            "fsp_enabled": 0,
            "engram_enabled": 0,
            "parallel_enabled": 0,
            "train_aux_only": 0,
            "freeze_non_mla": 1,
            "rwkv8_deltanet_layers": "",
            "train_rwkv8_layers": "",
        }
        if self.profile == "mtp":
            topology.update(install_mtp=1, train_mtp_only=1)
        elif self.profile == "mutor":
            topology.update(mutor_enabled=1, train_aux_only=1)
        elif self.profile == "fsp":
            topology.update(fsp_enabled=1, train_aux_only=1)
        elif self.profile == "parallel":
            topology.update(parallel_enabled=1, train_aux_only=1)
        elif self.profile == "rwkv8":
            topology.update(
                rwkv8_deltanet_layers=self.rwkv8_layers,
                train_rwkv8_layers=self.rwkv8_layers,
            )
        elif self.profile == "engram":
            topology.update(engram_enabled=1)
        elif self.profile == "full_backbone":
            topology.update(freeze_non_mla=0)
        return TrainConfig(
            model_dir=self.model_dir,
            patch_dir=self.patch_dir,
            tokens_bin=self.tokens_bin,
            total_tokens_in_bin=self.total_tokens_in_bin,
            eval_tokens=self.eval_tokens,
            out_dir=self.output_dir,
            resume="",
            seq_len=self.sequence_length,
            micro_batch_size=self.micro_batch_size,
            grad_accum_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
            lr=self.learning_rate,
            min_lr=self.minimum_learning_rate,
            warmup_steps=self.warmup_steps,
            resume_warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            grad_clip=self.max_gradient_norm,
            log_every=self.log_every_steps,
            eval_every=self.eval_every_steps,
            eval_batches=self.eval_batches,
            save_every=self.save_every_steps,
            seed=self.seed,
            optimizer="trainvm",
            mtp_loss_weight=self.mtp_loss_weight,
            mutor_weight=self.mutor_weight,
            mutor_num_registers=self.mutor_num_registers,
            mutor_d_min=self.mutor_min_horizon,
            mutor_d_max=self.mutor_max_horizon,
            fsp_weight=self.fsp_weight,
            fsp_num_positions=self.fsp_num_positions,
            fsp_tau=self.fsp_horizon,
            fsp_idf_path=self.fsp_idf_path,
            parallel_horizons=",".join(str(value) for value in self.parallel_horizons),
            parallel_weights=",".join(str(value) for value in self.parallel_weights),
            parallel_head_expansion=self.parallel_head_expansion,
            rwkv8_swap_mode=self.rwkv8_swap_mode,
            engram_patch_dir=self.engram_patch_dir,
            engram_lr_mult=self.engram_learning_rate_multiplier,
            **topology,
        )


__all__ = ["PROFILE_ADAPTERS", "TransformerMLATrainConfig"]
