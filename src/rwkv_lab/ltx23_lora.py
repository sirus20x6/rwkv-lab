"""LTX-2.3 video LoRA training orchestration.

This module deliberately delegates model loading, latent preprocessing, flow
matching, and PEFT checkpoint serialization to Lightricks' official LTX-2
trainer.  It owns the reproducible boundary around that trainer: validating an
LTX-2.3 run, generating its structured YAML, constructing commands, and
recording the exact external revision and arguments used.

Typical usage::

    python -m rwkv_lab.ltx23_lora run \
        --ltx-root /opt/LTX-2 \
        --model-path models/ltx-2.3-22b-dev.safetensors \
        --text-encoder-path models/gemma-3-12b \
        --dataset data/videos.jsonl \
        --work-dir runs/my_ltx23_lora
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence

import yaml


RUN_SCHEMA = "rwkv-lab.ltx23-lora.v1"
OFFICIAL_REPOSITORY = "https://github.com/Lightricks/LTX-2"
MINIMUM_TRAINER_VERSION = (1, 1, 7)
DEFAULT_TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
FFN_TARGETS = (
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
)


def parse_resolution_bucket(value: str) -> tuple[int, int, int]:
    """Parse and validate LTX-2's ``width x height x frames`` geometry."""
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", value.strip().lower())
    if match is None:
        raise ValueError(
            f"invalid resolution bucket {value!r}; expected WIDTHxHEIGHTxFRAMES"
        )
    width, height, frames = (int(item) for item in match.groups())
    if width % 32 or height % 32:
        raise ValueError(
            f"LTX-2.3 width and height must be divisible by 32, got {width}x{height}"
        )
    if frames % 8 != 1:
        raise ValueError(
            f"LTX-2.3 frame count must satisfy frames % 8 == 1, got {frames}"
        )
    return width, height, frames


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def trainer_root(ltx_root: Path) -> Path:
    return ltx_root / "packages" / "ltx-trainer"


def official_trainer_version(ltx_root: Path) -> str:
    project_file = trainer_root(ltx_root) / "pyproject.toml"
    if not project_file.is_file():
        raise ValueError(
            f"{ltx_root} is not the official LTX-2 monorepo "
            f"(missing {project_file.relative_to(ltx_root)})"
        )
    text = project_file.read_text()
    project_section = text.split("[project]", 1)[-1].split("\n[", 1)[0]
    match = re.search(
        r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']\s*$", project_section
    )
    if match is None:
        raise ValueError(f"could not read ltx-trainer version from {project_file}")
    return match.group(1)


def official_revision(ltx_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ltx_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _checkpoint_declares_ltx23(path: Path) -> bool:
    if "ltx-2.3" in path.name.lower():
        return True
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return "2.3" in json.dumps(metadata, sort_keys=True).lower()
    except Exception:
        return False


@dataclass(frozen=True)
class LTX23LoraSpec:
    ltx_root: Path
    model_path: Path
    text_encoder_path: Path
    preprocessed_data_root: Path
    output_dir: Path
    resolution_buckets: tuple[str, ...] = ("960x544x49",)
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGETS
    with_audio: bool = True
    learning_rate: float = 1e-4
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    quantization: str | None = None
    load_text_encoder_in_8bit: bool = False
    offload_optimizer_during_validation: bool = False
    num_dataloader_workers: int = 2
    validation_prompts: tuple[str, ...] = ()
    validation_interval: int | None = None
    validation_inference_steps: int = 30
    checkpoint_interval: int | None = 250
    keep_last_n: int = 3
    save_training_state: str = "minimal"
    no_resume: bool = False
    load_checkpoint: Path | None = None
    seed: int = 42
    wandb_project: str | None = None
    hub_model_id: str | None = None
    allow_unverified_checkpoint: bool = False

    def validate(self, *, require_preprocessed: bool = False) -> None:
        root = self.ltx_root.expanduser().resolve()
        expected_scripts = (
            trainer_root(root) / "scripts" / "process_dataset.py",
            trainer_root(root) / "scripts" / "train.py",
        )
        missing = [str(path) for path in expected_scripts if not path.is_file()]
        if missing:
            raise ValueError(
                f"{root} is missing official trainer scripts: {', '.join(missing)}"
            )
        version = official_trainer_version(root)
        if _version_tuple(version) < MINIMUM_TRAINER_VERSION:
            minimum = ".".join(str(part) for part in MINIMUM_TRAINER_VERSION)
            raise ValueError(
                f"ltx-trainer {version} is too old for this integration; need >= {minimum}"
            )
        if not self.model_path.is_file() or self.model_path.suffix != ".safetensors":
            raise ValueError("model_path must be a local .safetensors checkpoint")
        if (
            not self.allow_unverified_checkpoint
            and not _checkpoint_declares_ltx23(self.model_path)
        ):
            raise ValueError(
                "checkpoint does not identify itself as LTX-2.3; use the canonical "
                "ltx-2.3-22b-dev.safetensors or pass --allow-unverified-checkpoint"
            )
        if not self.text_encoder_path.is_dir():
            raise ValueError("text_encoder_path must be a local Gemma model directory")
        if self.rank < 2 or self.alpha < 1:
            raise ValueError("LoRA rank must be >= 2 and alpha must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("LoRA dropout must be in [0, 1]")
        if not self.target_modules:
            raise ValueError("at least one LoRA target module is required")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.validation_interval is not None and self.validation_interval < 1:
            raise ValueError("validation_interval must be positive or disabled")
        if self.validation_interval is not None and not self.validation_prompts:
            raise ValueError(
                "validation_interval requires at least one validation prompt"
            )
        if self.validation_inference_steps < 1:
            raise ValueError("validation_inference_steps must be positive")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive or disabled")
        if self.keep_last_n < -1:
            raise ValueError("keep_last_n must be -1 or non-negative")
        if self.num_dataloader_workers < 0:
            raise ValueError("num_dataloader_workers must be non-negative")
        for bucket in self.resolution_buckets:
            parse_resolution_bucket(bucket)
        if len(self.resolution_buckets) > 1 and self.batch_size != 1:
            raise ValueError("multiple resolution buckets require batch_size=1")
        if self.load_checkpoint is not None and not self.load_checkpoint.exists():
            raise ValueError(f"resume checkpoint does not exist: {self.load_checkpoint}")
        if require_preprocessed:
            needed = ["conditions", "latents"]
            if self.with_audio:
                needed.append("audio_latents")
            absent = [
                name
                for name in needed
                if not (self.preprocessed_data_root / name).is_dir()
            ]
            if absent:
                raise ValueError(
                    f"preprocessed dataset is missing {absent} under "
                    f"{self.preprocessed_data_root}; run the prepare action first"
                )

    def official_config(self) -> dict[str, Any]:
        """Return a minimal config accepted by the official structured trainer."""
        width, height, frames = parse_resolution_bucket(self.resolution_buckets[0])
        strategy: dict[str, Any] = {
            "name": "flexible",
            "video": {"is_generated": True, "latents_dir": "latents"},
        }
        if self.with_audio:
            strategy["audio"] = {
                "is_generated": True,
                "latents_dir": "audio_latents",
            }
        samples = [{"prompt": prompt} for prompt in self.validation_prompts]
        return {
            "model": {
                "model_path": str(self.model_path.expanduser().resolve()),
                "text_encoder_path": str(
                    self.text_encoder_path.expanduser().resolve()
                ),
                "training_mode": "lora",
                "load_checkpoint": (
                    str(self.load_checkpoint.expanduser().resolve())
                    if self.load_checkpoint is not None
                    else None
                ),
            },
            "lora": {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_modules": list(self.target_modules),
            },
            "training_strategy": strategy,
            "optimization": {
                "learning_rate": self.learning_rate,
                "steps": self.steps,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "optimizer_type": self.optimizer_type,
                "scheduler_type": self.scheduler_type,
                "scheduler_params": {},
                "enable_gradient_checkpointing": self.gradient_checkpointing,
            },
            "acceleration": {
                "mixed_precision_mode": self.mixed_precision,
                "quantization": self.quantization,
                "load_text_encoder_in_8bit": self.load_text_encoder_in_8bit,
                "offload_optimizer_during_validation": (
                    self.offload_optimizer_during_validation
                ),
            },
            "data": {
                "preprocessed_data_root": str(
                    self.preprocessed_data_root.expanduser().resolve()
                ),
                "num_dataloader_workers": self.num_dataloader_workers,
            },
            "validation": {
                "samples": samples,
                "video_dims": [width, height, frames],
                "frame_rate": 25.0,
                "seed": self.seed,
                "inference_steps": self.validation_inference_steps,
                "interval": self.validation_interval,
                "guidance_scale": 4.0,
                "stg_scale": 1.0,
                "stg_blocks": [29],
                "stg_mode": "stg_av" if self.with_audio else "stg_v",
                "generate_audio": self.with_audio,
                "generate_video": True,
                "skip_initial_validation": True,
            },
            "checkpoints": {
                "interval": self.checkpoint_interval,
                "keep_last_n": self.keep_last_n,
                "precision": "bfloat16",
                "no_resume": self.no_resume,
                "save_training_state": self.save_training_state,
            },
            "flow_matching": {
                "timestep_sampling_mode": "shifted_logit_normal",
                "timestep_sampling_params": {},
            },
            "hub": {
                "push_to_hub": self.hub_model_id is not None,
                "hub_model_id": self.hub_model_id,
            },
            "wandb": {
                "enabled": self.wandb_project is not None,
                "project": self.wandb_project or "ltx-2.3-lora",
                "entity": None,
                "tags": ["ltx-2.3", "lora", "video"],
                "log_validation_videos": True,
            },
            "seed": self.seed,
            "output_dir": str(self.output_dir.expanduser().resolve()),
        }


def _runner_prefix(runner: str, python: str) -> list[str]:
    if runner == "uv":
        executable = shutil.which("uv")
        if executable is None:
            raise ValueError("runner 'uv' requested but uv is not installed")
        return [executable, "run"]
    return [python]


def preprocess_command(
    spec: LTX23LoraSpec,
    dataset: Path,
    *,
    runner: str = "uv",
    python: str = sys.executable,
    preprocess_batch_size: int = 1,
    lora_trigger: str | None = None,
    vae_tiling: bool = False,
    remove_llm_prefixes: bool = False,
    overwrite: bool = False,
) -> list[str]:
    command = [
        *_runner_prefix(runner, python),
        "python" if runner == "uv" else str(
            trainer_root(spec.ltx_root) / "scripts" / "process_dataset.py"
        ),
    ]
    if runner == "uv":
        command.append("scripts/process_dataset.py")
    command.extend(
        [
            str(dataset.expanduser().resolve()),
            "--resolution-buckets",
            ";".join(spec.resolution_buckets),
            "--model-path",
            str(spec.model_path.expanduser().resolve()),
            "--text-encoder-path",
            str(spec.text_encoder_path.expanduser().resolve()),
            "--output-dir",
            str(spec.preprocessed_data_root.expanduser().resolve()),
            "--batch-size",
            str(preprocess_batch_size),
        ]
    )
    if lora_trigger:
        command.extend(["--lora-trigger", lora_trigger])
    if vae_tiling:
        command.append("--vae-tiling")
    if remove_llm_prefixes:
        command.append("--remove-llm-prefixes")
    if not spec.with_audio:
        command.append("--skip-audio")
    if spec.load_text_encoder_in_8bit:
        command.append("--load-text-encoder-in-8bit")
    if overwrite:
        command.append("--overwrite")
    return command


def train_command(
    spec: LTX23LoraSpec,
    config_path: Path,
    *,
    runner: str = "uv",
    python: str = sys.executable,
    processes: int = 1,
    disable_progress_bars: bool = False,
) -> list[str]:
    prefix = _runner_prefix(runner, python)
    if processes > 1:
        if runner == "uv":
            command = [*prefix, "accelerate", "launch"]
        else:
            command = [*prefix, "-m", "accelerate.commands.launch"]
        command.extend(["--num_processes", str(processes), "scripts/train.py"])
    elif runner == "uv":
        command = [*prefix, "python", "scripts/train.py"]
    else:
        command = [
            *prefix,
            str(trainer_root(spec.ltx_root) / "scripts" / "train.py"),
        ]
    command.append(str(config_path.expanduser().resolve()))
    if disable_progress_bars:
        command.append("--disable-progress-bars")
    return command


def write_run_files(
    spec: LTX23LoraSpec,
    config_path: Path,
    *,
    prepare: Sequence[str] | None = None,
    train: Sequence[str] | None = None,
) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(spec.official_config(), sort_keys=False)
    config_path.write_text(rendered)
    receipt = {
        "schema": RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": official_revision(spec.ltx_root),
        "official_trainer_version": official_trainer_version(spec.ltx_root),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "model": {
            "path": str(spec.model_path.resolve()),
            "size_bytes": spec.model_path.stat().st_size,
        },
        "spec": {
            key: (
                str(value)
                if isinstance(value, Path)
                else [str(item) for item in value]
                if isinstance(value, tuple)
                else value
            )
            for key, value in asdict(spec).items()
        },
        "commands": {
            "prepare": list(prepare) if prepare is not None else None,
            "train": list(train) if train is not None else None,
        },
    }
    receipt_path = config_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path


def _run(command: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(list(command), cwd=cwd, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, preprocess, and train an official LTX-2.3 video LoRA"
    )
    parser.add_argument("action", choices=("plan", "prepare", "train", "run"))
    parser.add_argument(
        "--ltx-root",
        default=os.environ.get("LTX2_ROOT"),
        required=os.environ.get("LTX2_ROOT") is None,
        help="clone of https://github.com/Lightricks/LTX-2",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text-encoder-path", required=True)
    parser.add_argument("--dataset", help="JSON/JSONL/CSV with video and caption columns")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--preprocessed-data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--config-out")
    parser.add_argument("--resolution-bucket", action="append", dest="buckets")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-module", action="append", dest="targets")
    parser.add_argument("--include-ffn", action="store_true")
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="jointly train LTX-2.3's synchronized audio branch",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default=None)
    parser.add_argument(
        "--scheduler",
        choices=(
            "constant",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "step",
        ),
        default="linear",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=(
            "none",
            "int8-quanto",
            "int4-quanto",
            "int2-quanto",
            "fp8-quanto",
            "fp8uz-quanto",
        ),
    )
    parser.add_argument(
        "--load-text-encoder-in-8bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--num-dataloader-workers", type=int, default=2)
    parser.add_argument(
        "--offload-optimizer-during-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--validation-prompt", action="append", default=[])
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--validation-inference-steps", type=int, default=30)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument(
        "--save-training-state",
        choices=("full", "minimal", "off"),
        default="minimal",
    )
    parser.add_argument("--resume")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project")
    parser.add_argument("--hub-model-id")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--preprocess-batch-size", type=int, default=1)
    parser.add_argument("--lora-trigger")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--remove-llm-prefixes", action="store_true")
    parser.add_argument("--overwrite-preprocessed", action="store_true")
    parser.add_argument("--runner", choices=("uv", "python"), default="uv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--disable-progress-bars", action="store_true")
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _spec_from_args(args: argparse.Namespace) -> tuple[LTX23LoraSpec, Path, Path]:
    work = Path(args.work_dir).expanduser().resolve()
    preprocessed = Path(
        args.preprocessed_data_root or work / "preprocessed"
    ).expanduser().resolve()
    output = Path(args.output_dir or work / "output").expanduser().resolve()
    config = Path(args.config_out or work / "ltx23_lora.yaml").expanduser().resolve()
    targets = list(args.targets or DEFAULT_TARGETS)
    if args.include_ffn:
        targets.extend(FFN_TARGETS)
    targets = list(dict.fromkeys(targets))
    rank = args.rank if args.rank is not None else (16 if args.low_vram else 32)
    optimizer = args.optimizer or ("adamw8bit" if args.low_vram else "adamw")
    quantization = args.quantization or ("int8-quanto" if args.low_vram else None)
    if quantization == "none":
        quantization = None
    text_8bit = (
        args.low_vram
        if args.load_text_encoder_in_8bit is None
        else args.load_text_encoder_in_8bit
    )
    spec = LTX23LoraSpec(
        ltx_root=Path(args.ltx_root).expanduser().resolve(),
        model_path=Path(args.model_path).expanduser().resolve(),
        text_encoder_path=Path(args.text_encoder_path).expanduser().resolve(),
        preprocessed_data_root=preprocessed,
        output_dir=output,
        resolution_buckets=tuple(args.buckets or ("960x544x49",)),
        rank=rank,
        alpha=args.alpha if args.alpha is not None else rank,
        dropout=args.dropout,
        target_modules=tuple(targets),
        with_audio=args.audio,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        optimizer_type=optimizer,
        scheduler_type=args.scheduler,
        gradient_checkpointing=args.gradient_checkpointing,
        quantization=quantization,
        load_text_encoder_in_8bit=text_8bit,
        offload_optimizer_during_validation=(
            args.low_vram
            if args.offload_optimizer_during_validation is None
            else args.offload_optimizer_during_validation
        ),
        num_dataloader_workers=args.num_dataloader_workers,
        validation_prompts=tuple(args.validation_prompt),
        validation_interval=args.validation_interval or None,
        validation_inference_steps=args.validation_inference_steps,
        checkpoint_interval=args.checkpoint_interval or None,
        keep_last_n=args.keep_last_n,
        save_training_state=args.save_training_state,
        no_resume=args.no_resume,
        load_checkpoint=Path(args.resume).expanduser().resolve() if args.resume else None,
        seed=args.seed,
        wandb_project=args.wandb_project,
        hub_model_id=args.hub_model_id,
        allow_unverified_checkpoint=args.allow_unverified_checkpoint,
    )
    return spec, work, config


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec, work, config_path = _spec_from_args(args)
    dataset = (
        Path(args.dataset).expanduser().resolve() if args.dataset is not None else None
    )
    if args.action in {"prepare", "run"}:
        if dataset is None:
            raise SystemExit("--dataset is required for prepare and run")
    if dataset is not None and not dataset.is_file():
        raise SystemExit(f"dataset does not exist: {dataset}")
    if args.processes < 1 or args.preprocess_batch_size < 1:
        raise SystemExit("process counts and preprocessing batch size must be positive")

    spec.validate(require_preprocessed=args.action == "train")
    prepare = (
        preprocess_command(
            spec,
            dataset,
            runner=args.runner,
            python=args.python,
            preprocess_batch_size=args.preprocess_batch_size,
            lora_trigger=args.lora_trigger,
            vae_tiling=args.vae_tiling,
            remove_llm_prefixes=args.remove_llm_prefixes,
            overwrite=args.overwrite_preprocessed,
        )
        if dataset is not None
        else None
    )
    train = train_command(
        spec,
        config_path,
        runner=args.runner,
        python=args.python,
        processes=args.processes,
        disable_progress_bars=args.disable_progress_bars,
    )
    receipt = write_run_files(spec, config_path, prepare=prepare, train=train)
    print(f"config: {config_path}", flush=True)
    print(f"receipt: {receipt}", flush=True)
    cwd = trainer_root(spec.ltx_root)

    if args.action in {"prepare", "run"}:
        assert prepare is not None
        _run(prepare, cwd=cwd, dry_run=args.dry_run)
    if args.action in {"train", "run"}:
        if args.action == "run" and not args.dry_run:
            spec.validate(require_preprocessed=True)
        _run(train, cwd=cwd, dry_run=args.dry_run)
    elif args.action == "plan":
        if prepare is not None:
            print("prepare: " + shlex.join(prepare), flush=True)
        print("train: " + shlex.join(train), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
