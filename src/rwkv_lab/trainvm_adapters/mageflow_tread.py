"""Typed MageFlow terminal-checkpoint to TREAD-controller conversion."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_MAGE_FLOW_BASE_ID = "microsoft/Mage-Flow-Base"
_MAGE_FLOW_BASE_REVISION = "59a9cfd58cf6ecef28245852c6bdace3f12428a2"


class MageFlowTreadConversionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MageFlowTreadConversionConfig:
    domain: str
    model_path: str
    tread_config: str
    model_id: str
    model_revision: str
    attention_backend: str = "flash2"

    def validate(self) -> None:
        fields = {
            "domain": self.domain,
            "model_path": self.model_path,
            "tread_config": self.tread_config,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "attention_backend": self.attention_backend,
        }
        if any(
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > 4096
            or any(character in value for character in ("\x00", "\n", "\r"))
            for value in fields.values()
        ):
            raise MageFlowTreadConversionError(
                "TREAD conversion string fields are invalid"
            )
        if self.domain not in {"photo", "animation"}:
            raise MageFlowTreadConversionError(
                "TREAD conversion domain must be photo or animation"
            )
        if not Path(self.model_path).is_absolute():
            raise MageFlowTreadConversionError("model_path must be absolute")
        if not Path(self.tread_config).is_absolute():
            raise MageFlowTreadConversionError("tread_config must be absolute")
        if (
            self.model_id != _MAGE_FLOW_BASE_ID
            or self.model_revision != _MAGE_FLOW_BASE_REVISION
        ):
            raise MageFlowTreadConversionError(
                "TREAD conversion requires the qualified Mage-Flow-Base revision"
            )
        if self.attention_backend not in {"flash2", "flash4"}:
            raise MageFlowTreadConversionError(
                "TREAD conversion attention backend is unsupported"
            )

    def canonical_digest(self) -> str:
        encoded = json.dumps(
            {
                "attention_backend": self.attention_backend,
                "domain": self.domain,
                "model_id": self.model_id,
                "model_path": self.model_path,
                "model_revision": self.model_revision,
                "tread_config": self.tread_config,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def checkpoint_terminal_paths(
    checkpoint_directory: Path,
    domain: str,
) -> tuple[Path, Path | None]:
    """Resolve the requested terminal expert and optional shared backbone."""

    try:
        contract = json.loads(
            (checkpoint_directory / "checkpoint.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MageFlowTreadConversionError(
            "terminal checkpoint metadata is unavailable"
        ) from error
    if not isinstance(contract, Mapping):
        raise MageFlowTreadConversionError(
            "terminal checkpoint metadata is not an object"
        )
    experts = contract.get("experts")
    if isinstance(experts, Mapping):
        expert_name = experts.get(domain)
    elif contract.get("domain") == domain:
        expert_name = contract.get("expert")
    else:
        expert_name = None
    expert = _checkpoint_file(
        checkpoint_directory,
        expert_name,
        label=f"{domain} terminal expert",
        required=True,
    )
    shared = _checkpoint_file(
        checkpoint_directory,
        contract.get("shared_backbone"),
        label="shared terminal backbone",
        required=False,
    )
    assert expert is not None
    return expert, shared


def _checkpoint_file(
    root: Path,
    value: object,
    *,
    label: str,
    required: bool,
) -> Path | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or Path(value).name != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise MageFlowTreadConversionError(
            f"terminal checkpoint {label} path is invalid"
        )
    candidate = root / value
    try:
        status = candidate.lstat()
    except OSError as error:
        raise MageFlowTreadConversionError(
            f"terminal checkpoint {label} is unavailable"
        ) from error
    if not candidate.is_file() or candidate.is_symlink() or status.st_size <= 0:
        raise MageFlowTreadConversionError(
            f"terminal checkpoint {label} is not a regular file"
        )
    return candidate


def build_tread_controller(
    config: MageFlowTreadConversionConfig,
    checkpoint_directory: Path,
    output_directory: Path,
) -> Mapping[str, object]:
    """Build the zero-gated controller on the sealed BF16 CUDA path."""

    config.validate()
    expert_path, shared_path = checkpoint_terminal_paths(
        checkpoint_directory, config.domain
    )

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise MageFlowTreadConversionError(
            "TREAD conversion requires a BF16 CUDA device"
        )
    from mage_flow import MageFlowPipeline

    from rwkv_lab.mage_flow_terminal_experts import (
        install_terminal_expert,
        load_terminal_expert,
        load_terminal_shared_backbone,
    )
    from rwkv_lab.mage_flow_tread_looping import (
        TreadLoopConfig,
        install_tread_factored_looping,
        save_tread_loop_controller,
    )

    try:
        tread_document = json.loads(
            Path(config.tread_config).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MageFlowTreadConversionError(
            "TREAD configuration is unavailable"
        ) from error
    if not isinstance(tread_document, dict):
        raise MageFlowTreadConversionError(
            "TREAD configuration must be a JSON object"
        )
    try:
        tread = TreadLoopConfig.from_dict(tread_document)
    except (TypeError, ValueError) as error:
        raise MageFlowTreadConversionError(
            "TREAD configuration is invalid"
        ) from error
    device = torch.device("cuda")
    pipeline = MageFlowPipeline.from_pretrained(
        config.model_path,
        device=str(device),
        attn_type=config.attention_backend,
    )
    transformer = pipeline.model.transformer
    reference = next(transformer.parameters())
    if reference.device.type != "cuda" or reference.dtype != torch.bfloat16:
        raise MageFlowTreadConversionError(
            "TREAD conversion requires the local MageFlow model in CUDA BF16"
        )
    terminal = install_terminal_expert(
        transformer,
        config.domain,
        device=device,
        dtype=reference.dtype,
    )
    controller = install_tread_factored_looping(transformer, tread)
    load_terminal_expert(terminal, config.domain, expert_path)
    if shared_path is not None:
        load_terminal_shared_backbone(transformer, shared_path)
    controller_path = output_directory / "controller.safetensors"
    report = dict(save_tread_loop_controller(controller, controller_path))
    report["path"] = controller_path.name
    report.update(
        {
            "schema": "rwkv-lab.mageflow-tread-conversion-report.v1",
            "domain": config.domain,
            "base_model": config.model_id,
            "base_revision": config.model_revision,
            "attention_backend": config.attention_backend,
            "config_digest": config.canonical_digest(),
            "shared_backbone_loaded": shared_path is not None,
        }
    )
    _write_canonical_json(output_directory / "report.json", report)
    return report


def _write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MageFlowTreadConversionError(
            "TREAD conversion report is not finite JSON"
        ) from error
    if not encoded or len(encoded) > 128 * 1024:
        raise MageFlowTreadConversionError(
            "TREAD conversion report exceeds its byte bound"
        )
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


__all__ = [
    "MageFlowTreadConversionConfig",
    "MageFlowTreadConversionError",
    "build_tread_controller",
    "checkpoint_terminal_paths",
]
