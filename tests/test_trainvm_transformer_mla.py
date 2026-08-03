from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from rwkv_lab.trainvm_adapters.transformer_mla import (
    PROFILE_ADAPTERS,
    TransformerMLATrainConfig,
)


def _config(profile: str, **overrides) -> TransformerMLATrainConfig:
    values = {
        "profile": profile,
        "model_dir": "/sealed/model",
        "patch_dir": "/sealed/mla-patch",
        "tokens_bin": "/sealed/tokens.bin",
        "output_dir": "/run/output",
        "total_tokens_in_bin": 1_000_000,
        "eval_tokens": 10_000,
        "max_steps": 1_000,
        "warmup_steps": 10,
        "eval_every_steps": 20,
        "save_every_steps": 100,
    }
    values.update(overrides)
    return TransformerMLATrainConfig(**values)


def test_transformer_profiles_have_distinct_authority_identities() -> None:
    assert set(PROFILE_ADAPTERS) == {
        "mla",
        "mtp",
        "mutor",
        "fsp",
        "parallel",
        "rwkv8",
        "engram",
        "full_backbone",
    }
    assert len(set(PROFILE_ADAPTERS.values())) == len(PROFILE_ADAPTERS)
    assert all(value.startswith("rwkv-lab.transformer-mla") for value in PROFILE_ADAPTERS.values())


@pytest.mark.parametrize(
    ("profile", "extra", "expected"),
    [
        ("mla", {}, {}),
        ("mtp", {}, {"install_mtp": 1, "train_mtp_only": 1}),
        ("mutor", {}, {"mutor_enabled": 1, "train_aux_only": 1}),
        ("fsp", {}, {"fsp_enabled": 1, "train_aux_only": 1}),
        (
            "parallel",
            {"parallel_horizons": [2, 4], "parallel_weights": [0.75, 0.25]},
            {
                "parallel_enabled": 1,
                "parallel_horizons": "2,4",
                "parallel_weights": "0.75,0.25",
                "train_aux_only": 1,
            },
        ),
        (
            "rwkv8",
            {"rwkv8_layers": "4,8"},
            {"rwkv8_deltanet_layers": "4,8", "train_rwkv8_layers": "4,8"},
        ),
        (
            "engram",
            {"engram_patch_dir": "/sealed/engram"},
            {"engram_enabled": 1, "engram_patch_dir": "/sealed/engram"},
        ),
        ("full_backbone", {}, {"freeze_non_mla": 0}),
    ],
)
def test_transformer_profile_lowers_to_one_exact_legacy_topology(
    profile: str, extra: dict[str, object], expected: dict[str, object]
) -> None:
    config = _config(profile, **extra)
    lowered = config.trainer_configuration()
    assert config.adapter == PROFILE_ADAPTERS[profile]
    assert lowered.optimizer == "trainvm"
    assert lowered.out_dir == "/run/output"
    for name, value in expected.items():
        assert getattr(lowered, name) == value
    enabled = {
        "mtp": bool(lowered.install_mtp or lowered.train_mtp_only),
        "mutor": bool(lowered.mutor_enabled),
        "fsp": bool(lowered.fsp_enabled),
        "parallel": bool(lowered.parallel_enabled),
        "rwkv8": bool(lowered.rwkv8_deltanet_layers),
        "engram": bool(lowered.engram_enabled),
        "full_backbone": not bool(lowered.freeze_non_mla),
    }
    assert {name for name, value in enabled.items() if value} == (
        set() if profile == "mla" else {profile}
    )


def test_transformer_profiles_reject_cross_profile_and_ambiguous_topology_fields() -> None:
    with pytest.raises(TypeError, match="resume"):
        _config("mla", resume="/untyped/checkpoint.pt")
    with pytest.raises(ValueError, match="MuToR fields"):
        _config("mla", mutor_weight=0.4)
    with pytest.raises(ValueError, match="FSP fields"):
        _config("mutor", fsp_idf_path="/sealed/idf.pt")
    with pytest.raises(ValueError, match="Engram fields"):
        _config("fsp", engram_patch_dir="/sealed/engram")
    with pytest.raises(ValueError, match="parallel-head fields"):
        _config("fsp", parallel_horizons=(2, 4))
    with pytest.raises(ValueError, match="sorted and unique"):
        _config(
            "parallel",
            parallel_horizons=[4, 2],
            parallel_weights=[0.5, 0.5],
        )
    with pytest.raises(ValueError, match="nonempty and aligned"):
        _config("parallel", parallel_horizons=[2, 4], parallel_weights=[0.5])
    with pytest.raises(ValueError, match="sorted and unique"):
        _config("rwkv8", rwkv8_layers="8,4")
    with pytest.raises(ValueError, match="canonical comma-separated"):
        _config("rwkv8", rwkv8_layers="4, 8")
    with pytest.raises(ValueError, match="requires at least one"):
        _config("rwkv8")
    with pytest.raises(ValueError, match="engram_patch_dir"):
        _config("engram")


def test_transformer_profile_is_immutable_and_has_no_machine_specific_defaults() -> None:
    config = _config("mla")
    with pytest.raises(FrozenInstanceError):
        config.profile = "mtp"  # type: ignore[misc]
    changed = replace(config, profile="full_backbone")
    assert changed.profile == "full_backbone"
    assert all(
        not value.startswith("/thearray/")
        for value in (
            config.model_dir,
            config.patch_dir,
            config.tokens_bin,
            config.output_dir,
        )
    )


def test_engram_runtime_never_injects_the_maintainer_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/rwkv_lab/train_mla.py",
        "src/rwkv_lab/engram_integration.py",
        "src/rwkv_lab/load_mla_engram.py",
    ):
        assert "/thearray/git/engram/python" not in (root / relative).read_text(
            encoding="utf-8"
        )


def test_canonical_checkpoint_persists_trainvm_composition_and_control_state(
    tmp_path,
) -> None:
    from rwkv_lab.train_mla import TrainConfig, save_checkpoint

    module = torch.nn.Linear(2, 2)
    module._save_key = "layer_0"  # type: ignore[attr-defined]
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-4)
    config = TrainConfig(out_dir=str(tmp_path))
    evidence = {
        "optimizer": {
            "category": "optimizer",
            "implementation": "rwkv_lab.optimizer.torch_adamw.v1",
            "descriptor_digest": "sha256:" + "a" * 64,
        }
    }
    controls = {
        "effective_control_revision": 3,
        "effective_controls": {},
    }

    checkpoint = save_checkpoint(
        7,
        [module],  # type: ignore[list-item]
        optimizer,
        config,
        plateau_state={"mult": 1.0},
        component_evidence=evidence,
        component_composition_digest="sha256:" + "b" * 64,
        worker_control_state=controls,
    )
    payload = torch.load(checkpoint / "ckpt.pt", weights_only=False)

    assert checkpoint == tmp_path / "step_000007"
    assert payload["trainvm_component_evidence"] == evidence
    assert payload["component_composition_digest"] == "sha256:" + "b" * 64
    assert payload["trainvm_worker_controls"] == controls
    assert payload["plateau_state"] == {"mult": 1.0}

    # A same-step rewrite atomically replaces the completed directory rather
    # than deleting it before promotion.
    second = save_checkpoint(
        7,
        [module],  # type: ignore[list-item]
        optimizer,
        config,
        plateau_state={"mult": 0.5},
        component_evidence=evidence,
        component_composition_digest="sha256:" + "b" * 64,
        worker_control_state=controls,
    )
    assert second == checkpoint
    assert not (tmp_path / ".step_000007.tmp").exists()
    replaced = torch.load(second / "ckpt.pt", weights_only=False)
    assert replaced["plateau_state"] == {"mult": 0.5}
