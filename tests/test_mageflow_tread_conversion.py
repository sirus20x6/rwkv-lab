from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rwkv_lab.trainvm_adapters.mageflow_tread import (
    MageFlowTreadConversionConfig,
    MageFlowTreadConversionError,
    checkpoint_terminal_paths,
)


def test_tread_conversion_config_is_pinned_and_canonical(tmp_path) -> None:
    config = MageFlowTreadConversionConfig(
        domain="animation",
        model_path=str(tmp_path / "model"),
        tread_config=str(tmp_path / "tread.json"),
        model_id="microsoft/Mage-Flow-Base",
        model_revision="59a9cfd58cf6ecef28245852c6bdace3f12428a2",
    )
    config.validate()
    assert config.canonical_digest().startswith("sha256:")

    wrong_revision = replace(config, model_revision="main")
    with pytest.raises(MageFlowTreadConversionError, match="qualified"):
        wrong_revision.validate()


def test_checkpoint_terminal_paths_select_domain_without_path_escape(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    animation = checkpoint / "animation.safetensors"
    photo = checkpoint / "photo.safetensors"
    shared = checkpoint / "shared.safetensors"
    for path in (animation, photo, shared):
        path.write_bytes(path.name.encode())
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "experts": {
                    "animation": animation.name,
                    "photo": photo.name,
                },
                "shared_backbone": shared.name,
            }
        ),
        encoding="utf-8",
    )

    assert checkpoint_terminal_paths(checkpoint, "animation") == (
        animation,
        shared,
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "experts": {"animation": "../outside.safetensors"},
                "shared_backbone": shared.name,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MageFlowTreadConversionError, match="path is invalid"):
        checkpoint_terminal_paths(checkpoint, "animation")


def test_checkpoint_terminal_paths_reject_symlinked_weights(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (checkpoint / "animation.safetensors").symlink_to(outside)
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "domain": "animation",
                "expert": "animation.safetensors",
                "shared_backbone": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MageFlowTreadConversionError, match="not a regular file"):
        checkpoint_terminal_paths(checkpoint, "animation")
