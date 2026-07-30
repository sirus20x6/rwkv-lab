from __future__ import annotations

import json

import pytest

from rwkv_lab.mage_flow_terminal_train import (
    TerminalExpertTrainConfig,
    _epoch_batches,
    prepare_run,
)


def _row(index: int, domain: str) -> dict:
    return {
        "image": f"/images/{index}.png",
        "image_id": f"id-{index}",
        "domain": domain,
        "source": "test",
        "caption": f"Caption {index}",
        "conditioning_text": f"Caption {index}",
        "conditioning_kind": "human",
        "is_captioned": True,
        "training_scope": "expert_and_selected_shared",
        "train_width": 512,
        "train_height": 512,
        "latent_tokens": 1024,
    }


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_terminal_config_selects_one_resident_domain(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [_row(0, "photo"), _row(1, "animation"), _row(2, "photo")],
    )
    expert = tmp_path / "photo.safetensors"
    expert.touch()
    config = TerminalExpertTrainConfig(
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
        max_steps=2,
    )
    config.validate()
    assert config.backbone_learning_rate_multiplier == pytest.approx(0.5)
    assert config.train_backbone_final_fraction == pytest.approx(1 / 3)

    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["domain"] == "photo"
    assert receipt["resident_experts"] == 1
    assert receipt["train_examples"] == 2
    assert "mage_flow_terminal_train" in (tmp_path / "plan/launch.sh").read_text()
    config.attention_backend = "flash4"
    prepare_run(config, tmp_path / "fa4-plan")
    assert ".venv-mage-flow-fa4" in (
        tmp_path / "fa4-plan/launch.sh"
    ).read_text()


def test_terminal_config_rejects_neutral_or_missing_domain(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(0, "animation")])
    expert = tmp_path / "expert.safetensors"
    expert.touch()
    general = TerminalExpertTrainConfig(
        domain="general",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="photo or animation"):
        general.validate()
    photo = TerminalExpertTrainConfig(
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="no photo"):
        photo.validate()


def test_domain_window_batches_are_deterministic_and_cover_once():
    first = _epoch_batches(11, batch_size=3, seed=42, epoch=0)
    second = _epoch_batches(11, batch_size=3, seed=42, epoch=0)
    later = _epoch_batches(11, batch_size=3, seed=42, epoch=1)
    assert first == second
    assert first != later
    assert sorted(index for batch in first for index in batch) == list(range(11))
    assert [len(batch) for batch in first] == [3, 3, 3, 2]
