import json
from pathlib import Path
import sys

import pytest
import yaml

from rwkv_lab.ltx23_lora import (
    DEFAULT_TARGETS,
    LTX23LoraSpec,
    main,
    parse_resolution_bucket,
    preprocess_command,
    train_command,
    write_run_files,
)


def _fake_install(tmp_path: Path, *, preprocessed: bool = False):
    root = tmp_path / "LTX-2"
    trainer = root / "packages" / "ltx-trainer"
    scripts = trainer / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "process_dataset.py").touch()
    (scripts / "train.py").touch()
    (trainer / "pyproject.toml").write_text(
        '[project]\nname = "ltx-trainer"\nversion = "1.1.7"\n'
    )
    model = tmp_path / "ltx-2.3-22b-dev.safetensors"
    model.write_bytes(b"fixture")
    encoder = tmp_path / "gemma"
    encoder.mkdir()
    data = tmp_path / "preprocessed"
    if preprocessed:
        for name in ("latents", "conditions", "audio_latents"):
            (data / name).mkdir(parents=True)
    spec = LTX23LoraSpec(
        ltx_root=root,
        model_path=model,
        text_encoder_path=encoder,
        preprocessed_data_root=data,
        output_dir=tmp_path / "output",
    )
    return spec, trainer


def test_ltx23_geometry_contract():
    assert parse_resolution_bucket("960x544x49") == (960, 544, 49)
    with pytest.raises(ValueError, match="divisible by 32"):
        parse_resolution_bucket("950x544x49")
    with pytest.raises(ValueError, match=r"frames % 8"):
        parse_resolution_bucket("960x544x48")


def test_ltx23_official_config_is_lora_and_joint_av(tmp_path):
    spec, _ = _fake_install(tmp_path)
    spec.validate()
    config = spec.official_config()
    assert config["model"]["training_mode"] == "lora"
    assert config["lora"]["target_modules"] == list(DEFAULT_TARGETS)
    assert config["training_strategy"]["video"]["is_generated"]
    assert config["training_strategy"]["audio"]["is_generated"]
    assert config["validation"]["video_dims"] == [960, 544, 49]
    assert config["flow_matching"]["timestep_sampling_mode"] == (
        "shifted_logit_normal"
    )


def test_ltx23_video_only_config_and_preprocess_command(tmp_path):
    spec, trainer = _fake_install(tmp_path)
    spec = LTX23LoraSpec(**{**spec.__dict__, "with_audio": False})
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"video":"clip.mp4","caption":"example"}\n')
    command = preprocess_command(
        spec,
        dataset,
        runner="python",
        python=sys.executable,
        lora_trigger="SUBJECT",
        vae_tiling=True,
    )
    assert command[:2] == [
        sys.executable,
        str(trainer / "scripts" / "process_dataset.py"),
    ]
    assert "--skip-audio" in command
    assert command[command.index("--lora-trigger") + 1] == "SUBJECT"
    assert "audio" not in spec.official_config()["training_strategy"]


def test_ltx23_multi_gpu_train_command(tmp_path):
    spec, _ = _fake_install(tmp_path, preprocessed=True)
    config = tmp_path / "run.yaml"
    command = train_command(
        spec,
        config,
        runner="python",
        python="/venv/bin/python",
        processes=4,
        disable_progress_bars=True,
    )
    assert command[:3] == [
        "/venv/bin/python",
        "-m",
        "accelerate.commands.launch",
    ]
    assert command[command.index("--num_processes") + 1] == "4"
    assert command[-1] == "--disable-progress-bars"


def test_ltx23_preprocessed_contract(tmp_path):
    spec, _ = _fake_install(tmp_path, preprocessed=True)
    spec.validate(require_preprocessed=True)
    (spec.preprocessed_data_root / "audio_latents").rmdir()
    with pytest.raises(ValueError, match="audio_latents"):
        spec.validate(require_preprocessed=True)


def test_ltx23_run_files_include_config_hash_and_revision(tmp_path):
    spec, _ = _fake_install(tmp_path)
    config_path = tmp_path / "work" / "ltx23_lora.yaml"
    receipt_path = write_run_files(
        spec,
        config_path,
        prepare=["prepare"],
        train=["train"],
    )
    config = yaml.safe_load(config_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert config["lora"]["rank"] == 32
    assert receipt["schema"] == "rwkv-lab.ltx23-lora.v1"
    assert len(receipt["config_sha256"]) == 64
    assert receipt["commands"] == {"prepare": ["prepare"], "train": ["train"]}


def test_ltx23_plan_cli_writes_reproducible_artifacts(tmp_path):
    spec, _ = _fake_install(tmp_path)
    work = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--ltx-root",
                str(spec.ltx_root),
                "--model-path",
                str(spec.model_path),
                "--text-encoder-path",
                str(spec.text_encoder_path),
                "--work-dir",
                str(work),
                "--runner",
                "python",
                "--low-vram",
                "--no-audio",
            ]
        )
        == 0
    )
    config = yaml.safe_load((work / "ltx23_lora.yaml").read_text())
    assert config["lora"]["rank"] == 16
    assert config["optimization"]["optimizer_type"] == "adamw8bit"
    assert config["acceleration"]["quantization"] == "int8-quanto"
    assert "audio" not in config["training_strategy"]
