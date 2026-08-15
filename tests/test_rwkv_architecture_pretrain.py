from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from rwkv_lab.build_corpus import build_bytes
from rwkv_lab.config import _lm_command, _validate_lm_architecture_config
from rwkv_lab.rwkv_pretrain import (
    BYTE_PRETRAIN_ARCHITECTURES,
    build_prototype_pretrain_model,
    pretrain_architecture_identity,
    validate_pretrain_resume_architecture,
)


ARCHITECTURES = ("rwkv8", "blt_rwkv7", "rosa_blt", "kan_rwkv")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_imported_architecture_has_pretrainer_hidden_contract(architecture):
    model = build_prototype_pretrain_model(
        architecture,
        vocab_size=256,
        d_model=16,
        n_layers=2,
        head_size=16,
        deepembed=architecture in ("rwkv8", "kan_rwkv"),
    )
    ids = torch.randint(0, 256, (2, 8))
    if architecture in BYTE_PRETRAIN_ARCHITECTURES:
        hidden, auxiliary = model(
            ids, hidden_only=True, target_bytes=ids, return_aux=True
        )
    else:
        hidden, auxiliary = model(ids, hidden_only=True), None
    loss = F.cross_entropy(model.head(hidden[:, :-1]).reshape(-1, 256), ids[:, 1:].reshape(-1))
    if auxiliary is not None:
        loss = loss + 0.1 * auxiliary
    loss.backward()
    assert hidden.shape == (2, 8, 16)
    assert torch.isfinite(loss)
    assert model.emb.weight.grad is not None
    if architecture in BYTE_PRETRAIN_ARCHITECTURES:
        assert all(block.ffn.entropy_head.weight.grad is not None for block in model.blocks)


def test_byte_corpus_preserves_utf8_and_document_offsets(tmp_path):
    prefix = tmp_path / "bytes"
    documents = ["hello", "κόσμε", "世界"]
    build_bytes(documents, str(prefix))
    tokens = np.fromfile(prefix.with_suffix(".bin"), dtype=np.uint16)
    offsets = np.load(prefix.with_suffix(".off.npy"))
    expected = b"".join(document.encode("utf-8") for document in documents)
    assert bytes(tokens.astype(np.uint8)) == expected
    assert offsets.tolist() == [0, 5, 5 + len("κόσμε".encode("utf-8"))]
    assert int(tokens.max()) < 256


@pytest.mark.parametrize("architecture", ["rwkv8", "kan_rwkv"])
@pytest.mark.parametrize("de_dim", [0, 4])
def test_prototype_deepembed_is_identity_initialized_and_trainable(architecture, de_dim):
    model = build_prototype_pretrain_model(
        architecture,
        vocab_size=32,
        d_model=16,
        n_layers=2,
        head_size=16,
        deepembed=True,
        de_dim=de_dim,
    )
    ids = torch.randint(0, 32, (1, 4))
    for block in model.blocks:
        gate_values = block.de_emb(ids)
        if block.de_proj is not None:
            assert torch.count_nonzero(block.de_emb.weight) > 0
            gate_values = block.de_proj(gate_values)
        assert torch.count_nonzero(gate_values) == 0

    hidden = model(ids, hidden_only=True)
    loss = model.head(hidden).square().mean()
    loss.backward()
    for block in model.blocks:
        output_parameter = block.de_proj.weight if block.de_proj is not None else block.de_emb.weight
        assert output_parameter.grad is not None
        assert torch.count_nonzero(output_parameter.grad) > 0


def test_declarative_byte_contract_fails_before_corpus_build():
    cfg = {
        "data": {"encoding": "world", "sources": []},
        "model": {"vocab_size": 256},
        "configs": {"blt": {"architecture": "blt_rwkv7"}},
    }
    with pytest.raises(ValueError, match="data.encoding: bytes"):
        _validate_lm_architecture_config(cfg)


def test_declarative_command_contains_architecture_knobs():
    command = _lm_command(
        ["--data", "corpus.bin"],
        None,
        "runs/test",
        {
            "architecture": "kan_rwkv",
            "vocab_size": 256,
            "d_model": 32,
            "n_layers": 2,
            "head_size": 16,
            "kan_grid_size": 7,
            "kan_spline_order": 2,
        },
        {"steps": 1, "batch": 1, "seq_len": 8},
        {},
        0,
        "runs/test/ckpt.pt",
    )
    joined = " ".join(command)
    assert "--architecture kan_rwkv" in joined
    assert "--vocab-size 256" in joined
    assert "--kan-grid-size 7" in joined
    assert "--kan-spline-order 2" in joined


def _identity(architecture="rwkv8"):
    return pretrain_architecture_identity(
        architecture=architecture,
        vocab_size=256,
        d_model=16,
        n_layers=2,
        head_size=16,
        deepembed=False,
        de_dim=0,
        blt_threshold=3.5,
        blt_max_patch=16,
        kan_grid_size=5,
        kan_spline_order=3,
        rosa_memory_size=4,
    )


def test_resume_architecture_identity_is_fail_closed():
    current = _identity()
    validate_pretrain_resume_architecture(current, current)
    incompatible = {**current, "architecture": "kan_rwkv"}
    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        validate_pretrain_resume_architecture(incompatible, current)


def _trainer_command(corpus: Path, out: Path, checkpoint: Path, architecture: str, steps: int,
                     resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rwkv_lab.rwkv_pretrain",
        "--data",
        str(corpus),
        "--out",
        str(out),
        "--save",
        str(checkpoint),
        "--architecture",
        architecture,
        "--vocab-size",
        "256",
        "--d-model",
        "16",
        "--n-layers",
        "2",
        "--head-size",
        "16",
        "--seq-len",
        "8",
        "--batch",
        "1",
        "--steps",
        str(steps),
        "--val-windows",
        "1",
        "--eval-every",
        "1",
        "--log-every",
        "1",
        "--warmup",
        "0",
        "--gpu-data",
        "off",
        "--no-cpu-prefetch",
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def _run_trainer(command: list[str], cwd: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(cwd / "src"),
        "RWKV8_FORCE_PYREF": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=cwd, env=environment, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_imported_architecture_runs_real_pretrainer(tmp_path, architecture):
    corpus = tmp_path / "corpus.bin"
    np.arange(192, dtype=np.uint16).tofile(corpus)
    out = tmp_path / architecture
    checkpoint = out / "checkpoint.pt"
    root = Path(__file__).resolve().parents[1]
    _run_trainer(_trainer_command(corpus, out, checkpoint, architecture, 1), root)

    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert rows[0]["kind"] == "run_config"
    assert rows[0]["architecture"] == architecture
    assert any(row["kind"] == "eval" and row["step"] == 0 for row in rows)
    assert any(row["kind"] == "train" and row["step"] == 1 for row in rows)
    assert rows[-1]["kind"] == "checkpoint"
    assert checkpoint_blob["arch"]["architecture"] == architecture
    assert checkpoint_blob["arch"]["vocab_size"] == 256
    if architecture in BYTE_PRETRAIN_ARCHITECTURES:
        train_row = next(row for row in rows if row["kind"] == "train")
        assert train_row["blt_entropy_loss"] > 0


def test_rwkv8_split_resume_matches_uninterrupted_training(tmp_path):
    corpus = tmp_path / "corpus.bin"
    np.arange(192, dtype=np.uint16).tofile(corpus)
    root = Path(__file__).resolve().parents[1]

    full_out = tmp_path / "full"
    full_checkpoint = full_out / "checkpoint.pt"
    _run_trainer(_trainer_command(corpus, full_out, full_checkpoint, "rwkv8", 2), root)

    split_out = tmp_path / "split"
    split_checkpoint = split_out / "checkpoint.pt"
    _run_trainer(_trainer_command(corpus, split_out, split_checkpoint, "rwkv8", 1), root)
    _run_trainer(
        _trainer_command(
            corpus, split_out, split_checkpoint, "rwkv8", 2, resume=split_checkpoint
        ),
        root,
    )

    full = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
    split = torch.load(split_checkpoint, map_location="cpu", weights_only=False)
    assert full["step"] == split["step"] == 2
    assert full["arch"] == split["arch"]
    assert full["model"].keys() == split["model"].keys()
    for name in full["model"]:
        torch.testing.assert_close(full["model"][name], split["model"][name], rtol=0, atol=0)
    rows = [json.loads(line) for line in (split_out / "train.jsonl").read_text().splitlines()]
    assert sum(row["kind"] == "run_config" for row in rows) == 2
    assert rows[-1]["kind"] == "checkpoint"


def test_declarative_architecture_campaign_reaches_registry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "corpus.txt"
    source.write_text("A real declarative byte-corpus training document. " * 64)
    database = tmp_path / "trainboard.db"
    runs = tmp_path / "runs"
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "architecture integration test",
                "registry_db": str(database),
                "runs_dir": str(runs),
                "data": {
                    "encoding": "bytes",
                    "sources": [
                        {
                            "kind": "local",
                            "patterns": [str(source)],
                            "weight": 1.0,
                        }
                    ],
                    "doc_boundary": True,
                    "cap_mb": 1,
                },
                "seeds": 1,
                "model": {
                    "architecture": "rwkv7",
                    "vocab_size": 256,
                    "d_model": 16,
                    "n_layers": 2,
                    "head_size": 16,
                },
                "train": {
                    "steps": 1,
                    "batch": 1,
                    "seq_len": 8,
                    "warmup": 0,
                    "val_windows": 1,
                    "eval_every": 1,
                    "log_every": 1,
                    "gpu_data": "off",
                    "cpu_prefetch": False,
                },
                "configs": {
                    "baseline": {},
                    "blt": {"architecture": "blt_rwkv7"},
                },
            }
        )
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "RWKV8_FORCE_PYREF": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "rwkv_lab.config", "run", str(config_path)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT status FROM campaigns").fetchone()[0] == "complete"
    trials = connection.execute(
        "SELECT status, metrics_json, series_json FROM trials ORDER BY id"
    ).fetchall()
    assert len(trials) == 2
    assert all(status == "complete" for status, _, _ in trials)
    assert all(json.loads(metrics)["step"] == 1 for _, metrics, _ in trials)
    assert all(json.loads(series) for _, _, series in trials)
    assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 4
    dashboard_logs = sorted(runs.glob("*/train.jsonl"))
    assert len(dashboard_logs) == 2
    assert all(path.parent.parent == runs for path in dashboard_logs)
