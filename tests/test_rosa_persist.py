import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def test_rosa_soft_scale_persists():
    pytest.importorskip("rosa_soft", reason="requires the optional rosa_soft extension")

    from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController
    from rwkv_lab import convert_train
    from rwkv_lab.rosa_soft_layer import RosaAnchorLayer

    out = Path(tempfile.mkdtemp(prefix="rosa_persist_test_"))
    args = types.SimpleNamespace(
        prior_rwkv_layers="",
        model_dir="m",
        patch_dir="",
        layer=16,
        save_optimizer=False,
        optimizer="spectral_muon",
    )
    student, codec = nn.Linear(4, 4), nn.Linear(4, 4)
    rosa = RosaAnchorLayer(256, M=4, window_size=32, route_dim=128)
    rosa.scale = 0.0421
    ctl = RosaAnchorScaleController(RosaAnchorScaleConfig(seq_len=1024, qk_bits=4, window_size=32))
    ctl.top_prob_ema = 0.7311

    sd = convert_train._save_ckpt(out, 500, student, codec, args, opt=None, rosa_soft=rosa, rosa_ctl=ctl)
    blob = torch.load(sd / "ckpt.pt", map_location="cpu", weights_only=False)
    assert blob["rosa_soft_scale"] == 0.0421, blob.get("rosa_soft_scale")
    assert blob["rosa_ctl_ema"] == 0.7311, blob.get("rosa_ctl_ema")
    assert "rosa_soft" in blob

    rosa_restore = {"scale": blob.get("rosa_soft_scale"), "ema": blob.get("rosa_ctl_ema")}
    ctl2 = RosaAnchorScaleController(
        RosaAnchorScaleConfig(
            seq_len=1024,
            qk_bits=4,
            window_size=32,
            initial_scale=rosa_restore.get("scale"),
        )
    )
    ctl2.top_prob_ema = float(rosa_restore["ema"])
    assert ctl2.scale == 0.0421 and ctl2.top_prob_ema == 0.7311

    convert_train._save_best(out, 501, 9.9, student, codec, args, opt=None, rosa_soft=rosa, rosa_ctl=ctl)
    convert_train._flush_best_saves()
    bb = torch.load(out / "best" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert bb["rosa_soft_scale"] == 0.0421 and bb["rosa_ctl_ema"] == 0.7311
