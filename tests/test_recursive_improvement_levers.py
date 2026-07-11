"""Fast invariants for the P0-P2 model/training research levers."""
from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn

from rwkv_lab.byte_patches import entropy_patch_ids, pool_patches
from rwkv_lab.circuit_trace import aggregate_edges, trace_linear_recurrence
from rwkv_lab.data_mixture import MixtureObservation, MixtureSurrogate
from rwkv_lab.diloco import DecoupledDiLoCo, LearnerUpdate, pseudo_gradient
from rwkv_lab.nvfp4 import (NVFP4Linear, convert_to_nvfp4_training,
                            hadamard_transform, nvfp4_fake_quant)
from rwkv_lab.online_memory import OnlineAssociativeMemory
from rwkv_lab.rlvr import ExactAnswerVerifier, PythonExpressionVerifier, policy_loss
from rwkv_lab.speculative import EAGLE3DraftHead, verify_greedy_draft
from rwkv_lab.u_mup import UMuPConfig, initialize_u_mup, parameter_groups


def test_u_mup_preserves_zero_adapters_and_scales_optimizer_groups():
    m = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 8, bias=False))
    m[1].weight.data.zero_()
    cfg = UMuPConfig(base_width=8, width=16, depth=4, base_depth=2)
    manifest = initialize_u_mup(m, cfg)
    assert torch.count_nonzero(m[1].weight) == 0
    groups = parameter_groups(m.named_parameters(), lr=1e-3, weight_decay=.1, config=cfg)
    assert manifest and all("u_mup_lr_mult" in g for g in groups)


def test_online_memory_is_identity_at_init_and_stateful():
    mem = OnlineAssociativeMemory(8, d_memory=4, mode="atlas", atlas_window=3)
    x = torch.randn(2, 6, 8, requires_grad=True)
    y, state = mem(x, return_state=True)
    assert torch.equal(x, y)
    assert state.steps == 6 and state.memory.norm() > 0
    y.sum().backward()


def test_rlvr_objectives_and_safe_verifiers():
    logp = torch.tensor([[-.1, -.2], [-.2, -.3], [-.3, -.4], [-.4, -.5]], requires_grad=True)
    old = logp.detach() - .01
    rewards = torch.tensor([1., 0., 1., 1.]); groups = torch.tensor([0, 0, 1, 1])
    mask = torch.ones_like(logp)
    for algorithm in ("gspo", "dr_grpo", "dapo"):
        out = policy_loss(logp, old, rewards, groups, mask, algorithm=algorithm)
        assert torch.isfinite(out.loss)
    assert ExactAnswerVerifier("  Yes ")("yes") == 1
    assert PythonExpressionVerifier(7)("1 + 2 * 3") == 1
    assert PythonExpressionVerifier(7)("__import__('os')") == 0


def test_regmix_surrogate_finds_known_corner():
    obs = []
    for x in np.linspace(0, 1, 9):
        w = (x, 1 - x)
        obs.append(MixtureObservation(w, (x - .75) ** 2 + .1))
    model = MixtureSurrogate(ridge=1e-8).fit(obs)
    best, loss = model.search(candidates=2000, seed=3)
    assert abs(best[0] - .75) < .08 and loss < .12


def test_nvfp4_fake_quant_and_linear_gradients():
    x = torch.randn(3, 8, requires_grad=True)
    q = nvfp4_fake_quant(x, block_size=4)
    assert q.shape == x.shape and torch.unique(q.detach()).numel() > 1
    assert torch.allclose(hadamard_transform(hadamard_transform(x)), x, atol=1e-5)
    layer = NVFP4Linear(8, 4, rht=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None and x.grad is not None
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    assert convert_to_nvfp4_training(model, rht=True) == 2
    assert convert_to_nvfp4_training(model, rht=True) == 0


def test_decoupled_diloco_token_weighted_outer_step():
    reference = {"w": torch.tensor([1., 1.])}
    a, b = {"w": torch.tensor([0., 1.])}, {"w": torch.tensor([1., -1.])}
    updates = [LearnerUpdate("a", 0, 1, pseudo_gradient(reference, a)),
               LearnerUpdate("b", 0, 3, pseudo_gradient(reference, b))]
    params = {"w": nn.Parameter(reference["w"].clone())}
    outer = DecoupledDiLoCo(outer_lr=1, momentum=0)
    assert outer.apply(params, updates) == 2
    assert torch.allclose(params["w"], torch.tensor([.75, -.5]))


def test_byte_patch_pooling_and_unpool_mapping():
    entropy = torch.tensor([[0., 0., 5., 0., 0., 5.]])
    ids = entropy_patch_ids(entropy, threshold=4, min_patch=2, max_patch=3)
    assert ids.tolist() == [[0, 0, 1, 1, 1, 2]]
    x = torch.arange(12.).view(1, 6, 2)
    batch = pool_patches(x, ids)
    assert batch.unpool().shape == x.shape and batch.mask.sum() == 3


def test_eagle3_head_and_lossless_greedy_verification():
    head = EAGLE3DraftHead(8, 32, draft_steps=3)
    x = torch.randn(2, 5, 8)
    assert head(x, x, x).shape == (2, 5, 3, 32)
    target = lambda context: (sum(context) + 1) % 7
    tokens, accepted = verify_greedy_draft([1], [2, 0, 0], target_next=target)
    assert tokens == [2, 4] and accepted == 1


def test_recurrent_circuit_edges_exactly_reconstruct_reads():
    torch.manual_seed(4)
    T, K, V = 5, 3, 2
    reads, keys, values = torch.randn(T, K), torch.randn(T, K), torch.randn(T, V)
    decays = torch.sigmoid(torch.randn(T, K))
    edges = trace_linear_recurrence(reads, keys, values, decays)
    traced = aggregate_edges(edges, T, V)
    state, direct = torch.zeros(K, V), []
    for t in range(T):
        state = decays[t, :, None] * state + keys[t, :, None] * values[t]
        direct.append(reads[t] @ state)
    assert torch.allclose(traced, torch.stack(direct), atol=1e-5)


def test_declarative_lm_command_translates_new_flag_kinds():
    from rwkv_lab.config import _lm_command
    from rwkv_lab.experiment import LEVERS
    cmd = _lm_command(["--data", "x.bin"], None, "out", {}, {},
                      {"nvfp4": True, "nvfp4_rht": False, "online_memory": True,
                       "online_memory_mode": "atlas", "online_memory_kernel": "eager",
                       "u_mup_base_width": 256},
                      0, "ckpt.pt")
    assert cmd.count("--nvfp4") == 1 and "--nvfp4-rht" not in cmd
    assert cmd[cmd.index("--online-memory") + 1] == "1"
    assert cmd[cmd.index("--online-memory-mode") + 1] == "atlas"
    assert cmd[cmd.index("--online-memory-kernel") + 1] == "eager"
    atlas = _lm_command(["--data", "x.bin"], None, "out", {}, {}, LEVERS["mem_atlas"], 0, "c.pt")
    nv_rht = _lm_command(["--data", "x.bin"], None, "out", {}, {}, LEVERS["nvfp4_rht"], 0, "c.pt")
    nv_native = _lm_command(["--data", "x.bin"], None, "out", {}, {}, LEVERS["nvfp4_native"], 0, "c.pt")
    assert atlas[atlas.index("--online-memory-mode") + 1] == "atlas"
    assert "--nvfp4" in nv_rht and "--nvfp4-rht" in nv_rht
    assert nv_native[nv_native.index("--nvfp4-backend") + 1] == "transformer_engine"


def test_lm_command_uses_torchrun_for_fsdp2_and_system_controls():
    from rwkv_lab.config import _lm_command
    command = _lm_command(
        ["--data", "x.bin"], None, "out", {},
        {"distributed": "fsdp2", "world_size": 4, "activation_checkpointing": True,
         "cpu_offload": True, "lr_schedule": "constant", "decay_steps": 99,
         "optimizer": "muon", "muon": {"aro": True, "aro_compile": True}},
        {}, 0, "ckpt.pt")
    assert command[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert command[command.index("--nproc-per-node") + 1] == "4"
    for flag in ("--distributed", "--activation-checkpointing", "--cpu-offload",
                 "--lr-schedule", "--decay-steps", "--sm-aro", "--sm-aro-compile"):
        assert flag in command
