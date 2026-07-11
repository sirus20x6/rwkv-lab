import torch

from rwkv_lab.compositional_muon import compositional_pair_update
from rwkv_lab.distillation_merge import merge_expert_states, merge_logit_loss, tolerance_win_tie
from rwkv_lab.key_value_means import initialize_state, read_memory, state_budget, update_state
from rwkv_lab.m2rnn import M2RNN
from rwkv_lab.mamba3_recurrence import Mamba3Recurrence
from rwkv_lab.procedural_memory import ProceduralMemory
from rwkv_lab.rosa_plus import WittenBellFallback, rosa_or_fallback
from rwkv_lab.test_time_training import TTTPolicy, guarded_test_time_train


def test_kvm_append_merge_budget_and_readout():
    torch.manual_seed(10)
    state = initialize_state(torch.eye(4), torch.randn(4, 3))
    state = update_state(state, torch.randn(5, 4), torch.randn(5, 3), budget=6, sink_rows=1)
    assert state.keys.shape == (6, 4) and state.values.shape == (6, 3)
    assert read_memory(torch.randn(2, 4), state).shape == (2, 3)
    assert state_budget(100, 4, "sqrt") == 14
    assert state_budget(100, 4, "saturating", maximum=8) == 8


def test_procedural_memory_contrast_retrieval_and_intervention():
    memory = ProceduralMemory()
    memory.add_contrast(torch.tensor([1.0, 0]), torch.ones(3, 4), torch.zeros(3, 4), source="task-a")
    memory.add_contrast(torch.tensor([0.0, 1]), -torch.ones(3, 4), torch.zeros(3, 4), source="task-b")
    selected = memory.retrieve(torch.tensor([1.0, 0]), top_k=1)
    assert selected[0].source == "task-a"
    hidden = torch.zeros(2, 4)
    assert torch.equal(memory.intervene(hidden, torch.tensor([1.0, 0]), top_k=1), torch.ones(2, 4))


def test_mamba3_complex_mimo_recurrence_has_state_carry_and_gradients():
    layer = Mamba3Recurrence(8, 5, complex_state=True, n_inputs=2, n_outputs=2)
    x = torch.randn(3, 6, 8, requires_grad=True)
    y, state = layer(x); y.sum().backward()
    assert y.shape == x.shape and state.shape == (3, 2, 10)
    assert layer.in_proj.weight.grad is not None


def test_m2rnn_nonlinear_matrix_state():
    layer = M2RNN(8, 6, rank=3)
    y, state = layer(torch.randn(2, 5, 8))
    assert y.shape == (2, 5, 8) and state.shape == (2, 6, 6)
    assert not torch.equal(state, torch.zeros_like(state))


def test_compositional_muon_updates_composed_operator():
    left, right = torch.randn(6, 4), torch.randn(7, 4)
    before = left @ right.T
    receipt = compositional_pair_update(left, right, torch.randn_like(left), torch.randn_like(right), lr=1e-3)
    assert receipt["operator_update_norm"] > 0 and not torch.equal(before, left @ right.T)


def test_distillation_merge_and_tolerance_corrected_win_tie():
    merged = merge_expert_states([{"w": torch.zeros(2, 2)}, {"w": torch.ones(2, 2)}])
    assert torch.allclose(merged["w"], torch.full((2, 2), 0.5))
    assert merge_logit_loss(torch.randn(2, 7), [torch.randn(2, 7), torch.randn(2, 7)]) >= 0
    report = tolerance_win_tie(torch.tensor([1.1, 1.0, 0.7]), torch.ones(3), tolerance=0.05)
    assert report == {"wins": 1, "ties": 1, "losses": 1, "win_and_tie_rate": 2 / 3,
                      "tolerance": 0.05}


def test_guarded_ttt_accepts_improvement_and_rolls_back_regression():
    model = torch.nn.Linear(1, 1, bias=False); model.weight.data.zero_()
    loss_fn = lambda m: (m(torch.ones(1, 1)) - 1).square().mean()
    evaluate = lambda m: -float(loss_fn(m))
    accepted = guarded_test_time_train(model, loss_fn, evaluate,
                                       TTTPolicy(steps=2, learning_rate=0.2, max_parameters=1))
    assert accepted["accepted"] and model.weight.item() > 0
    saved = model.weight.detach().clone()
    rejected = guarded_test_time_train(model, lambda m: -m(torch.ones(1, 1)).square().mean(),
                                       evaluate, TTTPolicy(steps=1, learning_rate=1.0,
                                                           max_parameters=1, max_regression=0))
    assert rejected["rolled_back"] and torch.equal(model.weight, saved)


def test_witten_bell_fallback_is_normalized_and_rosa_wins_when_matched():
    fallback = WittenBellFallback(3); fallback.observe("abracadabra")
    dist, source = rosa_or_fallback(-1, "x", fallback, "abr")
    assert source == "witten-bell" and abs(sum(dist.values()) - 1) < 1e-6
    assert rosa_or_fallback(2, "a", fallback, "abr") == ({"a": 1.0}, "rosa")
