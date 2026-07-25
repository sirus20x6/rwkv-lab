import torch
import torch.nn.functional as F
import pytest
from torch import nn

from rwkv_lab.distributed import sparse_sync_parameter_rows
from rwkv_lab.fused_channelmix import (
    fused_channel_mix,
    qualify_channelmix_training,
)
from rwkv_lab.optimizer_speedups import (
    TailEMA,
    split_tied_embedding_head,
    tail_linear_multiplier,
    tie_embedding_head,
)
from rwkv_lab.rwkv8_deltanet import RWKV8ChannelMixDeltaNet
from rwkv_lab.spectral_muon import SpectralMuon


def test_tail_multiplier_and_partial_tail_ema():
    assert tail_linear_multiplier(20, 100, 0.5) == 1.0
    assert tail_linear_multiplier(75, 100, 0.5) == 0.5
    assert tail_linear_multiplier(100, 100, 0.5) == 0.0

    parameter = nn.Parameter(torch.tensor([1.0]))
    ema = TailEMA([("weight", parameter)], start_step=2, horizon=2, blend=0.5)
    ema.update(1)
    assert not ema.active
    parameter.data.fill_(3.0)
    ema.update(2)  # initialize at 3
    parameter.data.fill_(5.0)
    ema.update(3)  # shadow = 4
    backup = ema.swap_for_eval()
    torch.testing.assert_close(parameter, torch.tensor([4.5]))
    ema.restore(backup)
    torch.testing.assert_close(parameter, torch.tensor([5.0]))


class _TinyTiedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)
        self.head = nn.Linear(4, 8, bias=False)


def test_tied_head_split_preserves_group_topology_and_moments():
    model = _TinyTiedLM()
    tie_embedding_head(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    group_count = len(optimizer.param_groups)
    model.emb.weight.grad = torch.ones_like(model.emb.weight)
    optimizer.step()
    old_state = optimizer.state[model.emb.weight]["exp_avg"].clone()

    new_head = split_tied_embedding_head(model, optimizer)
    assert model.head.weight is new_head
    assert model.head.weight is not model.emb.weight
    assert len(optimizer.param_groups) == group_count
    torch.testing.assert_close(optimizer.state[new_head]["exp_avg"], old_state)


def test_fused_channelmix_matches_eager_forward_and_backward():
    torch.manual_seed(2)
    x_ref = torch.randn(2, 3, 5, requires_grad=True)
    key_ref = torch.randn(11, 5, requires_grad=True)
    value_ref = torch.randn(5, 11, requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_()
    key_fast = key_ref.detach().clone().requires_grad_()
    value_fast = value_ref.detach().clone().requires_grad_()

    expected = F.linear(F.relu(F.linear(x_ref, key_ref)).square(), value_ref)
    actual = fused_channel_mix(x_fast, key_fast, value_fast)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(actual, expected)
    for left, right in (
        (x_fast.grad, x_ref.grad),
        (key_fast.grad, key_ref.grad),
        (value_fast.grad, value_ref.grad),
    ):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_channelmix_cached_fp8_falls_back_exactly_on_cpu():
    torch.manual_seed(3)
    module = RWKV8ChannelMixDeltaNet(8, 16)
    x = torch.randn(2, 4, 8)
    expected = module(x)
    module.enable_fused_training(cached_fp8_up=True)
    actual = module(x)
    torch.testing.assert_close(actual, expected)
    assert module._key_weight_fp8.numel() == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cached_fp8_channelmix_forward_and_bf16_backward_are_close():
    torch.manual_seed(31)
    reference = RWKV8ChannelMixDeltaNet(64, 128).cuda().bfloat16().train()
    candidate = RWKV8ChannelMixDeltaNet(64, 128).cuda().bfloat16().train()
    candidate.load_state_dict(reference.state_dict())
    candidate.enable_fused_training(cached_fp8_up=True)
    x_ref = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_()
    expected = reference(x_ref)
    actual = candidate(x_fast)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=6e-2, atol=3e-2)
    torch.testing.assert_close(x_fast.grad, x_ref.grad, rtol=6e-2, atol=3e-2)
    # W2 backward is BF16-identical modulo the FP8 forward activation.
    torch.testing.assert_close(
        candidate.value.weight.grad,
        reference.value.weight.grad,
        rtol=8e-2,
        atol=4e-2,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_channelmix_qualification_initializes_and_checks_cached_fp8():
    module = RWKV8ChannelMixDeltaNet(64, 128).cuda().bfloat16().train()
    sample = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
    report = qualify_channelmix_training(
        module, sample, allow_cached_fp8=True, repeats=1
    )
    assert report["adopted"] in report["times_ms"]
    assert hasattr(module, "_key_weight_fp8")
    assert not any(
        "_key_weight_fp8" in reason for reason in report["rejected"].values()
    )


def test_muon_fallback_adam_cadence_averages_skipped_gradients():
    initial = torch.tensor([1.0, -1.0])
    cadenced = nn.Parameter(initial.clone())
    reference = nn.Parameter(initial.clone())
    slow = SpectralMuon(
        [{"params": [cadenced], "use_muon": False, "lr": 0.01}],
        adam_update_interval=2,
        weight_decay=0.0,
    )
    ordinary = SpectralMuon(
        [{"params": [reference], "use_muon": False, "lr": 0.01}],
        weight_decay=0.0,
    )
    cadenced.grad = torch.ones_like(cadenced)
    slow.step()
    torch.testing.assert_close(cadenced, initial)
    cadenced.grad = torch.full_like(cadenced, 3.0)
    slow.step()
    reference.grad = torch.full_like(reference, 2.0)
    ordinary.step()
    torch.testing.assert_close(cadenced, reference)


def test_muon_shape_postconditioning_is_finite_and_opt_in():
    torch.manual_seed(4)
    base = nn.Parameter(torch.randn(6, 4))
    shaped = nn.Parameter(base.detach().clone())
    plain = SpectralMuon(
        [{"params": [base], "use_muon": True, "lr": 0.01}],
        ns_steps=2,
        weight_decay=0.1,
    )
    enhanced = SpectralMuon(
        [{"params": [shaped], "use_muon": True, "lr": 0.01}],
        ns_steps=2,
        weight_decay=0.1,
        row_update_floor=0.3,
        radial_brake=0.5,
        radius_pin=True,
        cautious_weight_decay=True,
    )
    gradient = torch.randn_like(base)
    base.grad = gradient.clone()
    shaped.grad = gradient.clone()
    plain.step()
    enhanced.step()
    assert torch.isfinite(shaped).all()
    assert not torch.equal(base, shaped)


def test_sparse_row_sync_is_noop_without_process_group():
    parameter = nn.Parameter(torch.zeros(7, 3))
    parameter.grad = torch.zeros(7, 3)
    parameter.grad[1] = 1.0
    parameter.grad[4] = 2.0
    before = parameter.grad.clone()
    assert sparse_sync_parameter_rows(parameter, torch.tensor([1, 1, 4])) == 2
    torch.testing.assert_close(parameter.grad, before)


def test_sparse_row_sync_counts_every_row_it_would_transmit():
    """The count reports rows that must be sent, not the size of the hint.

    A dense gradient has to be transmitted in full: reporting the hint's size
    here would be the same mistake that let real gradient rows be zeroed.
    """
    parameter = nn.Parameter(torch.zeros(7, 3))
    parameter.grad = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    assert sparse_sync_parameter_rows(parameter, torch.tensor([1, 1, 4])) == 7


def test_lm_launcher_propagates_mined_speedup_flags():
    from rwkv_lab.config import _lm_command

    command = _lm_command(
        ["--data", "tokens.bin"],
        None,
        "out",
        {},
        {
            "steps": 10,
            "optimizer": "muon",
            "fp8": True,
            "fp8_head": True,
            "fused_channelmix": True,
            "cached_fp8_up": True,
            "compile": True,
            "compile_fullgraph": True,
            "fsdp_sparse_embeddings": True,
            "distributed": "fsdp2",
            "tail_ema_start": 0.8,
            "tie_head_until": 0.66,
            "lmtp_cooldown_fraction": 0.5,
            "muon": {
                "row_update_floor": 0.3,
                "cautious_weight_decay": True,
                "adam_update_interval": 2,
            },
        },
        {},
        0,
        "save.pt",
    )
    for flag in (
        "--fp8-head",
        "--fused-channelmix",
        "--cached-fp8-up",
        "--compile-fullgraph",
        "--fsdp-sparse-embeddings",
        "--tail-ema-start",
        "--tie-head-until",
        "--lmtp-cooldown-fraction",
        "--sm-row-update-floor",
        "--sm-cautious-wd",
        "--muon-adam-interval",
    ):
        assert flag in command


def test_sparse_row_sync_keeps_rows_the_caller_did_not_list():
    """The row hint is an optimization, never the definition of what to send.

    This helper zeroes the gradient and rebuilds it from the transmitted rows,
    so trusting an incomplete hint silently discarded real gradient. The touched
    set is now derived from the gradient itself and the hint is only unioned in.
    """
    param = nn.Parameter(torch.zeros(10, 4))
    param.grad = torch.zeros(10, 4)
    param.grad[3] = 1.0
    param.grad[7] = 2.0            # reachable, but absent from the hint
    expected = param.grad.clone()

    transmitted = sparse_sync_parameter_rows(param, torch.tensor([3]))

    assert transmitted == 2
    assert torch.equal(param.grad, expected)


@pytest.mark.parametrize("shape", [(6,), (6, 3), (6, 3, 2)])
def test_sparse_row_sync_handles_every_replicated_table_rank(shape):
    param = nn.Parameter(torch.zeros(*shape))
    param.grad = torch.zeros(*shape)
    param.grad[4] = 1.0
    assert sparse_sync_parameter_rows(param, torch.tensor([], dtype=torch.long)) == 1


def test_tail_ema_excludes_whole_name_components_not_substrings():
    """"emb" must exclude the token table without also eating `loop_index_embed`."""
    model = nn.Module()
    model.emb = nn.Embedding(4, 3)
    model.loop_index_embed = nn.Parameter(torch.zeros(2, 3))
    model.head = nn.Linear(3, 4, bias=False)

    ema = TailEMA(model.named_parameters(), start_step=0, horizon=4, blend=0.5)
    tracked = {name for name, _ in ema.named}

    assert "emb.weight" not in tracked
    assert "loop_index_embed" in tracked
    assert "head.weight" in tracked


def test_tail_ema_resumes_into_a_model_that_gained_a_parameter():
    """A shadow saved before a parameter existed must seed it, not KeyError."""
    small = nn.Sequential(nn.Linear(3, 3))
    source = TailEMA(small.named_parameters(), start_step=0, horizon=4, blend=0.5)
    source.update(0)
    source.update(1)

    grown = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    target = TailEMA(grown.named_parameters(), start_step=0, horizon=4, blend=0.5)
    target.load_state_dict(source.state_dict())
    target.update(2)

    assert set(target.shadow) == {name for name, _ in target.named}
    target.restore(target.swap_for_eval())
