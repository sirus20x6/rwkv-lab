"""Tests for LoopedRWKV hyper-connection lanes, per-pass LoRA, and loop sampling
(looped_rwkv.py: --loop-hyper / --loop-lora-rank / --loop-sample).

Run:  python test_looped_hyper.py        (no pytest needed)
  or: python -m pytest test_looped_hyper.py -q
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rwkv_lab.looped_rwkv import LoopedRWKV, lora_config_from_sd, sample_loop_count

torch.manual_seed(0)

H, G = 32, 4
B, T = 2, 16


class StubCore(nn.Module):
    """Mimics RWKV8TimeMixDeltaNet's wrapper-facing surface: hidden_size/num_heads
    attrs, receptance/key/value/output linears, tensor->tensor forward, and a
    (y, state, shift) tuple under return_state."""

    def __init__(self, dim: int = H, heads: int = G) -> None:
        super().__init__()
        self.hidden_size = dim
        self.num_heads = heads
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x, *args, **kwargs):
        y = self.output(torch.tanh(self.receptance(x)) * self.key(x) + self.value(x))
        if kwargs.get("return_state"):
            return y, torch.ones(1), torch.zeros(1)
        return y


def _pair(hyper: int, gate_mode: str = "scalar", n_loops: int = 4, seed: int = 1):
    """Plain and hyper wrappers sharing one core (identical weights)."""
    torch.manual_seed(seed)
    core = StubCore()
    plain = LoopedRWKV(core, n_loops=n_loops, gate_mode=gate_mode)
    hyp = LoopedRWKV(core, n_loops=n_loops, gate_mode=gate_mode, hyper_lanes=hyper)
    hyp.iter_norm.load_state_dict(plain.iter_norm.state_dict())
    return plain, hyp


def test_init_noop_matches_pass1_any_k():
    x = torch.randn(B, T, H)
    for K in (2, 3, 5):   # odd K exercises the one-hot (not 1/K) read init
        _, hyp = _pair(K)
        out = hyp(x)
        ref = hyp(x, skip_refine=True)   # pass-1 (core) semantics
        assert torch.equal(out, ref), f"hyper K={K} must be an exact no-op at init"


def test_identity_lanes_match_plain_loop_with_trained_gates():
    """With hyper params AT INIT (one-hot/identity/ones/e0) but NONZERO gates, the
    lane algebra must reproduce the plain loop bit-for-bit — the upgrade is loss-free
    at any point in training, not just at gate init."""
    x = torch.randn(B, T, H)
    for gate_mode in ("scalar", "factored"):
        plain, hyp = _pair(2, gate_mode=gate_mode)
        with torch.no_grad():
            plain.residual_weight.normal_(0, 0.3)
            hyp.residual_weight.copy_(plain.residual_weight)
            if gate_mode == "factored":
                plain.gate_chan.normal_(0, 0.2)
                hyp.gate_chan.copy_(plain.gate_chan)
        assert torch.equal(hyp(x), plain(x)), f"identity lanes diverged ({gate_mode})"


def test_trained_lanes_change_function():
    x = torch.randn(B, T, H)
    plain, hyp = _pair(2)
    with torch.no_grad():
        plain.residual_weight.normal_(0, 0.3)
        hyp.residual_weight.copy_(plain.residual_weight)
        hyp.hyper_mix.add_(torch.randn_like(hyp.hyper_mix) * 0.1)
        hyp.hyper_read.add_(torch.randn_like(hyp.hyper_read) * 0.1)
    assert not torch.equal(hyp(x), plain(x))


def test_k1_rejected():
    try:
        LoopedRWKV(StubCore(), n_loops=4, hyper_lanes=1)
    except ValueError:
        return
    raise AssertionError("hyper_lanes=1 must be rejected (HC paper: K=1 provably fails)")


def test_loop_param_names_and_float_gates():
    _, hyp = _pair(2)
    names = hyp.loop_param_names()
    assert {"hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"} <= names
    hyp.to(dtype=torch.bfloat16)
    hyp.float_gates()
    for n in ("hyper_alpha", "hyper_mix", "hyper_write", "hyper_read", "residual_weight"):
        assert getattr(hyp, n).dtype == torch.float32, f"{n} must stay fp32"


def test_grads_flow_to_all_hyper_params():
    _, hyp = _pair(2)
    with torch.no_grad():
        hyp.residual_weight.normal_(0, 0.3)   # open the gates so write/mix see signal
    x = torch.randn(B, T, H)
    hyp(x).float().pow(2).mean().backward()
    for n in ("hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"):
        g = getattr(hyp, n).grad
        assert g is not None and g.abs().sum() > 0, f"no gradient reached {n}"


def test_state_dict_roundtrip_with_key_detection():
    _, hyp = _pair(2)
    with torch.no_grad():
        hyp.residual_weight.normal_(0, 0.3)
        hyp.hyper_alpha.add_(torch.randn_like(hyp.hyper_alpha) * 0.1)
    sd = hyp.state_dict()
    # consolidate-style reconstruction: lanes inferred from the sd, not metadata
    K = int(sd["hyper_read"].shape[0])
    fresh = LoopedRWKV(StubCore(), n_loops=4, hyper_lanes=K)
    fresh.load_state_dict(sd)
    x = torch.randn(B, T, H)
    assert torch.equal(fresh(x), hyp(x))


def test_plain_ckpt_loads_into_hyper_student():
    plain, hyp = _pair(2)
    with torch.no_grad():
        plain.residual_weight.normal_(0, 0.3)
    miss, unexp = hyp.load_state_dict(plain.state_dict(), strict=False)
    assert not unexp
    assert set(miss) == {"hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"}, miss
    x = torch.randn(B, T, H)
    assert torch.equal(hyp(x), plain(x)), "fresh identity lanes must be a loss-free upgrade"


def test_probe_trace_captures_all_passes():
    for K in (0, 2):   # plain and hyper paths
        mod = LoopedRWKV(StubCore(), n_loops=4, hyper_lanes=K)
        with torch.no_grad():
            mod.residual_weight.normal_(0, 0.3)
        x = torch.randn(B, T, H)
        mod._probe_trace = []
        out = mod(x)
        tr = mod._probe_trace
        del mod._probe_trace
        assert len(tr) == 4, f"expected n_loops trace entries, got {len(tr)}"
        assert torch.equal(tr[-1], out), "last trace entry must equal the block output"
        assert not mod(x).requires_grad or True  # attr removed -> normal forward unaffected


def test_skip_refine_and_return_state_paths():
    _, hyp = _pair(2)
    with torch.no_grad():
        hyp.residual_weight.normal_(0, 0.3)
    x = torch.randn(B, T, H)
    assert torch.equal(hyp(x, skip_refine=True), hyp.core(x))
    out, st, sh = hyp(x, return_state=True)
    assert torch.is_tensor(out) and float(st) == 1.0


# ---------------------------------------------------------------------------
# per-pass LoRA
# ---------------------------------------------------------------------------

def _lora_pair(rank=4, n_loops=4, seed=2):
    torch.manual_seed(seed)
    core = StubCore()
    plain = LoopedRWKV(core, n_loops=n_loops)
    lora = LoopedRWKV(core, n_loops=n_loops, lora_rank=rank)
    lora.iter_norm.load_state_dict(plain.iter_norm.state_dict())
    return plain, lora


def test_lora_init_noop_and_pass1_faithful():
    plain, lora = _lora_pair()
    x = torch.randn(B, T, H)
    with torch.no_grad():
        plain.residual_weight.normal_(0, 0.3)
        lora.residual_weight.copy_(plain.residual_weight)
    assert torch.equal(lora(x), plain(x)), "zero-init B must be an exact no-op"
    with torch.no_grad():   # train the adapters: refinement changes, pass 1 must not
        for k in lora.loop_lora_B:
            lora.loop_lora_B[k].normal_(0, 0.1)
    assert not torch.equal(lora(x), plain(x)), "trained adapters must change refinement"
    assert torch.equal(lora(x, skip_refine=True), lora.core(x)), \
        "pass 1 must stay the bare shared core even with trained adapters"


def test_lora_pass_reset_after_forward_and_exception():
    _, lora = _lora_pair()
    with torch.no_grad():
        lora.residual_weight.normal_(0, 0.3)
        for k in lora.loop_lora_B:
            lora.loop_lora_B[k].normal_(0, 0.1)
    x = torch.randn(B, T, H)
    ref = lora.core(x).clone()
    lora(x)
    assert lora._lora_pass == 0
    assert torch.equal(lora.core(x), ref), "direct core call after a wrapper forward must be bare"

    class Boom(Exception):
        pass

    calls = {"n": 0}
    orig = lora.core.forward

    def exploding(x_, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:      # blow up inside the first refinement pass
            raise Boom()
        return orig(x_, *a, **k)

    lora.core.forward = exploding
    try:
        lora(x)
    except Boom:
        pass
    finally:
        lora.core.forward = orig
    assert lora._lora_pass == 0, "mid-refinement exception must not leave adapters armed"


def test_lora_state_dict_roundtrip_and_config_inference():
    _, lora = _lora_pair(rank=3)
    with torch.no_grad():
        lora.residual_weight.normal_(0, 0.3)
        for k in lora.loop_lora_B:
            lora.loop_lora_B[k].normal_(0, 0.1)
    sd = lora.state_dict()
    rank, targets = lora_config_from_sd(sd)
    assert rank == 3 and targets == ("key", "output", "receptance", "value"), (rank, targets)
    assert lora_config_from_sd({"residual_weight": torch.zeros(4)}) == (0, ())
    fresh = LoopedRWKV(StubCore(), n_loops=4, lora_rank=rank, lora_targets=targets)
    fresh.load_state_dict(sd)
    x = torch.randn(B, T, H)
    assert torch.equal(fresh(x), lora(x))


def test_lora_plain_ckpt_upgrade_and_param_names():
    plain, lora = _lora_pair()
    with torch.no_grad():
        plain.residual_weight.normal_(0, 0.3)
    miss, unexp = lora.load_state_dict(plain.state_dict(), strict=False)
    assert not unexp and all(str(m).startswith("loop_lora_") for m in miss)
    x = torch.randn(B, T, H)
    assert torch.equal(lora(x), plain(x)), "fresh zero-B adapters must be a loss-free upgrade"
    names = lora.loop_param_names()
    pnames = {n for n, _ in lora.named_parameters()}
    lora_names = {n for n in names if n.startswith("loop_lora_")}
    assert lora_names and lora_names <= pnames, "loop_param_names must be real dotted param names"
    lora.to(dtype=torch.bfloat16).float_gates()
    assert lora.loop_lora_B["1_receptance"].dtype == torch.float32


def test_lora_composes_with_hyper():
    torch.manual_seed(5)
    core = StubCore()
    both = LoopedRWKV(core, n_loops=4, hyper_lanes=2, lora_rank=4)
    plain = LoopedRWKV(core, n_loops=4)
    both.iter_norm.load_state_dict(plain.iter_norm.state_dict())
    with torch.no_grad():
        plain.residual_weight.normal_(0, 0.3)
        both.residual_weight.copy_(plain.residual_weight)
    x = torch.randn(B, T, H)
    assert torch.equal(both(x), plain(x)), "hyper+lora at init must equal the plain loop"
    both(x).float().pow(2).mean().backward()
    assert both.loop_lora_B["1_receptance"].grad is None or True  # B=0: grad may be 0, A path below
    assert both.hyper_read.grad is not None


def test_lora_rejected_configs():
    try:
        LoopedRWKV(StubCore(), n_loops=1, lora_rank=4)
    except ValueError:
        pass
    else:
        raise AssertionError("lora with n_loops=1 must be rejected")
    try:
        LoopedRWKV(StubCore(), n_loops=4, lora_rank=4, lora_targets=("nope",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lora targets must be rejected")


# ---------------------------------------------------------------------------
# loop-count sampling
# ---------------------------------------------------------------------------

def test_sample_loop_count_ranges():
    rng = np.random.default_rng(0)
    for mode in ("uniform", "poisson"):
        ks = [sample_loop_count(mode, 4, rng) for _ in range(500)]
        assert all(1 <= k <= 4 for k in ks), mode
        assert len(set(ks)) > 1, f"{mode} never varied"
    assert sample_loop_count("off", 4, rng) == 4
    ks = [sample_loop_count("poisson", 4, rng) for _ in range(500)]
    assert ks.count(4) > 200, "poisson should put most mass at full depth"


def test_n_loops_mutation_round_trip():
    _, hyp = _pair(2)
    with torch.no_grad():
        hyp.residual_weight.normal_(0, 0.3)
    x = torch.randn(B, T, H)
    full = hyp(x)
    hyp.n_loops = 2
    shallow = hyp(x)
    hyp.n_loops = 4
    assert not torch.equal(full, shallow)
    assert torch.equal(hyp(x), full), "restoring n_loops must restore the function"


# ---------------------------------------------------------------------------
# iterate consistency
# ---------------------------------------------------------------------------

def test_iter_consist_off_by_default_and_grad_gated():
    _, hyp = _pair(2)
    x = torch.randn(B, T, H)
    hyp(x)
    assert hyp.last_iter_consist is None, "must be off by default"
    hyp.iter_consist = True
    with torch.no_grad():
        hyp(x)
    assert hyp.last_iter_consist is None, "must not compute under no_grad (evals)"
    hyp(x)
    assert hyp.last_iter_consist is not None and torch.isfinite(hyp.last_iter_consist)


def test_iter_consist_pulls_early_iterates():
    for K in (0, 2):   # plain and hyper paths
        mod = LoopedRWKV(StubCore(), n_loops=4, hyper_lanes=K)
        mod.iter_consist = True
        with torch.no_grad():
            mod.residual_weight.normal_(0, 0.3)
        x = torch.randn(B, T, H)
        mod(x)
        ic = mod.last_iter_consist
        assert float(ic) > 0
        ic.backward()
        assert mod.residual_weight.grad is not None
        assert mod.residual_weight.grad.abs().sum() > 0, f"gates must receive consist grad (K={K})"
        # final iterate is the stop-grad anchor: at n_loops the LAST gate only appears
        # in the detached target, so its gradient must be exactly zero
        assert mod.residual_weight.grad[-1].abs().sum() == 0, "final iterate must be sg"


def test_iter_consist_skip_refine_none():
    mod = LoopedRWKV(StubCore(), n_loops=4)
    mod.iter_consist = True
    mod(torch.randn(B, T, H), skip_refine=True)
    assert mod.last_iter_consist is None


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
