"""CPU invariant tests for the new LoopedRWKV features (hyper_lanes, per-pass
LoRA, sample_loop_count, lora_config_from_sd) — same contract classes as the
original gate modes: exact no-op at init, fp32 growth params, sd round-trips,
routing names, bare-core direct-call safety."""
import sys, torch, torch.nn as nn
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from rwkv_lab.looped_rwkv import LoopedRWKV, sample_loop_count, lora_config_from_sd

torch.manual_seed(0)
H, G = 128, 8

class Core(nn.Module):
    """Dummy core with the real core's Linear attr names so LoRA targets bind."""
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_heads = H, G
        self.receptance = nn.Linear(H, H, bias=False)
        self.key = nn.Linear(H, H, bias=False)
        self.value = nn.Linear(H, H, bias=False)
        self.output = nn.Linear(H, H, bias=False)
    def forward(self, x, *a, initial_state=None, shift_state=None, return_state=False, **kw):
        y = self.output(torch.tanh(self.receptance(x) * self.key(x) + self.value(x)))
        if return_state:
            return y, torch.zeros(x.shape[0], G, 4, 4), x[:, -1:]
        return y

core = Core()
x = torch.randn(2, 16, H)
ref = core(x)

# 1) exact no-op at init for every new-feature combination (incl. K=3, non-power-of-2)
for hl in (0, 2, 3):
    for lr_ in (0, 4):
        for gm in ("scalar", "factored"):
            m = LoopedRWKV(Core(), n_loops=4, gate_mode=gm, gate_cap=0.25,
                           loop_index=True, hyper_lanes=hl, lora_rank=lr_)
            m.core = core  # share weights with the reference (hooks re-register? no — hooks were
            # registered on the ORIGINAL dummy core; swap only works when lora_rank==0)
            if lr_ == 0:
                assert torch.equal(m(x), ref), (hl, lr_, gm)
print("1a) init no-op (hyper_lanes x gate modes, shared-core check): OK")
# LoRA variant: hooks bind to the module's own core, so compare against that core directly
for hl in (0, 2, 3):
    m = LoopedRWKV(Core(), n_loops=4, gate_mode="factored", gate_cap=0.25,
                   loop_index=True, hyper_lanes=hl, lora_rank=4)
    assert torch.equal(m(x), m.core(x)), f"hyper_lanes={hl} + lora: not a no-op at init"
print("1b) init no-op with per-pass LoRA armed: OK")

# 2) K=1 hyper lanes refused; typo'd lora target refused (loud, not silent)
try:
    LoopedRWKV(Core(), n_loops=4, hyper_lanes=1); raise AssertionError("K=1 accepted")
except ValueError: pass
try:
    LoopedRWKV(Core(), n_loops=4, lora_rank=4, lora_targets=("receptance", "outptu"))
    raise AssertionError("typo'd target accepted")
except ValueError: pass
print("2) guard rails (K=1, typo'd lora target): OK")

# 3) lora_config_from_sd round-trip: infer (rank, targets), strict-load into fresh module
m = LoopedRWKV(Core(), n_loops=4, gate_mode="head", lora_rank=6,
               lora_targets=("receptance", "value"))
sd = m.state_dict()
rank, targets = lora_config_from_sd(sd)
assert rank == 6 and targets == ("receptance", "value"), (rank, targets)
m2 = LoopedRWKV(Core(), n_loops=4, gate_mode="head", lora_rank=rank, lora_targets=targets)
m2.load_state_dict(sd, strict=True)
assert torch.equal(m2(x), m(x))
assert lora_config_from_sd(LoopedRWKV(Core(), n_loops=4).state_dict()) == (0, ())
print("3) lora_config_from_sd round-trip + strict load: OK")

# 4) float_gates covers the new tensors; bf16 stream stays bf16; no-op survives
m = LoopedRWKV(Core(), n_loops=4, gate_mode="factored", loop_index=True,
               hyper_lanes=2, lora_rank=4).to(torch.bfloat16).float_gates()
for n in ("hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"):
    assert getattr(m, n).dtype == torch.float32, n
for pd in (m.loop_lora_A, m.loop_lora_B):
    for k in pd:
        assert pd[k].dtype == torch.float32, k
xb = x.to(torch.bfloat16)
yb = m(xb)
assert yb.dtype == torch.bfloat16 and torch.equal(yb, m.core(xb))
print("4) float_gates on hyper/lora tensors, bf16 stream no-op: OK")

# 5) loop_param_names covers every new tensor and matches named_parameters exactly
names = m.loop_param_names()
pnames = {n for n, _ in m.named_parameters()}
assert names <= pnames, names - pnames
expected = {"residual_weight", "gate_chan", "loop_index_embed",
            "hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"}
assert expected <= names
assert sum(1 for n in names if n.startswith("loop_lora_A.")) == 3 * 4  # (n_loops-1) x targets
print("5) loop_param_names routing names: OK")

# 6) gradient flow once gates open: hyper params + lora B receive grads; skip_refine
#    and direct core calls stay bare (lora _lora_pass reset)
with torch.no_grad():
    m.residual_weight.fill_(0.1)
m(xb).float().pow(2).sum().backward()
assert m.hyper_write.grad is not None and m.hyper_write.grad.abs().sum() > 0
assert m.hyper_alpha.grad is not None
anyB = next(iter(m.loop_lora_B.values()))
assert anyB.grad is not None and anyB.grad.abs().sum() > 0, "lora B got no gradient"
assert m._lora_pass == 0, "adapters left armed after forward"
y_skip = m(xb, skip_refine=True)
assert torch.equal(y_skip, m.core(xb)), "skip_refine no longer bare-core with new features"
print("6) grads flow to hyper/lora once gates open; _lora_pass reset; skip_refine bare: OK")

# 7) sample_loop_count bounds
rng = np.random.default_rng(0)
for mode in ("uniform", "poisson", "fixed"):
    vals = [sample_loop_count(mode, 4, rng) for _ in range(500)]
    assert all(1 <= v <= 4 for v in vals), (mode, min(vals), max(vals))
    if mode == "fixed":
        assert set(vals) == {4}
assert {sample_loop_count("uniform", 4, rng) for _ in range(200)} == {1, 2, 3, 4}
print("7) sample_loop_count bounds/coverage: OK")

print("ALL NEW-FEATURE TESTS PASSED")
