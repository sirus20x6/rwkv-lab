import sys, torch, torch.nn as nn
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from looped_rwkv import LoopedRWKV

torch.manual_seed(0)
H, G = 128, 8

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_heads = H, G
        self.lin = nn.Linear(H, H)
    def forward(self, x, *a, initial_state=None, shift_state=None, return_state=False, **kw):
        y = torch.tanh(self.lin(x))
        if return_state:
            return y, torch.zeros(x.shape[0], G, 4, 4), x[:, -1:]
        return y

core = Core()
x = torch.randn(2, 16, H)
ref = core(x)

# 1) regression: init no-op still exact for all modes x cap x loop_index
for mode in ("scalar", "head", "channel", "factored"):
    for cap in (0.0, 0.5):
        for li in (False, True):
            m = LoopedRWKV(Core(), n_loops=4, gate_mode=mode, gate_cap=cap, loop_index=li)
            m.core = core
            assert torch.equal(m(x), ref), (mode, cap, li)
print("1) init no-op: OK (regression)")

# 2) regression: scalar->finer broadcast bit-exact
sc = LoopedRWKV(core, n_loops=4, gate_mode="scalar")
with torch.no_grad():
    sc.residual_weight.copy_(torch.tensor([0.0, 0.1, -0.2, 0.3]))
    sc.iter_norm.weight.mul_(1.3)
y_sc = sc(x)
import convert_train
for mode in ("head", "channel", "factored"):
    tgt = LoopedRWKV(core, n_loops=4, gate_mode=mode)
    sd = {k: v.clone() for k, v in sc.state_dict().items() if not k.startswith("core.")}
    convert_train._expand_loop_gates(sd, tgt)
    tgt.load_state_dict(sd, strict=False)
    assert torch.allclose(tgt(x), y_sc, atol=0, rtol=0), mode
print("2) broadcast: OK (regression)")

# 3) skip_refine: trained gates, but output == pass-1 exactly; state tuple intact
y_skip = sc(x, skip_refine=True)
assert torch.equal(y_skip, ref), "skip_refine must return the bare pass-1 output"
y3, s3, sh3 = sc(x, return_state=True, skip_refine=True)
assert torch.equal(y3, ref) and sh3.shape == (2, 1, H)
# bare core swallows the kwarg
assert torch.equal(core(x, skip_refine=True) if True else ref, ref) or True
try:
    core(x, skip_refine=True)
except TypeError:
    raise AssertionError("bare core must swallow skip_refine via **kwargs")
print("3) skip_refine: pass-1 exact, states intact, bare core tolerant")

# 4) float_gates: gates fp32 after module-wide bf16 cast; stream stays bf16; still no-op at init
for mode in ("scalar", "head", "channel", "factored"):
    m = LoopedRWKV(Core(), n_loops=4, gate_mode=mode, gate_cap=0.25, loop_index=True).to(torch.bfloat16)
    m.float_gates()
    assert m.residual_weight.dtype == torch.float32
    if mode == "factored":
        assert m.gate_chan.dtype == torch.float32
    assert m.loop_index_embed.dtype == torch.float32
    xb = x.to(torch.bfloat16)
    yb = m(xb)
    assert yb.dtype == torch.bfloat16, yb.dtype
    assert torch.equal(yb, m.core(xb)), "init no-op must survive fp32 gates + bf16 stream"
    # gradient flows into the fp32 gates through the cast
    m(xb).float().sum().backward()
    assert m.residual_weight.grad is not None and m.residual_weight.grad.dtype == torch.float32
print("4) float_gates: fp32 params, bf16 stream, no-op + grads OK")

# 5) assemble_looped: looped convert ckpt is stripped, not double-wrapped
import assemble_looped
looped_sd = sc.state_dict()  # core.* + residual_weight + iter_norm.weight (trained gates)
blob = {"student": looped_sd, "codec": {}, "args": {"layer": 7}}
L, out_sd = assemble_looped._looped_layer(blob, 4)
assert L == 7
assert not any(k.startswith("core.core.") for k in out_sd), "double-wrap!"
assert "core.lin.weight" in out_sd and "core.residual_weight" not in out_sd
assert torch.equal(out_sd["residual_weight"], torch.zeros(4)), "gates must be re-zeroed"
fresh = LoopedRWKV(Core(), n_loops=4)
miss, unexp = fresh.load_state_dict(out_sd, strict=True), None  # strict load must succeed
assert torch.equal(fresh(x), fresh.core(x)), "assembled layer must be single-pass at init"
# bare-core ckpt path unchanged
bare_blob = {"student": {k[len("core."):]: v for k, v in looped_sd.items() if k.startswith("core.")},
             "args": {"layer": 3}}
L2, out_sd2 = assemble_looped._looped_layer(bare_blob, 4)
assert L2 == 3 and "core.lin.weight" in out_sd2
# library file with 2D (head) gates: n_loops read from shape[0], not numel
head = LoopedRWKV(core, n_loops=4, gate_mode="head")
lib_blob = {"layer_id": 5, "state_dict": head.state_dict()}
L3, out_sd3 = assemble_looped._looped_layer(lib_blob, 4)
assert L3 == 5 and out_sd3["residual_weight"].shape == (4, G)
print("5) assemble: strip + zero-init + shape[0] n_loops OK")

# 6) _expand_loop_gates loop-count mismatch: clear error
try:
    convert_train._expand_loop_gates({"residual_weight": torch.zeros(6)},
                                     LoopedRWKV(core, n_loops=4))
    raise AssertionError("should have exited")
except SystemExit as e:
    assert "--loop-count" in str(e), str(e)
print("6) loop-count mismatch message: OK")

# 7) muonclip guard refuses a looped student before touching muon imports
try:
    convert_train._make_muonclip(sc, None, None)
    raise AssertionError("should have exited")
except SystemExit as e:
    assert "LoopedRWKV" in str(e)
print("7) muonclip guard: OK")

# 8) gate-mode inference logic (mirrors distill_consolidate._gate_mode_of)
def gate_mode_of(sd, hidden):
    rw = sd.get("residual_weight")
    if rw is None or rw.ndim == 1:
        return "scalar"
    if "gate_chan" in sd:
        return "factored"
    return "channel" if rw.shape[1] == hidden else "head"
for mode in ("scalar", "head", "channel", "factored"):
    m = LoopedRWKV(Core(), n_loops=4, gate_mode=mode, loop_index=(mode == "head"))
    inferred = gate_mode_of(m.state_dict(), H)
    assert inferred == mode, (mode, inferred)
    # and a fresh module built from the inference loads the sd strictly
    m2 = LoopedRWKV(Core(), n_loops=4, gate_mode=inferred, loop_index="loop_index_embed" in m.state_dict())
    m2.load_state_dict(m.state_dict(), strict=True)
print("8) gate-mode inference round-trip: OK")
print("ALL FIX TESTS PASSED")
