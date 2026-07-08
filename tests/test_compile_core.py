"""GPU validation for --compile-core.

Two-tier correctness:
  * fp32 gate (STRICT): eager vs compiled with identical fp32 weights. Any real
    transformation bug shows here; tolerances are tight.
  * bf16 report (INFORMATIONAL): production dtype. Fusion reorders bf16 rounding,
    and the wkv recurrence amplifies decay-side ulps over T steps, so max-rel can
    legitimately reach ~1e-1; only egregious divergence fails.
Then benchmarks in bf16 (production config) on an idle GPU.
"""
import copy, sys, time, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.manual_seed(0)
from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet
from rwkv_lab.looped_rwkv import LoopedRWKV

dev = "cuda"
B, T, C, H, N = 8, 1024, 4096, 64, 64

def make_pair(dt):
    e = RWKV8TimeMixDeltaNet(hidden_size=C, num_heads=H, head_size=N,
                             depth_layer_id=16, depth_n_layer=36).to(dev, dt)
    with torch.no_grad():  # non-degenerate weights everywhere (paper init zeroes many)
        g = torch.Generator(device=dev).manual_seed(1)
        for n, p in e.named_parameters():
            base = 1.0 if n == "ln_x.weight" else 0.0
            p.copy_(base + 0.02 * torch.randn(p.shape, device=dev, generator=g, dtype=torch.float32).to(p.dtype))
    c = copy.deepcopy(e)
    c.compile()
    return e, c

def maxrel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)

def rmsrel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-9)).item()

# ============ fp32 STRICT gate ============
e32, c32 = make_pair(torch.float32)
x32 = torch.randn(B, T, C, device=dev)
d = maxrel(c32(x32), e32(x32))
print(f"[fp32] fwd T={T}: max rel = {d:.2e}")
assert d < 1e-3, "fp32 compiled forward diverges — REAL transformation bug"

s0 = torch.randn(B, H, N, N, device=dev); sh = torch.randn(B, 1, C, device=dev)
ye, se, she = e32(x32[:, :64], initial_state=s0, shift_state=sh, return_state=True)
yc, sc, shc = c32(x32[:, :64], initial_state=s0, shift_state=sh, return_state=True)
print(f"[fp32] fwd T=64 +states: y {maxrel(yc, ye):.2e}  state {maxrel(sc, se):.2e}")
assert maxrel(yc, ye) < 1e-3 and maxrel(sc, se) < 1e-3 and torch.equal(shc, she)

# fla's chunk_rwkv7 backward is non-deterministic (atomic accumulation): params
# whose gradient survives only as cancellation residue (e.g. x_a, |g|~1e-9) don't
# reproduce even eager-vs-eager. Self-calibrate: two eager backward runs define
# the DETERMINISTIC param set; compiled must match eager exactly there.
def param_grads(m, inp):
    for p in m.parameters(): p.grad = None
    inp.grad = None
    m(inp).pow(2).mean().backward()
    return {n: p.grad.clone() for n, p in m.named_parameters()
            if p.grad is not None and p.grad.norm() > 0}
xe = x32.clone().requires_grad_(True); xc = x32.clone().requires_grad_(True)
ge_a, ge_b = param_grads(e32, xe), param_grads(e32, xe)
gc = param_grads(c32, xc)
def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
det = {n for n in ge_a if cos(ge_a[n], ge_b[n]) > 0.9999}
skipped = sorted(set(ge_a) - det)
worst_cos, worst_name = 1.0, ""
for n in det:
    c = cos(ge_b[n], gc[n])
    if c < worst_cos: worst_cos, worst_name = c, n
print(f"[fp32] grad cosine over {len(det)} deterministic params (worst: {worst_name}): "
      f"{worst_cos:.6f}; skipped kernel-nondeterministic: {skipped}")
assert len(det) >= 15 and worst_cos > 0.9999, "fp32 gradient mismatch — REAL bug"
d_in = maxrel(xc.grad, xe.grad)
print(f"[fp32] input grad: max rel = {d_in:.2e}")
assert d_in < 1e-3
del e32, c32, x32, xe, xc; torch.cuda.empty_cache()

# ============ bf16 informational report ============
dt = torch.bfloat16
e16, c16 = make_pair(dt)
x16 = torch.randn(B, T, C, device=dev, dtype=dt)
print(f"[bf16] fwd T={T}: max rel = {maxrel(c16(x16), e16(x16)):.2e}  rms rel = {rmsrel(c16(x16), e16(x16)):.2e}")
assert rmsrel(c16(x16), e16(x16)) < 5e-2, "bf16 rms divergence egregious"
le = LoopedRWKV(e16, n_loops=4, gate_mode="factored").to(dev, dt).float_gates()
lc = LoopedRWKV(c16, n_loops=4, gate_mode="factored").to(dev, dt).float_gates()
with torch.no_grad():
    for m in (le, lc): m.residual_weight.fill_(0.1)
print(f"[bf16] LoopedRWKV(4) fwd: rms rel = {rmsrel(lc(x16), le(x16)):.2e}")
assert set(c16.state_dict()) == set(e16.state_dict()), "compile changed state_dict keys!"
print("state_dict keys unchanged under compile: OK")

# ============ benchmarks (bf16, idle GPU) ============
def bench(fn, n=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

from fla.ops.rwkv7 import chunk_rwkv7
rr = torch.randn(B, T, H, N, device=dev, dtype=dt)
gk = -torch.rand_like(rr) * 0.1 - 0.01
kk, vv = torch.randn_like(rr), torch.randn_like(rr)
aa = -torch.nn.functional.normalize(torch.randn_like(rr), dim=-1)
bb = -aa * 0.5
print(f"\n--- idle-GPU benchmarks (bf16, B={B}, T={T}) ---")
print(f"chunk_rwkv7 kernel        : {bench(lambda: chunk_rwkv7(rr, gk, kk, vv, aa, bb, scale=1.0)):7.2f} ms")
t_e = bench(lambda: e16(x16)); t_c = bench(lambda: c16(x16))
print(f"layer fwd  eager/compiled : {t_e:7.2f} / {t_c:7.2f} ms  ({t_e/t_c:.2f}x)")

xg_e = x16.clone().requires_grad_(True); xg_c = x16.clone().requires_grad_(True)
def fb(m, inp):
    m(inp).float().pow(2).mean().backward()
    inp.grad = None
    for p in m.parameters(): p.grad = None
t_e = bench(lambda: fb(e16, xg_e), n=10); t_c = bench(lambda: fb(c16, xg_c), n=10)
print(f"layer f+b  eager/compiled : {t_e:7.2f} / {t_c:7.2f} ms  ({t_e/t_c:.2f}x)")
t_e = bench(lambda: fb(le, xg_e), n=10); t_c = bench(lambda: fb(lc, xg_c), n=10)
print(f"loop4 f+b  eager/compiled : {t_e:7.2f} / {t_c:7.2f} ms  ({t_e/t_c:.2f}x)")

def dmt_shape(m):  # 16 sequential 64-token stateful calls (the DMT rollout pattern)
    st, shp = None, None
    for j in range(16):
        _, st, shp = m(x16[:, j*64:(j+1)*64], initial_state=st, shift_state=shp, return_state=True)
t_e = bench(lambda: dmt_shape(e16), n=10); t_c = bench(lambda: dmt_shape(c16), n=10)
print(f"DMT 16x64  eager/compiled : {t_e:7.2f} / {t_c:7.2f} ms  ({t_e/t_c:.2f}x)")
print("\nALL COMPILE-CORE CHECKS PASSED")
