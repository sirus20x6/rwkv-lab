"""GPU validation for --dmt-cuda-graph (smt_dmt.DMTGraphedRollout).

fp32 STRICT: graphed vs eager dmt_rollout_loss — losses and param grads must
match tightly across TWO iterations with different data (iteration 2 is the
replay-with-new-inputs test that catches stale static buffers), plus curriculum
growth (nb extends lazily) and the ragged-tail eager fallback.
Then a looped-student pass (unused gates under skip_refine) and bf16 benchmarks.
Run ONLY on an idle GPU.
"""
import copy, sys, time, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.manual_seed(0)
from rwkv8_deltanet import RWKV8TimeMixDeltaNet
from looped_rwkv import LoopedRWKV
from smt_dmt import dmt_rollout_loss, DMTGraphedRollout

dev = "cuda"
B, T, C, H, N = 8, 1024, 4096, 64, 64
STRIDE = 64

def make_layer(dt, seed=1):
    m = RWKV8TimeMixDeltaNet(hidden_size=C, num_heads=H, head_size=N,
                             depth_layer_id=16, depth_n_layer=36).to(dev, dt)
    with torch.no_grad():  # non-degenerate weights (paper init zeroes many)
        g = torch.Generator(device=dev).manual_seed(seed)
        for n, p in m.named_parameters():
            base = 1.0 if n == "ln_x.weight" else 0.0
            p.copy_(base + 0.02 * torch.randn(p.shape, device=dev, generator=g, dtype=torch.float32).to(p.dtype))
    return m

def run_loss(layer, h, tgt, bo, graphed=None):
    out = dmt_rollout_loss(layer, None, h, stride=STRIDE, block_out=bo,
                           target_states=tgt, graphed=graphed)
    return out["dmt_memory"].float() + out["dmt_block"].float(), out

def grads_of(m):
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}

def zero(m):
    for p in m.parameters(): p.grad = None

def data(rl, dt, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    nb = rl // STRIDE
    h = torch.randn(B, rl, C, device=dev, generator=g, dtype=torch.float32).to(dt)
    # target_states stay fp32 like the trainer's codec output (bf16 targets would
    # break the EAGER mse backward too — pre-existing, unrelated to graphs)
    tgt = torch.randn(B, nb + 1, H, N, N, device=dev, generator=g, dtype=torch.float32)
    bo = torch.randn(B, rl, C, device=dev, generator=g, dtype=torch.float32).to(dt)
    return h, tgt, bo

# ================= fp32 STRICT =================
e = make_layer(torch.float32)
gm = copy.deepcopy(e)
runner = DMTGraphedRollout(gm)

def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

for it, (rl, seed) in enumerate([(512, 10), (512, 11), (1024, 12)]):  # capture, replay-new-data, curriculum-grow
    h, tgt, bo = data(rl, torch.float32, seed)
    # fla's backward is non-deterministic (atomics): two EAGER runs define the
    # deterministic param set; the graphed run must match eager exactly there.
    zero(e); la, _ = run_loss(e, h, tgt, bo); la.backward(); ga = grads_of(e)
    zero(e); le, _ = run_loss(e, h, tgt, bo); le.backward(); ge = grads_of(e)
    det = {n for n in ge if ge[n] is not None and ge[n].norm() > 0
           and ga[n] is not None and _cos(ga[n], ge[n]) > 0.9999}
    zero(gm)
    lg, _ = run_loss(gm, h, tgt, bo, graphed=runner)
    dl = abs(le.item() - lg.item()) / max(abs(le.item()), 1e-9)
    print(f"[fp32] iter{it} rl={rl}: loss eager={le.item():.6f} graphed={lg.item():.6f} rel={dl:.2e}")
    assert dl < 1e-4, "graphed DMT loss diverges from eager (fp32) — REAL bug"
    lg.backward()
    gg = grads_of(gm)
    worst_cos, worst_n = 1.0, ""
    for n in det:
        assert gg[n] is not None and gg[n].norm() > 0, f"{n}: grad missing under graphs"
        c = _cos(ge[n], gg[n])
        if c < worst_cos: worst_cos, worst_n = c, n
    print(f"       grad cosine over {len(det)} deterministic params, worst ({worst_n}): {worst_cos:.6f}")
    assert len(det) >= 15 and worst_cos > 0.9999, "graphed DMT grads diverge (fp32) — REAL bug"
print(f"[fp32] curriculum: runner holds {len(runner._steps)} chunk graphs (8 -> 16 lazily)")
assert len(runner._steps) == 16

# grad-neutral capture check: capture during a fresh runner with pre-existing grads
e2 = copy.deepcopy(e); zero(e2)
h, tgt, bo = data(256, torch.float32, 20)
l0, _ = run_loss(e2, h, tgt, bo); l0.backward()
pre = grads_of(e2)
r2 = DMTGraphedRollout(e2)
_ = run_loss(e2, h, tgt, bo, graphed=r2)  # triggers capture (no backward); must not touch .grad
post = grads_of(e2)
for n in pre:
    if pre[n] is None: assert post[n] is None, n
    else: assert torch.equal(pre[n], post[n]), f"{n}: capture polluted .grad"
print("[fp32] capture is grad-neutral: OK")

# ragged tail: T=160 -> chunks 64,64,32; tail runs eager fallback
h, tgt, bo = data(192, torch.float32, 21)
h, bo = h[:, :160], bo[:, :160]
r3 = DMTGraphedRollout(copy.deepcopy(e))
le, _ = run_loss(e, h, tgt, bo)
lg, _ = run_loss(r3.layer, h, tgt, bo, graphed=r3)
assert abs(le.item() - lg.item()) / max(abs(le.item()), 1e-9) < 1e-4
print("[fp32] ragged-tail fallback: OK")
del e, gm, e2, runner, r2, r3; torch.cuda.empty_cache()

# ================= looped student (bf16, unused gates) =================
dt = torch.bfloat16
core = make_layer(dt)
loop_e = LoopedRWKV(core, n_loops=4, gate_mode="factored").to(dev, dt).float_gates()
loop_g = copy.deepcopy(loop_e)
rl = 512
h, tgt, bo = data(rl, dt, 30)
rz = DMTGraphedRollout(loop_g)
le, _ = run_loss(loop_e, h, tgt, bo)
lg, _ = run_loss(loop_g, h, tgt, bo, graphed=rz)
rel_l = abs(le.item() - lg.item()) / max(abs(le.item()), 1e-9)
print(f"[bf16] looped: loss eager={le.item():.5f} graphed={lg.item():.5f} rel={rel_l:.2e}")
assert rel_l < 3e-2, "bf16 graphed loss diverges egregiously"
le.backward(); lg.backward()
assert loop_g.residual_weight.grad is None or loop_g.residual_weight.grad.norm() == 0
print("[bf16] looped student: unused gates stay gradient-free under graphs: OK")

# ================= benchmarks (bf16, idle GPU) =================
mem0 = torch.cuda.memory_allocated() / 2**20
def bench(fn, n=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

be = make_layer(dt); bg = copy.deepcopy(be)
rb = DMTGraphedRollout(bg)
h, tgt, bo = data(1024, dt, 40)
def eager_it():
    zero(be); l, _ = run_loss(be, h, tgt, bo); l.backward()
def graph_it():
    zero(bg); l, _ = run_loss(bg, h, tgt, bo, graphed=rb); l.backward()
graph_it()  # capture outside timing
mem1 = torch.cuda.memory_allocated() / 2**20
t_e, t_g = bench(eager_it), bench(graph_it)
print(f"\n--- idle-GPU benchmark (bf16, B={B}, rl=1024, 16 chunks, fwd+bwd) ---")
print(f"DMT rollout eager   : {t_e:7.2f} ms")
print(f"DMT rollout graphed : {t_g:7.2f} ms  ({t_e/t_g:.2f}x)")
print(f"static graph buffers: ~{mem1-mem0:.0f} MiB for 16 chunk graphs")
print("\nALL DMT-CUDA-GRAPH CHECKS PASSED")
