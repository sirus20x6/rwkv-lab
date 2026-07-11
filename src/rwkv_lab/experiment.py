"""Tier-1 experiment layer for RWKV-Lab — makes lever A/Bs CONCLUSIVE.

The from-scratch sweeps were inconclusive because (a) single-seed runs have a ~0.1-nat noise
floor, (b) web-text ppl doesn't measure what the levers are for, and (c) bad configs burned full
runs. This wraps the small-model harness with the three fixes:

  1. Paired deterministic seed/data tapes with bootstrap confidence intervals, sign-flip tests,
     effect sizes, Holm correction, and next-seed power guidance.
  2. Preflight + successive-halving budget rungs that reject or eliminate weak arms early.
  3. Capability matrices (length/noise/NLL/calibration), measured time/memory/energy, factorial
     interaction arms, Pareto scoring, and fresh-seed confirmatory campaigns.
  4. A normalized trial registry with curves, lineage, RNG hashes, and a reproducibility capsule.

Levers are configured by name (baseline, loop2/3/4, hyper, cart, deq, ...); extend LEVERS below.

    python -m rwkv_lab.experiment --task recall:16 --configs baseline,loop3,loop3_hyper \
        --seeds 4 --steps 3000 --d-model 256 --n-layers 4
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, statistics, threading, time
import numpy as np
import torch
import torch.nn.functional as F

from rwkv_lab.rwkv_pretrain import (RWKV7Small, build_optimizer, add_muon_args, muon_opts_from,
                                    apply_fp8, enable_fast_matmul)
from rwkv_lab.synthetic_tasks import make_task, Task
from rwkv_lab.looped_rwkv import LoopedRWKV
from rwkv_lab import registry
from rwkv_lab.experiment_analysis import paired_stats, holm_adjust, pareto_front, sequential_holm

# Lever configs. A lever mixes LoopedRWKV kwargs (recurrent depth) with LookaheadSystem aux weights
# (latent-prediction training objectives, keys ending in _weight). {} = bare core baseline.
# NOTE: only nextlat is valid on the synthetic tasks — it predicts h[t+d] from within the sequence
# (no future tokens). top/lmtp/bst/jtp need a real token FUTURE, so they live in the LM path only.
LEVERS = {
    "baseline":     {},
    "loop2":        dict(n_loops=2),
    "loop3":        dict(n_loops=3),
    "loop4":        dict(n_loops=4),
    "loop3_hyper":  dict(n_loops=3, hyper_lanes=2),
    "loop3_cart":   dict(n_loops=3, cart_anchor=True),
    "loop3_deq":    dict(n_loops=3, loop_deq=True),
    "loop3_factor": dict(n_loops=3, gate_mode="factored"),
    "nextlat":       dict(nextlat_weight=0.1),               # next-latent prediction aux (light)
    "loop3_nextlat": dict(n_loops=3, nextlat_weight=0.1),    # recurrent depth + next-latent
    "seedchain":     dict(seed_chain=True),                  # Future-Seed: s_0^L = s_T^{L-1} (no loops)
    "engram":        dict(engram=True),                      # Engram LMB: token-SAM recall + learned table
    "deepembed":     dict(deepembed=True),                   # DeepEmbed v1: gate the FFN output
    "de_hidden":     dict(deepembed=True, de_mode="hidden"), # BlinkDL-exact: bilinear gate on FFN hidden
    "de_shift":      dict(deepembed=True, de_mode="hidden", de_shift=True),   # + separate DE token-shift
    "de_full":       dict(deepembed=True, de_mode="hidden", de_shift=True, de_emb_res=True),
    # LM-only latent objectives (need a real token future -> run via the LM path, not synthetic tasks)
    "top":           dict(top_weight=0.1),                   # token-order prediction (lookahead window)
    "lmtp":          dict(lmtp_weight=0.1),                  # leap multi-token prediction
    "bst":           dict(bst_weight=0.1),                   # belief-state (fwd+bwd) objective
    "jtp":           dict(jtp_weight=0.1),                   # joint multi-token prediction
    # Scale/adaptation comparison arms for the LM trainer and trainboard. Sources:
    # u-muP arXiv:2407.17465; Titans arXiv:2501.00663; MIRAS arXiv:2504.13173;
    # ATLAS arXiv:2505.23735; Nested Learning arXiv:2512.24695; NVFP4
    # arXiv:2509.25149 and TetraJet-v2 arXiv:2510.27527. Full URLs and adopted
    # mechanisms live in u_mup.py, online_memory.py, nvfp4.py, and README.md.
    "umup_256":      dict(u_mup_base_width=256, u_mup_base_depth=4),
    "mem_titans":    dict(online_memory=True, online_memory_mode="titans"),
    "mem_miras":     dict(online_memory=True, online_memory_mode="miras"),
    "mem_atlas":     dict(online_memory=True, online_memory_mode="atlas"),
    "mem_nested":    dict(online_memory=True, online_memory_mode="nested"),
    # QRWKV7/Recursal balanced state update for GroupNorm stability at scale:
    # https://huggingface.co/recursal/QRWKV7-7B-Instruct/blob/main/modeling_rwkv7qwen2.py
    "balance_state": dict(balance_state=True),
    "state_offset":  dict(state_offset=True, state_offset_interval=1),
    # Liu et al. (2026), https://arxiv.org/abs/2604.00801
    "routing_free_moe": dict(routing_free_moe=True, routing_free_experts=4,
                              routing_free_rank=32, routing_free_threshold=0.2,
                              routing_free_balance=0.5),
    "nvfp4":         dict(nvfp4=True),
    "nvfp4_rht":     dict(nvfp4=True, nvfp4_rht=True),
    "nvfp4_native":  dict(nvfp4=True, nvfp4_rht=True,
                            nvfp4_backend="transformer_engine"),
}

# Levers whose objective needs a real token FUTURE — only valid on the LM path (rwkv_pretrain),
# not the synthetic diagnostic tasks. The board disables these unless an LM corpus is selected.
LM_ONLY = ("top", "lmtp", "bst", "jtp", "umup_256", "mem_titans", "mem_miras",
           "mem_atlas", "mem_nested", "state_offset", "routing_free_moe",
           "nvfp4", "nvfp4_rht", "nvfp4_native")

_AUX_KEYS = ("nextlat_weight", "top_weight", "lmtp_weight", "bst_weight", "jtp_weight")


def _split_lever(kw: dict):
    """Partition a lever into (loop kwargs, aux latent-prediction weights)."""
    aux = {k: v for k, v in kw.items() if k in _AUX_KEYS}
    loop = {k: v for k, v in kw.items() if k not in _AUX_KEYS}
    return loop, aux


def _norm_loopkw(kw: dict) -> dict:
    """Fill LoopedRWKV defaults so RWKV7Small's loop path constructs cleanly."""
    if not kw:
        return {}
    d = dict(n_loops=2, hyper_lanes=0, gate_mode="scalar", gate_cap=0.0, cart_anchor=False,
             loop_deq=False, deq_window=1, fixed_point_halt=False, adaptive_halt=False)
    d.update(kw)
    return d


def build(task: Task, d_model, n_layers, head_size, lever, device="cpu") -> RWKV7Small:
    loop, _ = _split_lever(LEVERS[lever])
    seed_chain = bool(loop.pop("seed_chain", False))         # model kwargs, not LoopedRWKV kwargs
    engram = bool(loop.pop("engram", False))
    deepembed = bool(loop.pop("deepembed", False))
    de_dim = int(loop.pop("de_dim", 0))
    de_mode = str(loop.pop("de_mode", "out"))
    de_shift = bool(loop.pop("de_shift", False))
    de_emb_res = bool(loop.pop("de_emb_res", False))
    routing_free = bool(loop.pop("routing_free_moe", False))
    routing_free_kw = None
    if routing_free:
        routing_free_kw = {
            "n_experts": int(loop.pop("routing_free_experts", 4)),
            "rank": int(loop.pop("routing_free_rank", 32)),
            "threshold": float(loop.pop("routing_free_threshold", 0.2)),
            "balance_interpolation": float(loop.pop("routing_free_balance", 0.5)),
        }
    m = RWKV7Small(task.vocab, d_model, n_layers, head_size, _norm_loopkw(loop),
                   seed_chain=seed_chain, deepembed=deepembed, de_dim=de_dim, de_mode=de_mode,
                   de_shift=de_shift, de_emb_res=de_emb_res,
                   routing_free_kw=routing_free_kw).to(device, torch.bfloat16)
    if engram:                                               # attach AFTER .to (fp32 growth params)
        from rwkv_lab.rwkv_pretrain import enable_engram
        enable_engram(m, task.vocab, d_model, head_size, n_layers,
                      loop_count=loop.get("n_loops", 1) if loop else 1)
    return m


def _masked_ce(logits, y, m):
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (ce * m.reshape(-1)).sum() / m.sum().clamp_min(1)


def model_stats(model, loop_count: int = 1) -> dict:
    """Params + a rough forward-FLOP/token estimate, so A/Bs can be compute-normalised (a lever
    that helps but costs 2x FLOPs isn't free). FLOP/token ~ 2 * non-embedding params (matmul MACs);
    looped time-mix re-runs, so its share is multiplied by loop_count."""
    tot = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "emb." in n or "head." in n)
    att = sum(p.numel() for n, p in model.named_parameters() if ".att." in n)
    non_emb = tot - emb
    flop_per_tok = 2 * (non_emb + att * (max(loop_count, 1) - 1))   # loops re-run the time-mix
    return {"params_m": tot / 1e6, "flop_per_tok": flop_per_tok}


def loop_gate_stats(model) -> float:
    """Mean |loop gate| (residual_weight) across LoopedRWKV blocks. ~0 => the loops never engaged
    (stayed at zero-init identity) — the direct test of whether recurrent depth did anything."""
    gs = [m.residual_weight.detach().float().abs().mean().item()
          for m in model.modules() if isinstance(m, LoopedRWKV) and hasattr(m, "residual_weight")]
    return sum(gs) / len(gs) if gs else 0.0


def _seeded_batch(task, B, device, seed):
    """Generate a batch from a seed without perturbing model/dropout RNG state."""
    devs = [] if "cuda" not in str(device) else [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(int(seed))
        if devs:
            torch.cuda.manual_seed_all(int(seed))
        return task.batch(B, device, np.random.default_rng(int(seed)))


@torch.no_grad()
def _eval_metrics(model, task, B, device, *, eval_seed, iters=8, noise=0.0):
    model.eval(); correct = n = 0.0; nll = 0.0; confs, oks = [], []
    for i in range(iters):
        x, y, m = _seeded_batch(task, B, device, eval_seed + i)
        if noise > 0:
            devs = [] if "cuda" not in str(device) else [torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(eval_seed + 100_000 + i)
                corrupt = torch.rand(x.shape, device=x.device) < noise
                replacement = torch.randint(2, task.vocab, x.shape, device=x.device)
                x = torch.where(corrupt, replacement, x)
        logits = model(x).float()
        flat_mask = m.reshape(-1) > 0
        flat_logits, flat_y = logits.reshape(-1, logits.shape[-1])[flat_mask], y.reshape(-1)[flat_mask]
        prob = flat_logits.softmax(-1); confidence, pred = prob.max(-1)
        ok = pred == flat_y
        correct += ok.sum().item(); n += ok.numel()
        nll += F.cross_entropy(flat_logits, flat_y, reduction="sum").item()
        confs.append(confidence.cpu()); oks.append(ok.float().cpu())
    conf, ok = torch.cat(confs), torch.cat(oks)
    ece = 0.0
    for lo in torch.linspace(0, 0.9, 10):
        mask = (conf >= lo) & (conf < lo + 0.1)
        if mask.any():
            ece += float(mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs())
    model.train()
    return {"acc": correct / max(n, 1), "nll": nll / max(n, 1), "ece": ece,
            "finite": float(torch.isfinite(conf).all())}


def _eval_acc(model, task, B, device, rng=None, iters=8, eval_seed=12345):
    return _eval_metrics(model, task, B, device, eval_seed=eval_seed, iters=iters)["acc"]


@torch.no_grad()
def _copy_rollout_acc(model, task, B, device, seed):
    """Free-running copy accuracy: generated tokens feed the next prediction."""
    if not hasattr(task, "L") or not hasattr(task, "ns") or not task.name.startswith("copy"):
        return None
    devs = [] if "cuda" not in str(device) else [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        target = torch.randint(2, 2 + task.ns, (B, task.L), device=device)
    sep = torch.full((B,1), 1, dtype=torch.long, device=device)
    prefix_len = task.L + 1
    tokens = torch.empty(B, prefix_len + task.L, dtype=torch.long, device=device)
    tokens[:, :task.L] = target
    tokens[:, task.L:prefix_len] = sep
    model.eval()
    for offset in range(task.L):
        end = prefix_len + offset
        tokens[:, end] = model(tokens[:, :end]).float()[:, -1].argmax(-1)
    model.train()
    return float((tokens[:, prefix_len:] == target).float().mean())


def preflight(task, d_model, n_layers, head_size, lever, device, batch, steps=20):
    """Reject diverging / NaN / non-learning configs before a full run. Returns (ok, reason)."""
    torch.manual_seed(0)
    try:
        model = build(task, d_model, n_layers, head_size, lever, device)
    except Exception as e:
        return False, f"build failed: {type(e).__name__}: {e}"
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, fused=("cuda" in str(device)))
    losses = []
    for i in range(steps):
        x, y, m = _seeded_batch(task, batch, device, 90_000 + i)
        loss = _masked_ce(model(x).float(), y, m)
        if not torch.isfinite(loss):
            return False, f"non-finite loss at step {i}"
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            return False, f"non-finite grad at step {i}"
        opt.step(); losses.append(float(loss.detach()))
    if losses[-1] > losses[0] + 0.5:
        return False, f"diverging (loss {losses[0]:.2f} -> {losses[-1]:.2f})"
    return True, f"ok (loss {losses[0]:.2f} -> {losses[-1]:.2f})"


def _block_source(task, B, device, block, data_seed=0, start_step=0):
    """Amortize per-batch kernel launches for synthetic tasks: generate `block` batches' worth of
    examples in ONE task.batch() call, then serve them B at a time. All rows are iid, so this is
    distributionally identical to `block` separate calls but pays ~1/block the launch overhead.
    block<=1 disables (one generation per step)."""
    block_i = start_step // max(block, 1)
    first_offset = start_step % max(block, 1)
    while True:
        if block <= 1:
            yield _seeded_batch(task, B, device, data_seed + block_i); block_i += 1; continue
        X, Y, M = _seeded_batch(task, block * B, device, data_seed + block_i)
        block_i += 1
        for k in range(first_offset, block):
            sl = slice(k * B, (k + 1) * B)
            yield X[sl], Y[sl], M[sl]
        first_offset = 0


class _PowerSampler:
    def __init__(self, enabled=True):
        self.samples, self.stop_event, self.thread = [], threading.Event(), None
        self.nvml = self.handle = None
        if enabled and "cuda" in str(torch.device("cuda" if torch.cuda.is_available() else "cpu")):
            try:
                import pynvml
                pynvml.nvmlInit(); self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except Exception:
                self.nvml = None
    def start(self):
        if self.nvml is None: return
        def sample():
            while not self.stop_event.wait(0.2):
                try: self.samples.append((time.perf_counter(), self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
                except Exception: return
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
    def finish(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1)
        return sum((t1-t0)*(p0+p1)/2 for (t0,p0),(t1,p1) in zip(self.samples, self.samples[1:]))


def _profile_summary(event_rows, wall, tokens, first_step_s, device):
    out = {"train_seconds": wall, "tokens": tokens, "tok_per_sec": tokens / max(wall, 1e-9),
           "first_step_seconds": first_step_s}
    if event_rows:
        torch.cuda.synchronize()
        names = ("input_ms", "forward_ms", "backward_ms", "optimizer_ms", "step_ms")
        vals = {n: [] for n in names}
        for ev in event_rows:
            parts = [ev[i].elapsed_time(ev[i+1]) for i in range(4)]
            for n, v in zip(names[:-1], parts): vals[n].append(v)
            vals["step_ms"].append(sum(parts))
        for n, xs in vals.items():
            out[n + "_p50"] = float(np.percentile(xs, 50)); out[n + "_p95"] = float(np.percentile(xs, 95))
        out["compile_overhead_seconds"] = max(0.0, first_step_s - out["step_ms_p50"] / 1000)
        out["peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        out["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    return out


def train_eval(task, d_model, n_layers, head_size, lever, seed, device, steps, batch, lr, minutes=0.0,
               optimizer="adamw", weight_decay=0.01, warmup=0, muon_opts=None, fp8=False, do_compile=False,
               gen_block=1, eval_factors=(0.5, 1, 2, 4, 8), eval_noise=(0.05, 0.10),
               data_seed=None, profile=True, resume_path=None, checkpoint_path=None,
               schedule_steps=None, schedule_minutes=None):
    """Train one model on the task; return metrics incl. length-generalization accuracy. Budget is
    either a fixed step count (minutes=0) or wall-clock minutes (Karpathy-style fixed-time rounds)."""
    torch.manual_seed(seed)
    initial_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    model = build(task, d_model, n_layers, head_size, lever, device)
    if fp8:
        apply_fp8(model)
    _, aux = _split_lever(LEVERS[lever])
    heads = None
    if aux:                                                  # latent-prediction aux (e.g. nextlat)
        from rwkv_lab.lookahead_module import LookaheadSystem
        heads = LookaheadSystem(d_model, task.vocab, **aux).to(device, torch.bfloat16)
    named = list(model.named_parameters()) + (list(heads.named_parameters()) if heads else [])
    params = [p for _, p in named]
    opt = build_optimizer(named, optimizer, lr, weight_decay, muon_opts=muon_opts)
    schedule_steps = int(schedule_steps or steps)
    schedule_minutes = float(schedule_minutes or minutes)
    warm = warmup if warmup > 0 else max(1, (schedule_steps or 2000) // 20)
    step, elapsed_before, prior_series, last_loss = 0, 0.0, [], None
    resumed_from = None
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if heads is not None and ck.get("heads") is not None:
            heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0)); elapsed_before = float(ck.get("elapsed_seconds", 0.0))
        prior_series = list(ck.get("series", [])); last_loss = ck.get("last_loss")
        if ck.get("cpu_rng") is not None: torch.set_rng_state(ck["cpu_rng"].cpu())
        if "cuda" in str(device) and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=device)
        resumed_from = os.path.abspath(resume_path)
    t0 = time.perf_counter()
    start_step = step
    fwd = torch.compile(model) if do_compile else model   # compile the train forward; eval stays eager
    src = _block_source(task, batch, device, gen_block,
                        data_seed=seed if data_seed is None else data_seed, start_step=step)
    series, event_rows, first_step_s = prior_series, [], 0.0
    if "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)
    power = _PowerSampler(profile); power.start()
    sample_every = max(1, (steps or 1000) // 100)
    while ((elapsed_before + time.perf_counter() - t0 < minutes * 60)
           if minutes > 0 else (step < steps)):
        elapsed_total = elapsed_before + time.perf_counter() - t0
        frac = (min(1.0, elapsed_total / (schedule_minutes * 60)) if minutes > 0
                else step / max(schedule_steps, 1))
        w = min(1.0, (step + 1) / warm)                      # warmup
        cos = 0.5 * (1 + math.cos(math.pi * frac))           # 1 -> 0 over the budget
        for g in opt.param_groups:
            g["lr"] = lr * w * (0.1 + 0.9 * cos)             # warmup then cosine decay to 0.1x
        step += 1
        measured = profile and "cuda" in str(device) and step % sample_every == 0
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if measured else None
        if ev: ev[0].record()
        wall_step = time.perf_counter()
        x, y, m = next(src)
        if ev: ev[1].record()
        out = fwd(x, return_hidden=bool(heads))
        loss = _masked_ce((out[0] if heads else out).float(), y, m)
        if heads:                                            # within-sequence next-latent aux loss
            loss = loss + heads.compute(out[1], x, model.emb, model.head)["aux_total"]
        if ev: ev[2].record()
        opt.zero_grad(set_to_none=True); loss.backward()
        if ev: ev[3].record()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if ev: ev[4].record(); event_rows.append(ev)
        if step == start_step + 1:
            if "cuda" in str(device): torch.cuda.synchronize()
            first_step_s = time.perf_counter() - wall_step
        if step == 1 or step % sample_every == 0:
            series.append({"step": step, "loss": float(loss.detach())})
    if "cuda" in str(device): torch.cuda.synchronize()
    wall_increment = time.perf_counter() - t0
    wall = elapsed_before + wall_increment
    joules = power.finish()
    base_eval = _eval_metrics(model, task, batch, device, eval_seed=1_000_000 + seed * 10_000)
    final_loss = float(loss.detach()) if step > start_step else float(last_loss or base_eval["nll"])
    out = {"loss": final_loss, **base_eval, "gate": loop_gate_stats(model)}
    rollout = _copy_rollout_acc(model, task, batch, device, 1_500_000 + seed)
    if rollout is not None: out["rollout_acc"] = rollout
    out.update(model_stats(model, LEVERS.get(lever, {}).get("n_loops", 1)))
    # Capability matrix: multiple lengths plus corruption robustness.
    arg = getattr(task, "L", None) or getattr(task, "n", None)
    if arg:
        stem = task.name.rstrip("0123456789")
        for factor in eval_factors:
            length = max(1, int(round(arg * factor)))
            try:
                met = _eval_metrics(model, make_task(f"{stem}:{length}"), batch, device,
                                    eval_seed=2_000_000 + seed * 10_000 + length)
                key = str(factor).replace(".", "_")
                out[f"acc_len_{key}x"], out[f"nll_len_{key}x"] = met["acc"], met["nll"]
                if factor == 2: out["acc_2x"] = met["acc"]
            except Exception:
                pass
    for noise in eval_noise:
        met = _eval_metrics(model, task, batch, device, eval_seed=3_000_000 + seed * 10_000 + int(noise*1000), noise=noise)
        out[f"acc_noise_{int(noise*100):02d}"] = met["acc"]
    if hasattr(task, "distractors"):
        for mult in (1, 2, 4):
            stressed = type(task)(task.n, task.nk, task.nv, distractors=task.n * mult)
            met = _eval_metrics(model, stressed, batch, device,
                                eval_seed=4_000_000 + seed * 10_000 + mult)
            out[f"acc_recall_distractors_{mult}x"] = met["acc"]
    tokens_per_step = int(getattr(task, "L", 0) or getattr(task, "n", 0) or 1)
    if step > start_step:
        tokens_per_step = y.shape[1]
    prof = _profile_summary(event_rows, wall, step * batch * tokens_per_step, first_step_s, device)
    prof["energy_joules"] = joules; prof["joules_per_mtoken"] = joules * 1e6 / max(prof["tokens"], 1)
    out.update(prof)
    out["_series"], out["_profile"] = series, prof
    final_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    cuda_rng_hash = None
    if "cuda" in str(device):
        cuda_rng_hash = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    out["_rng"] = {"model_seed": seed, "data_seed": seed if data_seed is None else data_seed,
                   "eval_seed_base": 1_000_000 + seed * 10_000,
                   "initial_cpu_rng_sha256": initial_rng_hash, "final_cpu_rng_sha256": final_rng_hash,
                   "final_cuda_rng_sha256": cuda_rng_hash, "resumed_from": resumed_from,
                   "start_step": start_step, "final_step": step}
    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        blob = {"version": 1, "model": model.state_dict(),
                "heads": heads.state_dict() if heads is not None else None,
                "optimizer": opt.state_dict(), "step": step, "elapsed_seconds": wall,
                "series": series, "last_loss": final_loss, "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device).cpu() if "cuda" in str(device) else None,
                "lever": lever, "seed": seed,
                "data_seed": seed if data_seed is None else data_seed}
        tmp = checkpoint_path + ".tmp"
        torch.save(blob, tmp); os.replace(tmp, checkpoint_path)
        out["_checkpoint"] = os.path.abspath(checkpoint_path)
    return out


def _agg(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def factorial_configs(names, max_order=2):
    """Generate compatible interaction arms from named lever factors."""
    out = {n: dict(LEVERS[n]) for n in names}
    for order in range(2, max_order + 1):
        for combo in itertools.combinations(names, order):
            merged, ok = {}, True
            for name in combo:
                for k, v in LEVERS[name].items():
                    if k in merged and merged[k] != v: ok = False; break
                    merged[k] = v
                if not ok: break
            if ok: out["+".join(combo)] = merged
    return out


def _aggregate_runs(runs):
    keys = sorted(set.intersection(*(set(r) for r in runs)))
    return {k: _agg([float(r[k]) for r in runs]) for k in keys
            if not k.startswith("_") and all(isinstance(r[k], (int, float, bool)) for r in runs)}


def _paired_comparisons(runs_by_arm, baseline="baseline", metric="acc", seed=0, *,
                        look=None, total_looks=None, alpha=0.05,
                        spending="obrien_fleming"):
    if baseline not in runs_by_arm:
        return {}
    base = runs_by_arm[baseline]
    out = {}
    for name, runs in runs_by_arm.items():
        if name == baseline:
            continue
        n = min(len(base), len(runs))
        if n:
            look_alpha = alpha
            if look is not None and total_looks is not None:
                from rwkv_lab.experiment_analysis import alpha_spending
                look_alpha = alpha_spending(look, total_looks, alpha=alpha,
                                            method=spending)["increment"]
            out[name] = paired_stats([r[metric] for r in base[:n]], [r[metric] for r in runs[:n]],
                                     seed=seed, alpha=look_alpha)
    if look is not None and total_looks is not None:
        return sequential_holm(out, look, total_looks, alpha=alpha, method=spending)
    return holm_adjust(out, alpha=alpha)


def main():
    enable_fast_matmul()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="recall:16")
    ap.add_argument("--configs", default="baseline,loop3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=0.0, help="wall-clock budget per run (0 = use --steps)")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adamw8bit", "paged-adamw8bit", "muon"])
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=0, help="warmup steps (0 = auto, about 5 percent)")
    ap.add_argument("--fp8", action="store_true",
                    help="run eligible Linear GEMMs in fp8 (torchao Float8Linear; Blackwell/Hopper)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (fuses fp8 cast+GEMM; ~2x on Blackwell)")
    ap.add_argument("--gen-block", type=int, default=1,
                    help="generate N batches of synthetic data per launch (amortizes gen kernel launches)")
    add_muon_args(ap)
    ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--head-size", type=int, default=64); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--db", default=None, help="experiment registry path")
    ap.add_argument("--campaign-name", default="")
    ap.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--halving-rungs", default="0.02,0.1,0.3,1.0")
    ap.add_argument("--halving-eta", type=int, default=2)
    ap.add_argument("--checkpoint-dir", default="runs/experiment_checkpoints",
                    help="persistent rung checkpoints; promoted trials resume from the preceding rung")
    ap.add_argument("--sequential", action=argparse.BooleanOptionalAction, default=True,
                    help="control repeated rung looks with pre-registered alpha spending")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha-spending", default="obrien_fleming",
                    choices=["obrien_fleming", "pocock", "linear"])
    ap.add_argument("--factorial", action="store_true", help="add compatible interaction arms")
    ap.add_argument("--factorial-order", type=int, default=2)
    ap.add_argument("--eval-lengths", default="0.5,1,2,4,8")
    ap.add_argument("--eval-noise", default="0.05,0.10")
    ap.add_argument("--confirm-top", type=int, default=1,
                    help="fresh-seed confirmatory reruns for the top N exploratory arms; 0 disables")
    ap.add_argument("--confirm-seeds", type=int, default=8)
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = make_task(args.task)
    requested = [x.strip() for x in args.configs.split(",") if x.strip()]
    factors = [x for x in requested if x != "baseline"]
    if args.factorial:
        generated = factorial_configs(factors, args.factorial_order); LEVERS.update(generated)
        configs = ["baseline", *generated]
    else:
        configs = ["baseline", *[x for x in requested if x != "baseline"]]
    eval_factors = tuple(float(x) for x in args.eval_lengths.split(",") if x)
    eval_noise = tuple(float(x) for x in args.eval_noise.split(",") if x)
    rung_fracs = [1.0] if not args.halving else sorted({min(1.0, max(0.001, float(x)))
                                                        for x in args.halving_rungs.split(",")})
    if rung_fracs[-1] != 1.0: rung_fracs.append(1.0)
    campaign_cfg = {**vars(args), "resolved_configs": {n: LEVERS.get(n) for n in configs},
                    "device": dev, "rungs": rung_fracs,
                    "decision_policy": {"primary_metric": "acc", "direction": "maximize",
                                        "alpha": args.alpha, "multiple_testing": "holm",
                                        "sequential_testing": bool(args.sequential),
                                        "alpha_spending": args.alpha_spending,
                                        "planned_looks": len(rung_fracs),
                                        "exploration_rank": "mean_minus_standard_error",
                                        "confirmation": "positive_delta_and_corrected_significance",
                                        "fresh_seed_offset": 10_000}}
    cid = registry.create_campaign(args.task, campaign_cfg, name=args.campaign_name,
                                   capsule=registry.capture_capsule({"task": args.task}), db=args.db)
    arm_ids, valid = {}, []
    print(f"campaign={cid} task={task.name} configs={configs} seeds={args.seeds} dev={dev}", flush=True)
    for cfg in configs:
        arm_ids[cfg] = registry.ensure_arm(cid, cfg, LEVERS.get(cfg, {}), db=args.db)
        ok, why = preflight(task, args.d_model, args.n_layers, args.head_size, cfg, dev, args.batch)
        if ok:
            valid.append(cfg); print(f"  [{cfg}] preflight {why}", flush=True)
        else:
            registry.set_arm_status(arm_ids[cfg], "preflight_rejected", db=args.db)
            print(f"  [{cfg}] PREFLIGHT REJECTED: {why}", flush=True)

    active, final_runs, prior_checkpoints, final_comparisons = valid, {}, {}, {}
    try:
        for rung, frac in enumerate(rung_fracs):
            if not active: break
            budget_steps = max(1, int(round(args.steps * frac)))
            budget_minutes = args.minutes * frac if args.minutes else 0.0
            print(f"\n--- rung {rung+1}/{len(rung_fracs)} budget="
                  f"{budget_minutes:.2f}min" if budget_minutes else f"\n--- rung {rung+1}/{len(rung_fracs)} budget={budget_steps} steps", flush=True)
            rung_runs = {}
            for cfg in list(active):
                runs = []
                for s in range(args.seeds):
                    started = time.time()
                    try:
                        ckpt_dir = os.path.join(args.checkpoint_dir, f"campaign_{cid}", cfg,
                                                f"seed_{s:04d}")
                        ckpt_path = os.path.join(ckpt_dir, f"rung_{rung:02d}.pt")
                        run = train_eval(task, args.d_model, args.n_layers, args.head_size, cfg, s, dev,
                                         budget_steps, args.batch, args.lr, budget_minutes, args.optimizer,
                                         args.weight_decay, args.warmup, muon_opts_from(args), fp8=args.fp8,
                                         do_compile=args.compile, gen_block=args.gen_block,
                                         eval_factors=eval_factors, eval_noise=eval_noise,
                                         data_seed=500_000 + s, profile=args.profile,
                                         resume_path=prior_checkpoints.get((cfg, s)),
                                         checkpoint_path=ckpt_path, schedule_steps=args.steps,
                                         schedule_minutes=args.minutes)
                        series, prof, rng_state = run.pop("_series"), run.pop("_profile"), run.pop("_rng")
                        saved_ckpt = run.pop("_checkpoint", None)
                        tid = registry.record_trial(cid, arm_ids[cfg], s, rung,
                                                    budget_minutes * 60 if budget_minutes else budget_steps,
                                                    run, series=series, profile=prof, rng=rng_state,
                                                    started_ts=started, db=args.db)
                        if saved_ckpt:
                            registry.record_artifact(cid, saved_ckpt, "training_checkpoint", trial_id=tid,
                                                     metadata={"arm": cfg, "seed": s, "rung": rung,
                                                               "resumes": prior_checkpoints.get((cfg, s))},
                                                     db=args.db)
                            prior_checkpoints[(cfg, s)] = saved_ckpt
                        runs.append(run)
                    except Exception as e:
                        registry.record_trial(cid, arm_ids[cfg], s, rung,
                                              budget_minutes * 60 if budget_minutes else budget_steps,
                                              None, status="failed", error=f"{type(e).__name__}: {e}",
                                              started_ts=started, db=args.db)
                        print(f"  [{cfg} seed={s}] FAILED: {type(e).__name__}: {e}", flush=True)
                if runs:
                    rung_runs[cfg] = runs
                    agg = _aggregate_runs(runs); am, ast = agg["acc"]
                    print(f"  [{cfg}] acc {am:.3f}±{ast:.3f} tok/s {agg.get('tok_per_sec',(0,))[0]:.0f}", flush=True)
            final_runs = rung_runs
            look_kwargs = ({"look": rung + 1, "total_looks": len(rung_fracs),
                            "alpha": args.alpha, "spending": args.alpha_spending}
                           if args.sequential else {"alpha": args.alpha})
            final_comparisons = _paired_comparisons(rung_runs, seed=cid + rung, **look_kwargs)
            for name, st in final_comparisons.items():
                registry.record_comparison(cid, arm_ids[name], arm_ids["baseline"], "acc",
                                           f"explore_look_{rung + 1}", st, db=args.db)
                seq = st.get("sequential", {})
                boundary = f" α_look={seq['increment']:.3g}" if seq else ""
                print(f"  look {rung+1}: {name} Δ{st['delta']:+.4f} "
                      f"p_holm={st['p_adjusted']:.4g}{boundary}", flush=True)
            if rung < len(rung_fracs) - 1:
                challengers = [n for n in active if n != "baseline" and n in rung_runs]
                ranked = sorted(challengers, key=lambda n: _aggregate_runs(rung_runs[n])["acc"][0]
                                - _aggregate_runs(rung_runs[n])["acc"][1] / max(len(rung_runs[n])**0.5, 1), reverse=True)
                keep = max(1, math.ceil(len(ranked) / max(args.halving_eta, 2))) if ranked else 0
                promoted = ranked[:keep]
                eliminated = set(challengers) - set(promoted)
                for n in eliminated: registry.set_arm_status(arm_ids[n], f"eliminated_rung_{rung}", db=args.db)
                active = (["baseline"] if "baseline" in rung_runs else []) + promoted
                print(f"  promote -> {active}; eliminated -> {sorted(eliminated)}", flush=True)

        results = {name: _aggregate_runs(runs) for name, runs in final_runs.items()}
        comparisons = final_comparisons or _paired_comparisons(final_runs, seed=cid, alpha=args.alpha)
        for name, st in comparisons.items():
            registry.record_comparison(cid, arm_ids[name], arm_ids["baseline"], "acc", "explore", st, db=args.db)
            print(f"  Δ {name}: {st['delta']:+.4f} 95%CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] "
                  f"p_holm={st['p_adjusted']:.4g} dz={st['effect_size']:.2f} next_n={st['recommended_n']}", flush=True)
        for name, agg in results.items():
            registry.record(args.task, name, len(final_runs[name]), args.steps,
                            {k: list(v) for k, v in agg.items()}, db=args.db)
        rows = [{"name": n, **{k: v[0] for k,v in a.items()}} for n,a in results.items()]
        for row, flag in zip(rows, pareto_front(rows)):
            print(f"  {'PARETO' if flag else '      '} {row['name']:20} acc={row.get('acc',0):.3f} "
                  f"time={row.get('train_seconds',0):.1f}s peak={row.get('peak_alloc_mb',0):.0f}MB", flush=True)

        # Fresh, previously unused seeds: exploratory winners cannot confirm themselves.
        challengers = sorted([n for n in results if n != "baseline"],
                             key=lambda n: results[n]["acc"][0], reverse=True)[:args.confirm_top]
        if challengers and args.confirm_seeds > 0:
            confirm_cfg = {**campaign_cfg, "selected": challengers, "fresh_seed_offset": 10_000}
            ccid = registry.create_campaign(args.task, confirm_cfg, name=(args.campaign_name + " confirm").strip(),
                                            phase="confirm", parent_id=cid,
                                            capsule=registry.capture_capsule({"parent_campaign": cid}), db=args.db)
            c_names, c_arm, c_runs = ["baseline", *challengers], {}, {}
            for name in c_names: c_arm[name] = registry.ensure_arm(ccid, name, LEVERS[name], db=args.db)
            print(f"\n--- confirm campaign={ccid} arms={c_names} fresh_seeds={args.confirm_seeds} ---", flush=True)
            for name in c_names:
                c_runs[name] = []
                for j in range(args.confirm_seeds):
                    s = 10_000 + j; started = time.time()
                    run = train_eval(task, args.d_model, args.n_layers, args.head_size, name, s, dev,
                                     args.steps, args.batch, args.lr, args.minutes, args.optimizer,
                                     args.weight_decay, args.warmup, muon_opts_from(args), fp8=args.fp8,
                                     do_compile=args.compile, gen_block=args.gen_block,
                                     eval_factors=eval_factors, eval_noise=eval_noise,
                                     data_seed=600_000 + j, profile=args.profile)
                    series, prof, rng_state = run.pop("_series"), run.pop("_profile"), run.pop("_rng")
                    registry.record_trial(ccid, c_arm[name], s, 0, args.minutes*60 or args.steps, run,
                                          series=series, profile=prof, rng=rng_state, phase="confirm",
                                          started_ts=started, db=args.db); c_runs[name].append(run)
            cstats = _paired_comparisons(c_runs, seed=ccid)
            for name, st in cstats.items():
                confirmed = bool(st["significant"] and st["delta"] > 0)
                registry.record_comparison(ccid, c_arm[name], c_arm["baseline"], "acc", "confirm", st,
                                           confirmed=confirmed, db=args.db)
                print(f"  {name}: Δ{st['delta']:+.4f} CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] "
                      f"p_holm={st['p_adjusted']:.4g} {'CONFIRMED' if confirmed else 'not confirmed'}", flush=True)
            registry.finish_campaign(ccid, db=args.db)
        registry.finish_campaign(cid, db=args.db)
    except BaseException:
        registry.finish_campaign(cid, status="failed", db=args.db)
        raise
    if args.out:
        json.dump({"campaign_id": cid, "results": results}, open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
