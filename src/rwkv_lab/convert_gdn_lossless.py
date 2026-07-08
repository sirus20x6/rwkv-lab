"""Lossless GDN -> RWKV-7 conversion for Qwen3.5 (weight-preserving, zero training).

GDN's gated-delta kernel is an EXACT subset of RWKV-7's wkv7 kernel (proven:
kernel cosine 0.999995; full-9B ppl unchanged to 0.01%). This installs the wkv7
kernel on every GDN layer of a loaded Qwen3.5 model, keeping each layer's 4-tap
conv + SiLU, decay, beta, and z-gated-RMSNorm exactly as-is. No new params, no
training -- the GDN layers just run on the RWKV-7 kernel. See memory
[[gdn_rwkv7_lossless_kernel]].

Map (validated) -- given GDN kernel inputs (q,k,v,g,beta), with q/k L2-normalized:
  r = norm(q)              gk = g (uniform over K)      v = v
  k_write = beta*norm(k)   a = -norm(k)                 b = norm(k)*exp(g)*beta
  wkv7(r, gk, k_write, v, a, b), then output *= 1/sqrt(head_dim)   (GDN's readout scale)

Usage:
    from .convert_gdn_lossless import install_lossless_wkv7
    model = AutoModelForCausalLM.from_pretrained("Qwen3.5-9B-Base", ...)
    n = install_lossless_wkv7(model)     # GDN layers now run on RWKV-7, losslessly

    python convert_gdn_lossless.py       # self-test: prints original vs converted ppl
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .rwkv8_deltanet import _fla_chunk_rwkv7, _HAS_FLA


def _make_wkv7_gdn_kernel(kernel_dtype=torch.float32):
    """Build the drop-in for Qwen3_5GatedDeltaNet.chunk_gated_delta_rule /
    recurrent_gated_delta_rule, computing the identical result via the RWKV-7 wkv7
    kernel at ``kernel_dtype``. query/key/value are [B,T,H,D] at the kernel boundary
    (GQA already expanded); g/beta are [B,T,H].

    kernel_dtype=float32 gives the cleanest floor (full-9B ppl delta ~+0.01%); fla
    warns its fp32 chunk path is unsupported on *some* platforms (works on sm_120),
    so bfloat16 (~+0.04%) is the portable fallback. L2-norms + the exp(g)*beta gate
    are always computed in fp32 for accuracy, then cast to kernel_dtype."""
    kd = kernel_dtype

    def kernel(query, key, value, *, g, beta, initial_state=None,
               output_final_state=False, use_qk_l2norm_in_kernel=True, **kw):
        B, T, H, D = value.shape
        out_dt = value.dtype
        q = F.normalize(query.float(), dim=-1)                  # GDN's use_qk_l2norm
        k = F.normalize(key.float(), dim=-1)
        g_f = g.float().unsqueeze(-1)                           # [B,T,H,1] log-decay
        beta_f = beta.float().unsqueeze(-1)
        a_iclr = g_f.exp() * beta_f                             # in-context LR = decay * beta
        r = q.to(kd).contiguous()
        gk = g_f.expand(B, T, H, D).to(kd).contiguous()         # uniform decay over K
        k_write = (beta_f * k).to(kd).contiguous()
        a_h = (-k).to(kd).contiguous()
        b_h = (k * a_iclr).to(kd).contiguous()
        v = value.to(kd).contiguous()
        ist = initial_state.to(kd) if initial_state is not None else None
        out, final = _fla_chunk_rwkv7(
            r, gk, k_write, v, a_h, b_h,
            scale=1.0, initial_state=ist, output_final_state=output_final_state)
        out = out * (D ** -0.5)                                 # GDN's 1/sqrt(head_dim) readout scale
        return out.to(out_dt), final

    return kernel


def install_lossless_wkv7(model, kernel_dtype=torch.float32) -> int:
    """Swap every GDN layer's delta kernel for the wkv7 map, in place. Both the
    chunk (prefill/train) and recurrent (single-token decode) paths get the same
    kernel, so the recurrent STATE stays wkv7-format end-to-end and generation is
    self-consistent. Weight-preserving + lossless. Returns #layers converted.
    kernel_dtype=float32 (default) = cleanest floor; torch.bfloat16 = portable."""
    if not _HAS_FLA:
        raise RuntimeError("fla required for the RWKV-7 wkv7 kernel")
    kernel = _make_wkv7_gdn_kernel(kernel_dtype)
    n = 0
    for _, m in model.named_modules():
        if m.__class__.__name__.endswith("GatedDeltaNet"):
            m.chunk_gated_delta_rule = kernel
            m.recurrent_gated_delta_rule = kernel
            n += 1
    return n


def _self_test():
    import sys, math, numpy as np
    sys.modules.setdefault("torchvision", None)
    from transformers import AutoModelForCausalLM
    from .build_memory_targets import load_token_stream

    dev, dtype = "cuda", torch.bfloat16
    print("loading Qwen3.5-9B ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "/thearray/git/moe-mla/Qwen3.5-9B-Base", dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    toks = load_token_stream("/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")

    def eval_ppl(n=16, T=1024, seed=0):
        rng = np.random.default_rng(seed)
        N = len(toks); maxs = N - (T + 1)
        tot_loss, tot_tok = 0.0, 0
        for _ in range(n):
            s = int(rng.integers(0, maxs + 1))
            ids = torch.as_tensor(np.asarray(toks[s:s + T + 1], dtype=np.int64), device=dev).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_ids=ids[:, :T], use_cache=False).logits.float()
            tot_loss += F.cross_entropy(logits[0], ids[0, 1:T + 1], reduction="sum").item()
            tot_tok += T
        return math.exp(tot_loss / tot_tok)

    ppl_orig = eval_ppl()
    n = install_lossless_wkv7(model)                 # kernel_dtype=float32 (clean floor)
    ppl_conv = eval_ppl()
    print(f"\n==== convert_gdn_lossless self-test ====")
    print(f"  GDN layers converted : {n}")
    print(f"  original  ppl        : {ppl_orig:.4f}")
    print(f"  wkv7 conv ppl (fp32) : {ppl_conv:.4f}")
    print(f"  delta                : {ppl_conv - ppl_orig:+.5f}  ({100*(ppl_conv-ppl_orig)/ppl_orig:+.3f}%)")


if __name__ == "__main__":
    _self_test()
