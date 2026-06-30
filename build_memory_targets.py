#!/usr/bin/env python
"""Stage 0.5 — extract Gated-DeltaNet teacher memory targets for SMT/DMT.

For a chosen GDN (`linear_attn`) layer of the untouched Qwen3.5-9B teacher, run
the model on calibration windows and capture, per window:

  * h          : the layer input hidden states              [T, C]   (fp16)
  * block_out  : the linear_attn block output               [T, C]   (fp16)
  * state      : the GDN recurrent state at stride boundaries
                 [n_bounds+1, num_v_heads, head_k, head_v]  (fp16)
                 state[j] = recurrent state AFTER j*stride tokens (state[0]=0).

We intercept exactly at the kernel boundary by swapping the instance attribute
`linear_attn.chunk_gated_delta_rule` (modular_qwen3_5.py:284) with a wrapper that
(1) calls the real kernel for the true forward output, and (2) re-runs it
chunk-wise with output_final_state=True to record boundary states. No
re-implementation of the conv/in_proj preprocessing.

Output: a cache dir with memmaps (h.f16, block_out.f16, state.f16) + manifest.json,
consumed by smt_dmt.py (codec fit + SMT/DMT training).

Calibration data: a raw-int32 shard dir (babyllm qwen3.x_*_val) OR a flat uint32
tokens.bin (build_qwen35_data.py output). Both are id-compatible with the 9B.
"""
from __future__ import annotations

import sys
# torch 2.11 in this venv has a torchvision ABI mismatch (operator
# torchvision::nms missing). Qwen3.5's multimodal modeling lazily pulls in
# torchvision for its (unused-by-us) vision tower; mark it unavailable so
# transformers' guards skip it and the text decoder loads cleanly.
sys.modules.setdefault("torchvision", None)

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# calibration token source
# ---------------------------------------------------------------------------
def load_token_stream(path: str) -> np.ndarray:
    """Return a 1-D token array (memmap-backed concat) from either a shard dir
    (raw int32 shard_*.npy) or a flat uint32 tokens.bin."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(glob.glob(str(p / "shard_*.npy")))
        if shards:
            parts = [np.fromfile(s, dtype=np.int32) for s in shards]
            return np.concatenate(parts)
        tb = p / "tokens.bin"
        if tb.exists():
            return np.memmap(tb, dtype=np.uint32, mode="r")
        raise FileNotFoundError(f"no shard_*.npy or tokens.bin under {p}")
    # a file
    if p.suffix == ".bin" or "tokens" in p.name:
        return np.memmap(p, dtype=np.uint32, mode="r")
    return np.fromfile(p, dtype=np.int32)


# ---------------------------------------------------------------------------
# kernel-boundary capture
# ---------------------------------------------------------------------------
class _GDNStateCapture:
    """Wrapper installed over linear_attn.chunk_gated_delta_rule for one layer."""

    def __init__(self, real_fn, stride: int):
        self.real_fn = real_fn
        self.stride = stride
        self.want_states = True  # set False to skip the boundary re-run when states aren't needed
        self.states = None  # filled per forward: [n_bounds+1, B, Hv, Dk, Dv]

    def __call__(self, query, key, value, *, g, beta, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=True, **kw):
        # 1) true forward output (exactly the model's call)
        out, final = self.real_fn(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw,
        )
        if not self.want_states:
            self.states = None
            return out, final
        # 2) chunk-wise re-run to record boundary states
        B, T, Hv, Dk = query.shape
        Dv = value.shape[-1]
        st = None
        boundaries = [torch.zeros(B, Hv, Dk, Dv, device=query.device, dtype=torch.float32)]
        pos = 0
        while pos < T:
            hi = min(pos + self.stride, T)
            _o, st = self.real_fn(
                query[:, pos:hi], key[:, pos:hi], value[:, pos:hi],
                g=g[:, pos:hi], beta=beta[:, pos:hi],
                initial_state=st, output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            boundaries.append(st.float())
            pos = hi
        self.states = torch.stack(boundaries, dim=0)  # [n_bounds+1, B, Hv, Dk, Dv]
        return out, final


def find_gdn_layer(model, layer_idx: int):
    """Return the linear_attn (Qwen3_5GatedDeltaNet) module for the given index."""
    for name, mod in model.named_modules():
        if mod.__class__.__name__.endswith("GatedDeltaNet") and \
                getattr(mod, "layer_idx", None) == layer_idx:
            return name, mod
    raise RuntimeError(f"no GatedDeltaNet layer with layer_idx={layer_idx} found")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.5-9B-Base")
    ap.add_argument("--layer", type=int, required=True, help="GDN layer index to extract")
    ap.add_argument("--data", required=True, help="shard dir or tokens.bin (val split)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument("--state-stride", type=int, default=64)
    ap.add_argument("--out", required=True, help="output cache dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    print(f"loading teacher {args.model_dir} ({args.dtype}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=dtype, low_cpu_mem_usage=True,
    ).to(args.device).eval()

    # decoder backbone without lm_head (mirrors train_mla._text_model)
    text_model = getattr(model.model, "language_model", model.model)

    name, gdn = find_gdn_layer(model, args.layer)
    print(f"target layer: {name}  (heads v={gdn.num_v_heads} k={gdn.num_k_heads} "
          f"head_k={gdn.head_k_dim} head_v={gdn.head_v_dim})", flush=True)

    cap = _GDNStateCapture(gdn.chunk_gated_delta_rule, args.state_stride)
    gdn.chunk_gated_delta_rule = cap

    h_box, out_box = {}, {}

    def _pre(m, a, kw):
        hs = a[0] if a else kw.get("hidden_states")
        h_box["h"] = hs.detach()

    pre = gdn.register_forward_pre_hook(_pre, with_kwargs=True)
    post = gdn.register_forward_hook(
        lambda m, a, o: out_box.__setitem__("o", o.detach()))

    toks = load_token_stream(args.data)
    N = len(toks)
    T = args.seq_len
    max_start = N - (T + 1)
    rng = np.random.default_rng(args.seed)
    C = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    n_bounds = (T + args.state_stride - 1) // args.state_stride + 1
    Hv, Dk, Dv = gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim

    h_mm = np.memmap(out / "h.f16", dtype=np.float16, mode="w+", shape=(args.n_windows, T, C))
    o_mm = np.memmap(out / "block_out.f16", dtype=np.float16, mode="w+", shape=(args.n_windows, T, C))
    s_mm = np.memmap(out / "state.f16", dtype=np.float16, mode="w+", shape=(args.n_windows, n_bounds, Hv, Dk, Dv))

    parity_max = 0.0
    with torch.no_grad():
        for i in range(args.n_windows):
            start = int(rng.integers(0, max_start + 1))
            ids = torch.as_tensor(np.asarray(toks[start:start + T], dtype=np.int64),
                                  device=args.device).unsqueeze(0)
            text_model(input_ids=ids, use_cache=False)
            h = h_box["h"][0].float().cpu().numpy()
            bo = out_box["o"][0].float().cpu().numpy()
            st = cap.states[:, 0].float().cpu().numpy()  # [n_bounds, Hv, Dk, Dv]
            # sanity: last boundary state should be finite
            assert np.isfinite(st).all(), "non-finite teacher state"
            h_mm[i] = h.astype(np.float16)
            o_mm[i] = bo.astype(np.float16)
            s_mm[i, :st.shape[0]] = st.astype(np.float16)
            if i < 2:
                print(f"  win {i}: h{h.shape} out{bo.shape} state{st.shape} "
                      f"|state|rms={np.sqrt((st**2).mean()):.3f}", flush=True)
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{args.n_windows} windows", flush=True)
    h_mm.flush(); o_mm.flush(); s_mm.flush()
    pre.remove(); post.remove()

    manifest = {
        "model_dir": args.model_dir, "layer": args.layer,
        "seq_len": T, "n_windows": args.n_windows, "state_stride": args.state_stride,
        "n_bounds": int(n_bounds), "hidden_size": int(C),
        "num_v_heads": int(Hv), "head_k_dim": int(Dk), "head_v_dim": int(Dv),
        "files": {"h": "h.f16", "block_out": "block_out.f16", "state": "state.f16"},
        "dtype": "float16",
        "shapes": {"h": [args.n_windows, T, C], "block_out": [args.n_windows, T, C],
                   "state": [args.n_windows, int(n_bounds), int(Hv), int(Dk), int(Dv)]},
        "data": args.data,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote cache -> {out}\n{json.dumps(manifest['shapes'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
