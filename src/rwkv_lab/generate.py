"""Sample text from a rwkv_pretrain checkpoint — the lab's train→inspect closer.

``--engine megakernel`` uses the locally compiled Triton/Inductor/CUDA-Graph execution plan, with
device-side greedy feedback and exact-shape compiled prefill; an optional checkpoint-bound ``.pt2``
AOT artifact skips decode compilation on a matching GPU/compiler runtime.
``--engine auto`` selects it only after local production qualification, otherwise using exact
constant-state RWKV decoding for compatible checkpoints. Experimental levers without a causal state
contract fall back to full-prefix recomputation: seed-chain re-chains, Engram re-runs recall over
generated history, and other levers retain their training semantics. The JSON output records the
selected engine, fallback reason, plan identity, compile time, elapsed time, and tokens/s.

Tokenizer: encode shells out to ztok (its trie handles the RWKV world-vocab .txt); decode parses
the vocab file locally (`idx 'literal' bytelen` per line → id→bytes table).

    python -m rwkv_lab.generate --ckpt runs/lm_blend_seedchain/ckpt.pt \
        --prompt "Write a Python function that reverses a linked list." --max-new 300

Checkpoints written since the arch-record change rebuild themselves (blob["arch"]); older ones
need the --d-model/--n-layers/... fallback flags.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
import subprocess
import time
import torch

ZTOK = os.environ.get("ZTOK", "/thearray/git/ztok/zig-out/bin/ztok")
VOCAB = os.environ.get("VOCAB", "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt")
SEP = 1                                          # '\x00' — the corpus doc separator (EOD)


class WorldVocab:
    """RWKV world vocab: local decode table + ztok-backed encode."""

    def __init__(self, path: str = VOCAB):
        self.path = path
        self.tok = {0: b""}
        for line in open(path, encoding="utf-8"):
            sp1, sp2 = line.index(" "), line.rindex(" ")
            v = ast.literal_eval(line[sp1 + 1:sp2])
            self.tok[int(line[:sp1])] = v.encode("utf-8") if isinstance(v, str) else v
        # RWKV World uses greedy longest-byte matching.  Keeping the trie in
        # process avoids spawning ztok once for every caption during VLM SFT.
        self._trie: dict[int | None, dict] = {}
        for token_id, value in self.tok.items():
            if not value:
                continue
            node = self._trie
            for byte in value:
                node = node.setdefault(byte, {})
            node[None] = token_id

    def decode(self, ids) -> str:
        return b"".join(self.tok.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        raw, out, pos = text.encode("utf-8"), [], 0
        while pos < len(raw):
            node, best_id, best_end = self._trie, None, pos
            cursor = pos
            while cursor < len(raw) and raw[cursor] in node:
                node = node[raw[cursor]]
                cursor += 1
                if None in node:
                    best_id, best_end = node[None], cursor
            if best_id is None:
                raise ValueError(f"no World-vocab token begins with byte {raw[pos]:#x} at offset {pos}")
            out.append(best_id)
            pos = best_end
        return out


def build_from_ckpt(ckpt_path: str, device: str = "cuda", use_ema: bool = False,
                    arch_override: dict | None = None):
    """Rebuild the exact training architecture from blob['arch'] and load its weights."""
    from rwkv_lab.rwkv_pretrain import RWKV7Small, enable_engram
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = blob.get("arch") or arch_override
    if arch is None:
        raise SystemExit(f"{ckpt_path} predates self-describing checkpoints — pass the "
                         "--d-model/--n-layers/--head-size (+lever) flags to describe it")
    m = RWKV7Small(65536, arch["d_model"], arch["n_layers"], arch["head_size"],
                   arch.get("loop_kw") or {},
                   seed_chain=bool(arch.get("seed_chain")),
                   deepembed=bool(arch.get("deepembed")), de_dim=int(arch.get("de_dim") or 0),
                   de_mode=arch.get("de_mode") or "out", de_shift=bool(arch.get("de_shift")),
                   de_emb_res=bool(arch.get("de_emb_res")),
                   routing_free_kw=({"n_experts": int(arch.get("routing_free_experts") or 4),
                                     "rank": int(arch.get("routing_free_rank") or 32),
                                     "threshold": float(arch.get("routing_free_threshold", 0.2)),
                                     "balance_interpolation": float(arch.get("routing_free_balance", 0.5))}
                                    if arch.get("routing_free_moe") else None))
    if arch.get("byte_aware"):
        from rwkv_lab.tokenizer_experiments import install_byte_aware_embedding
        # The checkpoint carries the byte lookup buffers; construct their
        # geometry here, then load the exact saved tables below.
        install_byte_aware_embedding(m, {}, max_bytes=int(arch.get("byte_aware_max_bytes") or 16),
                                     byte_dim=int(arch.get("byte_aware_dim") or 0))
    if arch.get("state_offset"):
        from rwkv_lab.state_tuning import install_state_offset_adapter
        install_state_offset_adapter(m, interval=int(arch.get("state_offset_interval") or 1))
    if arch.get("online_memory"):
        from rwkv_lab.online_memory import install_online_memory
        install_online_memory(m, d_memory=(int(arch.get("online_memory_dim") or 0) or None),
                              mode=arch.get("online_memory_mode") or "titans",
                              learning_rate=float(arch.get("online_memory_lr") or 0.05),
                              retention=float(arch.get("online_memory_retention") or 0.99),
                              atlas_window=int(arch.get("online_memory_window") or 4))
    m = m.to(device, torch.bfloat16)
    if arch.get("nvfp4"):
        from rwkv_lab.nvfp4 import convert_to_nvfp4_training
        convert_to_nvfp4_training(
            m, rht=bool(arch.get("nvfp4_rht")),
            backend=arch.get("nvfp4_backend") or "fake")
    if arch.get("engram"):
        enable_engram(m, 65536, arch["d_model"], arch["head_size"], arch["n_layers"],
                      loop_count=(arch.get("loop_kw") or {}).get("n_loops", 1),
                      d_row=int(arch.get("engram_drow") or 64),
                      rows=int(arch.get("engram_rows") or 4096),
                      sites=arch.get("engram_sites") or "auto",
                      boundary_id=arch.get("engram_boundary_id"))
    sd = dict(blob["model"])
    if use_ema:
        ema = blob.get("ema")
        if not ema:
            raise SystemExit("--use-ema: checkpoint has no EMA weights (train with --ema)")
        for n, t in ema.items():                 # overlay the shadow weights, dtype-matched
            if n in sd:
                sd[n] = t.to(sd[n].dtype)
    m.load_state_dict(sd)
    m.eval()
    return m, blob


def _select_engine(model, requested: str, device: str = "cuda") -> tuple[str, str]:
    if requested not in ("auto", "megakernel", "recurrent", "prefix"):
        raise ValueError("generation engine must be auto, megakernel, recurrent, or prefix")
    reason = "model does not expose forward_recurrent"
    if hasattr(model, "recurrent_incompatibility"):
        reason = model.recurrent_incompatibility() or ""
    available = hasattr(model, "forward_recurrent") and not reason
    if requested == "recurrent" and not available:
        raise ValueError(f"recurrent generation unavailable: {reason}")
    if requested == "megakernel":
        from rwkv_lab.megakernel import megakernel_incompatibility
        mega_reason = megakernel_incompatibility(model, device)
        if mega_reason:
            raise ValueError(f"megakernel generation unavailable: {mega_reason}")
        return "megakernel", "compiled Triton + Inductor + CUDA Graph execution plan"
    if requested == "auto":
        if getattr(model, "_megakernel_adopted", False):
            return "megakernel", "locally parity/performance-qualified execution plan"
        return ("recurrent", "native constant-size recurrent state") if available else ("prefix", reason)
    return requested, "native constant-size recurrent state" if requested == "recurrent" else reason


def _sample_logits(logits: torch.Tensor, *, temperature: float, top_p: float,
                   top_k: int) -> int:
    logits = logits.float().clone()
    logits[0] = -float("inf")                    # PAD is never a real continuation
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if 0.0 < top_p < 1.0:
        probs, idx = logits.softmax(-1).sort(descending=True)
        keep = int((probs.cumsum(-1) < top_p).sum()) + 1
        filtered = torch.full_like(logits, -float("inf"))
        filtered[idx[:keep]] = logits[idx[:keep]]
        logits = filtered
    return int(torch.multinomial(logits.softmax(-1), 1))


@torch.no_grad()
def sample_with_stats(model, ids: list[int], *, max_new: int = 200,
                      temperature: float = 0.8, top_p: float = 0.95,
                      top_k: int = 0, stop_at_sep: bool = True,
                      device: str = "cuda", seed: int | None = None,
                      engine: str = "auto") -> tuple[list[int], dict]:
    """Decode through the exact recurrent path when qualified, else full-prefix."""

    if not ids:
        raise ValueError("generation prompt is empty")
    if seed is not None:
        torch.manual_seed(seed)
    selected, reason = _select_engine(model, engine, device)
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    if selected == "prefix":
        x = torch.empty((1, len(ids) + max_new), dtype=torch.long, device=device)
        x[:, :len(ids)] = prompt
        length = len(ids)
    else:
        x, length = prompt, len(ids)
    out, state, megakernel = [], None, None
    started = time.perf_counter()
    if selected == "megakernel":
        from rwkv_lab.megakernel import get_megakernel_backend
        megakernel = get_megakernel_backend(model, device=device)
        if temperature <= 0:
            generated = megakernel.generate_greedy(
                x, max_new=max_new,
                stop_token_id=(SEP if stop_at_sep else None),
            )[0].tolist()
            if stop_at_sep and SEP in generated:
                generated = generated[:generated.index(SEP)]
            elapsed = time.perf_counter() - started
            stats = {
                "engine": selected, "fallback_reason": reason, "seconds": elapsed,
                "tokens": len(generated),
                "tokens_per_second": len(generated) / max(elapsed, 1e-12),
                "device_side_greedy": True, "megakernel": megakernel.receipt(),
            }
            return generated, stats
        logits = megakernel.prefill(x)
        next_logits = logits[0, -1]
    elif selected == "recurrent":
        logits, state = model.forward_recurrent(x)
        next_logits = logits[0, -1]
    for step in range(max_new):
        if selected == "prefix":
            next_logits = model(x[:, :length])[0, -1]
        nxt = _sample_logits(next_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        if stop_at_sep and nxt == SEP:
            break
        out.append(nxt)
        token = x.new_tensor([[nxt]])
        if selected == "megakernel" and step + 1 < max_new:
            logits = megakernel.step(token)
            next_logits = logits[0, -1]
        elif selected == "recurrent" and step + 1 < max_new:
            logits, state = model.forward_recurrent(token, state)
            next_logits = logits[0, -1]
        elif selected == "prefix":
            x[:, length:length + 1] = token
            length += 1
    elapsed = time.perf_counter() - started
    stats = {"engine": selected, "fallback_reason": reason, "seconds": elapsed,
             "tokens": len(out), "tokens_per_second": len(out) / max(elapsed, 1e-12)}
    if megakernel is not None:
        stats["megakernel"] = megakernel.receipt()
    return out, stats


@torch.no_grad()
def sample(model, ids: list[int], *, max_new: int = 200, temperature: float = 0.8,
           top_p: float = 0.95, top_k: int = 0, stop_at_sep: bool = True,
           device: str = "cuda", seed: int | None = None,
           engine: str = "auto") -> list[int]:
    """Autoregressive sampling. Returns only newly generated token ids."""

    return sample_with_stats(
        model, ids, max_new=max_new, temperature=temperature, top_p=top_p,
        top_k=top_k, stop_at_sep=stop_at_sep, device=device, seed=seed,
        engine=engine,
    )[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Write a Python function that computes a factorial.")
    ap.add_argument("--raw", action="store_true",
                    help="use the prompt verbatim (default wraps it in the corpus chat format)")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--use-ema", action="store_true", help="sample the EMA shadow weights")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-stop-sep", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", choices=("auto", "megakernel", "recurrent", "prefix"),
                    default="auto", help="megakernel uses a compiled Triton/Inductor/CUDA-Graph "
                    "plan; auto adopts it only after local qualification")
    ap.add_argument("--megakernel-receipt", default="",
                    help="persisted production qualification receipt, checkpoint-hash verified")
    ap.add_argument("--megakernel-artifact", default="",
                    help="qualified .pt2 AOT plan plus adjacent .pt2.json manifest")
    ap.add_argument("--megakernel-serving-prepare", action="store_true",
                    help="serving-only: fold ln0 into emb.weight and release its duplicate")
    ap.add_argument("--json", action="store_true", help="machine output (for the trainboard)")
    # fallback arch flags for pre-arch-record checkpoints
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--head-size", type=int, default=64)
    args = ap.parse_args()

    override = ({"d_model": args.d_model, "n_layers": args.n_layers, "head_size": args.head_size}
                if args.d_model and args.n_layers else None)
    model, blob = build_from_ckpt(args.ckpt, args.device, use_ema=args.use_ema,
                                  arch_override=override)
    if args.megakernel_receipt:
        from rwkv_lab.megakernel import adopt_megakernel_receipt
        adopt_megakernel_receipt(model, args.megakernel_receipt, args.ckpt)
    if args.megakernel_artifact:
        from rwkv_lab.megakernel import adopt_megakernel_artifact
        adopt_megakernel_artifact(model, args.megakernel_artifact, args.ckpt)
    if args.megakernel_serving_prepare:
        from rwkv_lab.megakernel import finalize_megakernel_serving_embedding
        finalize_megakernel_serving_embedding(model)
    vocab = WorldVocab()
    text = args.prompt if args.raw else f"User: {args.prompt}\n\nAssistant:"
    ids = vocab.encode(text)
    new, stats = sample_with_stats(
        model, ids, max_new=args.max_new, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, stop_at_sep=not args.no_stop_sep,
        device=args.device, seed=args.seed, engine=args.engine)
    completion = vocab.decode(new)
    if args.json:
        print(json.dumps({"config": blob.get("config", ""), "step": blob.get("step", 0),
                          "prompt": text, "completion": completion, "tokens": len(new),
                          "generation": stats}))
    else:
        print(f"[{blob.get('config', '?')} @ step {blob.get('step', '?')}"
              + (" · EMA" if args.use_ema else "") + f" · {len(new)} tokens]\n")
        print(text + completion)


if __name__ == "__main__":
    main()
