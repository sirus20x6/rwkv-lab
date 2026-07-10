"""Sample text from a rwkv_pretrain checkpoint — the lab's train→inspect closer.

Full-recompute decoding: each new token re-runs the whole prefix through the normal training
forward. That is O(T²) but lab models are tiny (ms per forward), and it makes every lever correct
for free — seed-chain re-chains, DeepEmbed re-gates, and Engram's suffix-automaton recall re-runs
over the grown sequence (including the just-generated tokens, so the copy head can point at them).
No per-layer streaming-state API needed.

Tokenizer: encode shells out to ztok (its trie handles the RWKV world-vocab .txt); decode parses
the vocab file locally (`idx 'literal' bytelen` per line → id→bytes table).

    python -m rwkv_lab.generate --ckpt runs/lm_blend_seedchain/ckpt.pt \
        --prompt "Write a Python function that reverses a linked list." --max-new 300

Checkpoints written since the arch-record change rebuild themselves (blob["arch"]); older ones
need the --d-model/--n-layers/... fallback flags.
"""
from __future__ import annotations
import argparse, ast, json, os, subprocess
import torch
import torch.nn.functional as F

ZTOK = os.environ.get("ZTOK", "/thearray/git/ztok/zig-out/bin/ztok")
VOCAB = os.environ.get("VOCAB", "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt")
SEP = 1                                          # '\x00' — the corpus doc separator (EOD)


class WorldVocab:
    """RWKV world vocab: local decode table + ztok-backed encode."""

    def __init__(self, path: str = VOCAB):
        self.tok = {0: b""}
        for line in open(path, encoding="utf-8"):
            sp1, sp2 = line.index(" "), line.rindex(" ")
            v = ast.literal_eval(line[sp1 + 1:sp2])
            self.tok[int(line[:sp1])] = v.encode("utf-8") if isinstance(v, str) else v

    def decode(self, ids) -> str:
        return b"".join(self.tok.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        out = subprocess.run([ZTOK, "encode", "--model", VOCAB, text],
                             capture_output=True, text=True, check=True)
        return [int(t) for t in out.stdout.split()]


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
                   de_emb_res=bool(arch.get("de_emb_res"))).to(device, torch.bfloat16)
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


@torch.no_grad()
def sample(model, ids: list[int], *, max_new: int = 200, temperature: float = 0.8,
           top_p: float = 0.95, top_k: int = 0, stop_at_sep: bool = True,
           device: str = "cuda", seed: int | None = None) -> list[int]:
    """Autoregressive sampling by full-prefix recompute. Returns only the NEW token ids."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits = model(x)[0, -1].float()
        logits[0] = -float("inf")                # PAD is never a real continuation
        if temperature <= 0:                     # greedy
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = logits.topk(min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            if 0.0 < top_p < 1.0:                # nucleus
                probs, idx = logits.softmax(-1).sort(descending=True)
                keep = int((probs.cumsum(-1) < top_p).sum()) + 1
                mask = torch.full_like(logits, -float("inf"))
                mask[idx[:keep]] = logits[idx[:keep]]
                logits = mask
            nxt = int(torch.multinomial(logits.softmax(-1), 1))
        if stop_at_sep and nxt == SEP:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


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
    ap.add_argument("--json", action="store_true", help="machine output (for the trainboard)")
    # fallback arch flags for pre-arch-record checkpoints
    ap.add_argument("--d-model", type=int, default=0); ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--head-size", type=int, default=64)
    args = ap.parse_args()

    override = ({"d_model": args.d_model, "n_layers": args.n_layers, "head_size": args.head_size}
                if args.d_model and args.n_layers else None)
    model, blob = build_from_ckpt(args.ckpt, args.device, use_ema=args.use_ema,
                                  arch_override=override)
    vocab = WorldVocab()
    text = args.prompt if args.raw else f"User: {args.prompt}\n\nAssistant:"
    ids = vocab.encode(text)
    new = sample(model, ids, max_new=args.max_new, temperature=args.temperature,
                 top_p=args.top_p, top_k=args.top_k, stop_at_sep=not args.no_stop_sep,
                 device=args.device, seed=args.seed)
    completion = vocab.decode(new)
    if args.json:
        print(json.dumps({"config": blob.get("config", ""), "step": blob.get("step", 0),
                          "prompt": text, "completion": completion, "tokens": len(new)}))
    else:
        print(f"[{blob.get('config', '?')} @ step {blob.get('step', '?')}"
              + (" · EMA" if args.use_ema else "") + f" · {len(new)} tokens]\n")
        print(text + completion)


if __name__ == "__main__":
    main()
