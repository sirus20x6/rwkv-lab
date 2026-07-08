"""Load a pretrained BlinkDL RWKV-7 (g1g-1.5b) into OUR modules — the EXP-C payoff.

rwkv_finetune.py loads g1g into fla's model (baseline 2.457 nats). This loads the SAME weights
into our own RWKV8TimeMixDeltaNet stack (RWKV7Small), which is only possible now that the forward
is faithful native g070 (r_k=k_eff, out_correct off, v_first threaded — all Codex-audited). Our
param layout is a direct match to x070 (w1=(C,r), no transposes, unlike the fla remap), so the
remap is a straight name-copy. If the baseline ppl matches fla's ~2.457, the forward fixes are
proven end-to-end AND the loop/latent-prediction levers can run on the REAL pretrained g1g
(wrap the blocks' `att` in LoopedRWKV) instead of only from scratch.

    python -m rwkv_lab.native_g1g --model models/rwkv7-g1g-1.5b.pth   # loads + prints baseline ppl
"""
from __future__ import annotations
import argparse
import torch


def load_g1g_native(path, n_layers=24, d=2048, head_size=64, vocab=65536,
                    decay_r=96, a_r=96, v_r=64, gate_r=256, inter=8192,
                    device="cuda", dtype=torch.bfloat16):
    """Return (model, info). model is an RWKV7Small of our modules holding g1g's pretrained weights."""
    from rwkv_lab.rwkv_pretrain import RWKV7Small
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = RWKV7Small(vocab, d, n_layers, head_size, {},           # {} = bare cores (no loops yet)
                       att_kw=dict(decay_lora=decay_r, a_lora=a_r, v_lora=v_r, gate_lora=gate_r),
                       ffn_hidden=inter)
    tgt = model.state_dict()
    loaded, extra = {}, []
    for gk, gv in sd.items():                                       # direct name-copy (+ reshape)
        if gk in tgt and tgt[gk].numel() == gv.numel():
            loaded[gk] = gv.reshape(tgt[gk].shape).to(tgt[gk].dtype)
        else:
            extra.append((gk, tuple(gv.shape)))
    tgt.update(loaded)
    info = model.load_state_dict(tgt, strict=False)
    unfilled = [k for k in model.state_dict() if k not in loaded]
    return model.to(device, dtype).eval(), dict(loaded=len(loaded), n_ckpt=len(sd),
                                                unfilled=unfilled, extra=extra,
                                                missing=list(info.missing_keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/rwkv7-g1g-1.5b.pth")
    ap.add_argument("--vocab-file", default="research/RWKV-LM/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt")
    args = ap.parse_args()
    dev = "cuda"
    model, info = load_g1g_native(args.model, device=dev)
    print(f"loaded {info['loaded']}/{info['n_ckpt']} tensors; unfilled={len(info['unfilled'])} "
          f"extra={len(info['extra'])}", flush=True)
    if info["unfilled"]:
        print("  UNFILLED (our params with no g1g source):", info["unfilled"][:8])
    if info["extra"]:
        print("  EXTRA (g1g keys with no home):", info["extra"][:8])

    # baseline ppl on World-tokenized text — should match fla's 2.457 if the forward is faithful
    import sys, glob, torch.nn.functional as F
    sys.path.insert(0, "research/RWKV-LM/RWKV-v5/tokenizer")
    from rwkv_tokenizer import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.vocab_file)
    txt = ""
    for fp in sorted(glob.glob("/thearray/data/abliteration_runs/*/*.jsonl"),
                     key=lambda f: -__import__("os").path.getsize(f))[:4]:
        txt += open(fp, errors="ignore").read()
        if len(txt) > 200000:
            break
    ids = tok.encode(txt[:80000]); T = 2048
    x = torch.tensor(ids[:T + 1], dtype=torch.long, device=dev).unsqueeze(0)
    L = x.shape[1]
    with torch.no_grad():
        logits = model(x[:, :L - 1]).float()
        loss = F.cross_entropy(logits[0], x[0, 1:L])
    print(f"\n>>> our-module g1g BASELINE loss = {loss.item():.3f} nats (ppl {loss.exp().item():.2f})")
    print(">>> fla's faithful baseline was 2.457 — a match proves our native forward is correct end-to-end")


if __name__ == "__main__":
    main()
