"""Build a DOC-BOUNDARY-aware token corpus so training windows stay WITHIN one document.

The flat-concatenation corpora we used cut mid-document, so a random window could span two unrelated
files and the model learns spurious cross-doc context. This joins documents with a separator byte
(\\x00 -> World token 1), tokenizes the whole thing with ztok (fast), and writes the token .bin PLUS
a doc-offsets .npy (start index of each doc). rwkv_pretrain --doc-offsets then samples within-doc
windows. Default source = local repo files (each file = one doc); --dataset streams a diverse HF
corpus (each record = one doc) for web/multilingual breadth.

    python -m rwkv_lab.build_corpus --out models/corpus            # -> models/corpus.bin + .off.npy
    python -m rwkv_lab.build_corpus --dataset wikitext --out models/wiki
"""
from __future__ import annotations
import argparse, glob, os, subprocess, tempfile
import numpy as np

ZTOK = os.environ.get("ZTOK", "/thearray/git/ztok/zig-out/bin/ztok")
VOCAB = os.environ.get("VOCAB", "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt")
SEP_TOKEN = 1                                    # "\x00" tokenizes to World token 1


def gather_local(patterns, cap_mb) -> list[str]:
    docs, n = [], 0
    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if "/node_modules/" in fp or "/.git/" in fp or "/zig-out/" in fp:
                continue
            try:
                t = open(fp, errors="ignore").read()
            except Exception:
                continue
            if t.strip():
                docs.append(t.replace("\x00", ""))   # our own separator can't appear inside a doc
                n += len(t)
                if n > cap_mb * 1e6:
                    return docs
    return docs


def gather_hf(name, cap_mb) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True)
    docs, n = [], 0
    for rec in ds:
        t = (rec.get("text") or rec.get("content") or "").replace("\x00", "")
        if t.strip():
            docs.append(t); n += len(t)
            if n > cap_mb * 1e6:
                break
    return docs


def build(docs, out_prefix):
    joined = "\x00".join(docs)
    bin_path = out_prefix + ".bin"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(joined); tmp = f.name
    try:
        subprocess.run([ZTOK, "tokenize-dataset", "--model", VOCAB, "--input", tmp,
                        "--output", bin_path, "--format", "bin", "--dtype", "u16",
                        "--doc-mode", "whole"], check=True, stdout=subprocess.DEVNULL)
    finally:
        os.unlink(tmp)
    toks = np.fromfile(bin_path, dtype=np.uint16)
    sep = np.where(toks == SEP_TOKEN)[0]                        # boundaries -> doc starts
    offsets = np.concatenate([[0], sep + 1]).astype(np.uint64)
    off_path = out_prefix + ".off.npy"
    np.save(off_path, offsets)
    lens = np.diff(np.append(offsets, len(toks)))
    print(f"{len(toks)/1e6:.2f}M tokens, {len(offsets)} docs "
          f"(median {int(np.median(lens))} tok) -> {bin_path} + {off_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="", help="HF dataset name (streaming); empty = local files")
    ap.add_argument("--patterns", default="/thearray/git/moe-mla/**/*.py,/thearray/git/moe-mla/**/*.md")
    ap.add_argument("--cap-mb", type=float, default=12.0)
    args = ap.parse_args()
    if args.dataset:
        docs = gather_hf(args.dataset, args.cap_mb)
    else:
        docs = gather_local(args.patterns.split(","), args.cap_mb)
    print(f"gathered {len(docs)} documents", flush=True)
    build(docs, args.out)


if __name__ == "__main__":
    main()
