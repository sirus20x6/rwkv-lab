"""
Merge all Qwen-tokenized .bin files under /thearray/data/formatted_tokens/
EXCLUDING anything under cvevc/, into a single uint32 stream.

Result: /thearray/data/non_cvevc_tokens.bin
Manifest: /thearray/data/non_cvevc_tokens.bin.manifest.json
  - lists source files + cumulative offsets so we can later trace a given
    token position back to its source file.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

SRC_ROOT = Path("/thearray/data/formatted_tokens")
OUT_BIN = Path("/thearray/data/non_cvevc_tokens.bin")
OUT_MF = Path(str(OUT_BIN) + ".manifest.json")

EXCLUDE_PREFIX = "cvevc"   # skip any path under formatted_tokens/cvevc/


def enumerate_sources() -> list[Path]:
    sources: list[Path] = []
    for p in sorted(SRC_ROOT.rglob("*.bin")):
        rel = p.relative_to(SRC_ROOT)
        if rel.parts and rel.parts[0] == EXCLUDE_PREFIX:
            continue
        if str(p).endswith(".manifest.json"):
            continue
        sources.append(p)
    return sources


def main() -> None:
    sources = enumerate_sources()
    total_bytes = sum(p.stat().st_size for p in sources)
    print(f"sources: {len(sources)} files, {total_bytes/1e9:.2f} GB "
          f"= {total_bytes/4/1e9:.2f} B tokens")
    print(f"target:  {OUT_BIN}")
    print()

    manifest = {
        "output": str(OUT_BIN),
        "dtype": "uint32",
        "total_tokens": total_bytes // 4,
        "total_bytes": total_bytes,
        "sources": [],
    }

    t0 = time.time()
    with open(OUT_BIN, "wb") as fout:
        offset_bytes = 0
        for i, src in enumerate(sources):
            sz = src.stat().st_size
            with open(src, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 24)   # 16MB chunks
            manifest["sources"].append({
                "path": str(src),
                "byte_offset": offset_bytes,
                "bytes": sz,
                "tokens": sz // 4,
            })
            offset_bytes += sz
            done_pct = 100 * offset_bytes / total_bytes
            elapsed = time.time() - t0
            eta = elapsed * (total_bytes - offset_bytes) / max(1, offset_bytes)
            print(f"  [{i+1:3d}/{len(sources)}] {sz/1e9:6.2f} GB  "
                  f"{done_pct:5.1f}% done  ETA {eta:.0f}s  "
                  f"{src.relative_to(SRC_ROOT)}")

    OUT_MF.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"done in {time.time()-t0:.0f}s")
    print(f"wrote {OUT_BIN} ({total_bytes/1e9:.2f} GB, {total_bytes//4:,} tokens)")


if __name__ == "__main__":
    main()
