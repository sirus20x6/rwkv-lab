#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAGE_FLOW_PYTHON_BIN:-$REPO_ROOT/.venv-mage-flow/bin/python}"
export HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Mage-Flow Python environment not found: %s\n' "$PYTHON_BIN" >&2
  printf 'Run scripts/bootstrap_mage_flow_pretrain.sh first.\n' >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
import math
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

model_id = "microsoft/Mage-Flow-Base"
revision = "59a9cfd58cf6ecef28245852c6bdace3f12428a2"
root = Path(snapshot_download(repo_id=model_id, revision=revision))
config = json.loads((root / "transformer" / "config.json").read_text())

layout = {}
for component in ("transformer", "vae"):
    files = sorted((root / component).glob("*.safetensors"))
    tensors = parameters = 0
    dtypes = Counter()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                view = handle.get_slice(key)
                tensors += 1
                parameters += math.prod(view.get_shape())
                dtypes[str(view.get_dtype())] += 1
    layout[component] = {
        "files": len(files),
        "tensors": tensors,
        "parameters": parameters,
        "dtypes": dict(dtypes),
    }

expected = {
    "in_channels": 128,
    "out_channels": 128,
    "context_in_dim": 2560,
    "hidden_size": 3072,
    "num_heads": 24,
    "depth": 12,
    "packing": True,
}
observed = {key: config[key] for key in expected}
if observed != expected:
    raise RuntimeError(
        f"unexpected Mage-Flow-Base transformer geometry: {observed!r}"
    )
if layout["transformer"]["dtypes"] != {"BF16": 397}:
    raise RuntimeError(f"unexpected transformer layout: {layout['transformer']!r}")
if layout["vae"]["dtypes"] != {"BF16": 839}:
    raise RuntimeError(f"unexpected VAE layout: {layout['vae']!r}")

print(
    json.dumps(
        {
            "model": model_id,
            "revision": revision,
            "snapshot": str(root),
            "config": observed,
            "layout": layout,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
