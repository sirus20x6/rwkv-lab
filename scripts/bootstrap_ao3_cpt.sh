#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${AO3_CPT_VENV:-$ROOT/.venv-ao3-cpt}"
PYTHON_BIN="${AO3_CPT_PYTHON:-python3.11}"
CUDA_INDEX="${AO3_CPT_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
export UV_NO_PROGRESS=1

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$VENV"
fi
uv pip install --python "$VENV/bin/python" \
  torch==2.13.0 --index-url "$CUDA_INDEX"
uv pip install --python "$VENV/bin/python" \
  "transformers==5.5.0" "accelerate==1.13.0" "bitsandbytes>=0.49.2" \
  "peft>=0.18.1" "safetensors>=0.8" "tokenizers>=0.22" \
  "zstandard>=0.25" "numpy>=2.0" "psutil>=6" "ninja>=1.13" \
  "packaging>=25" "einops>=0.8"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(
    "$VENV/bin/python" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
  )"
fi
export TORCH_CUDA_ARCH_LIST
echo "Building CUDA extensions for compute capability $TORCH_CUDA_ARCH_LIST"
MAX_JOBS="${MAX_JOBS:-16}" uv pip install --python "$VENV/bin/python" \
  --no-build-isolation "flash-attn==2.8.3"
AO3_CPT_VENV="$VENV" "$ROOT/scripts/install_causal_conv1d_for_gpu.sh"
uv pip install --python "$VENV/bin/python" "flash-linear-attention>=0.4.1"
uv pip install --python "$VENV/bin/python" -e "$ROOT" --no-deps

"$VENV/bin/python" - <<'PY'
import bitsandbytes
import flash_attn
import fla
import peft
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] >= 10
print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "flash_attn": flash_attn.__version__,
        "fla": fla.__version__,
    }
)
PY
