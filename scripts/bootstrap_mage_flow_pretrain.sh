#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAGE_SOURCE="${MAGE_FLOW_SOURCE:-$REPO_ROOT/.external/Mage}"
MAGE_VENV="${MAGE_FLOW_VENV:-$REPO_ROOT/.venv-mage-flow}"
MAGE_REVISION="${MAGE_FLOW_SOURCE_REVISION:-ef932e2cc3e94bb026d937a6cffae65492adc0fb}"
PYTHON_BIN="${MAGE_FLOW_PYTHON:-python3.11}"
CUDA_INDEX="${MAGE_FLOW_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
export HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"

if [[ ! -d "$MAGE_SOURCE/.git" ]]; then
  mkdir -p "$(dirname "$MAGE_SOURCE")"
  git clone https://github.com/microsoft/Mage.git "$MAGE_SOURCE"
fi
git -C "$MAGE_SOURCE" fetch origin "$MAGE_REVISION"
git -C "$MAGE_SOURCE" checkout --detach "$MAGE_REVISION"

if [[ ! -x "$MAGE_VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$MAGE_VENV"
fi
uv pip install --python "$MAGE_VENV/bin/python" \
  torch==2.13.0 torchvision==0.28.0 --index-url "$CUDA_INDEX"
uv pip install --python "$MAGE_VENV/bin/python" \
  -r "$MAGE_SOURCE/mage_flow/requirements.txt"
uv pip install --python "$MAGE_VENV/bin/python" setuptools wheel ninja "wandb>=0.19"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(
    "$MAGE_VENV/bin/python" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
  )"
fi
export TORCH_CUDA_ARCH_LIST
echo "Building CUDA extensions for compute capability $TORCH_CUDA_ARCH_LIST"
DS_BUILD_CPU_ADAM=1 DS_SKIP_CUDA_CHECK=1 uv pip install --python "$MAGE_VENV/bin/python" \
  --no-build-isolation deepspeed==0.19.3
MAX_JOBS="${MAX_JOBS:-16}" uv pip install --python "$MAGE_VENV/bin/python" \
  --no-build-isolation flash-attn==2.8.3
uv pip install --python "$MAGE_VENV/bin/python" \
  -e "$MAGE_SOURCE/mage_flow" --no-deps
uv pip install --python "$MAGE_VENV/bin/python" -e "$REPO_ROOT" --no-deps

if [[ "${MAGE_FLOW_DOWNLOAD_MODEL:-1}" == "1" ]]; then
  "$MAGE_VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="microsoft/Mage-Flow-Edit-Base",
    revision="8654a7bc0283ab2946385230b5b2eb944e0b76ea",
)
print({"model_snapshot": path})
PY
fi

"$MAGE_VENV/bin/python" - <<'PY'
import accelerate
import deepspeed
import diffusers
import flash_attn
import mage_flow
import torch
import transformers
from deepspeed.ops.adam import cpu_adam_op
from deepspeed.ops.op_builder import CPUAdamBuilder

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert CPUAdamBuilder().is_compatible()
assert cpu_adam_op is not None
print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "accelerate": accelerate.__version__,
        "deepspeed": deepspeed.__version__,
        "flash_attn": flash_attn.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY
