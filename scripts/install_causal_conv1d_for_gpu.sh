#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${CAPTION_CPT_VENV:-$ROOT/.venv-caption-cpt}"
VERSION="${CAUSAL_CONV1D_VERSION:-1.6.2.post1}"
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
export UV_NO_PROGRESS=1

if [[ "$VERSION" != "1.6.2.post1" ]]; then
  echo "The local SM-only patch is pinned to causal-conv1d 1.6.2.post1" >&2
  exit 2
fi

if [[ -z "${CAUSAL_CONV1D_CUDA_ARCH:-}" ]]; then
  CAUSAL_CONV1D_CUDA_ARCH="$(
    "$VENV/bin/python" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}{minor}")
PY
  )"
fi
export CAUSAL_CONV1D_CUDA_ARCH

if "$VENV/bin/python" -c \
  'import causal_conv1d, sys; sys.exit(causal_conv1d.__version__ != sys.argv[1])' \
  "$VERSION" 2>/dev/null; then
  echo "causal-conv1d $VERSION is already installed"
  exit 0
fi

uv pip install --python "$VENV/bin/python" pip
"$VENV/bin/python" -m pip download \
  --no-deps \
  --no-binary=:all: \
  --no-build-isolation \
  --dest "$BUILD_ROOT" \
  "causal-conv1d==$VERSION"
tar -xf "$BUILD_ROOT/causal_conv1d-$VERSION.tar.gz" -C "$BUILD_ROOT"
patch -d "$BUILD_ROOT/causal_conv1d-$VERSION" -p1 \
  < "$ROOT/patches/causal-conv1d-1.6.2-sm-override.patch"

echo "Building causal-conv1d $VERSION for SM$CAUSAL_CONV1D_CUDA_ARCH only"
MAX_JOBS="${MAX_JOBS:-16}" uv pip install \
  --python "$VENV/bin/python" \
  --no-build-isolation \
  "$BUILD_ROOT/causal_conv1d-$VERSION"
