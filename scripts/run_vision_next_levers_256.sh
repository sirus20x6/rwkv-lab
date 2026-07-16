#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VISION_NEXT_PREFIX_TOKENS="${VISION_NEXT_PREFIX_TOKENS:-256}"
export VISION_NEXT_RUN="${VISION_NEXT_RUN:-runs/moonvit_rwkv_next_levers_so400m_256}"
export VISION_NEXT_BATCH_TOKENS="${VISION_NEXT_BATCH_TOKENS:-4096}"
exec scripts/run_vision_next_levers.sh
