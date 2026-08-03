#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
out="${COCO_ROOT:-datasets/coco2017}"
mkdir -p "$out"

fetch() {
  local url="$1" destination="$2"
  curl --fail --location --retry 8 --retry-all-errors \
    --continue-at - --output "$destination" "$url" || {
      # Some mirrors reject a range at exact EOF. A valid archive is already
      # complete, so test it before deciding that the transfer failed.
      unzip -tqq "$destination"
    }
  unzip -tqq "$destination"
}

fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  "$out/annotations_trainval2017.zip"
fetch http://images.cocodataset.org/zips/train2017.zip \
  "$out/train2017.zip"
fetch https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip \
  "$out/lvis_v1_train.json.zip"

echo "COCO 2017 images/masks and LVIS v1 open-vocabulary masks are complete in $out"
