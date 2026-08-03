#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

# Pin the generator major below protobuf 7 because the optional SM120 CUTLASS
# stack currently requires protobuf <7. Runtime code generation is forbidden;
# these deterministic bindings are committed with the protocol source.
uvx --from grpcio-tools==1.76.0 \
  python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=src \
  --grpc_python_out=src \
  proto/trainvm/v1/trainvm.proto
