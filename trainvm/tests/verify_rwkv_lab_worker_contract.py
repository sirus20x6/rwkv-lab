#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

from rwkv_lab.trainvm_adapters.handlers import supported_adapter_keys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_rwkv_lab_worker_contract.py TRAINVM")
    fingerprint = "sha256:" + "a" * 64
    document = json.loads(
        subprocess.check_output(
            [sys.argv[1], "inspect-rwkv-lab-worker", fingerprint],
            text=True,
        )
    )
    native = {
        (
            profile["key"]["adapter"],
            profile["key"]["version"],
            profile["key"]["operation"],
            profile["key"]["contract"],
        )
        for profile in document["adapter_registry"]["profiles"]
    }
    python = set(supported_adapter_keys())
    if native != python:
        raise SystemExit(
            f"native/worker adapter key drift: native={sorted(native)!r} "
            f"python={sorted(python)!r}"
        )
    capabilities = document["provided_capabilities"]
    if capabilities != sorted(set(capabilities)):
        raise SystemExit("native worker capability output is not canonical")
    print("native/Python rwkv_lab worker contract parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
