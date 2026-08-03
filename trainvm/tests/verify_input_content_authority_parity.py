from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_input_content_authority_parity.py TRAINVM SOURCE_ROOT"
        )
    trainvm = Path(sys.argv[1]).resolve(strict=True)
    source_root = Path(sys.argv[2]).resolve(strict=True)
    sys.path.insert(0, str(source_root))
    from rwkv_lab.trainvm_adapters.content_authority import (
        measure_input_content_root,
    )

    with tempfile.TemporaryDirectory(
        prefix="trainvm-input-content-parity-"
    ) as raw:
        root = Path(raw) / "dataset"
        (root / "nested" / "empty").mkdir(parents=True)
        (root / "alpha.txt").write_bytes(b"alpha")
        (root / "café.txt").write_bytes(b"coffee")
        (root / "nested" / "zero.bin").write_bytes(b"")
        native = json.loads(
            subprocess.check_output(
                [str(trainvm), "inspect-input-content-root", str(root)],
                text=True,
            )
        )
        worker = asdict(measure_input_content_root(root))
        if native != worker:
            raise SystemExit(
                "native and Python input-content identities disagree: "
                + json.dumps({"native": native, "worker": worker}, sort_keys=True)
            )
    print("native/Python input content authority parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
