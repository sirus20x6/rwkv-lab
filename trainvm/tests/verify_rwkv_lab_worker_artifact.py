#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: verify_rwkv_lab_worker_artifact.py BUILDER MATERIALIZER TRAINVM SOURCE_ROOT"
        )
    builder = Path(sys.argv[1]).resolve(strict=True)
    materializer = Path(sys.argv[2]).resolve(strict=True)
    trainvm = Path(sys.argv[3]).resolve(strict=True)
    source_root = Path(sys.argv[4]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="trainvm-worker-artifact-") as raw:
        directory = Path(raw)
        first = directory / "first.pyz"
        second = directory / "second.pyz"
        receipts = []
        for output in (first, second):
            receipts.append(
                json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            str(builder),
                            "--source-root",
                            str(source_root),
                            "--output",
                            str(output),
                        ],
                        text=True,
                    )
                )
            )
        if first.read_bytes() != second.read_bytes():
            raise SystemExit("worker artifact build is not byte deterministic")
        if receipts[0]["archive_sha256"] != digest(first) or receipts[0][
            "archive_sha256"
        ] != receipts[1]["archive_sha256"]:
            raise SystemExit("worker artifact receipt does not bind exact bytes")

        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            if (
                names.count("__main__.py") != 1
                or names.count("TRAINVM_WORKER_MANIFEST.json") != 1
                or any("__pycache__" in name or name.endswith(".pyc") for name in names)
            ):
                raise SystemExit("worker artifact member surface is not closed")
            manifest = json.loads(archive.read("TRAINVM_WORKER_MANIFEST.json"))
            expected = {
                item["path"]: (item["size"], item["sha256"])
                for item in manifest["members"]
            }
            for name, (size, sha256) in expected.items():
                data = archive.read(name)
                if len(data) != size or "sha256:" + hashlib.sha256(data).hexdigest() != sha256:
                    raise SystemExit(f"worker member manifest mismatch: {name}")

        loaded = subprocess.run(
            [sys.executable, "-I", str(first), "--artifact-load-test"],
            env={},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if loaded.returncode == 0 or (
            "authority worker accepts only its fixed bootstrap descriptor"
            not in loaded.stderr
        ):
            raise SystemExit(
                "isolated worker artifact did not reach its closed argv boundary: "
                + loaded.stderr[-1000:]
            )

        python_path = Path(sys.executable).resolve(strict=True)
        python_root = Path(python_path.anchor) / "usr"
        if not python_root.is_dir() or python_root not in python_path.parents:
            python_root = python_path.parent
        deployment = json.loads(
            subprocess.check_output(
                [
                    str(trainvm),
                    "inspect-rwkv-lab-deployment",
                    str(first),
                    digest(first),
                    str(python_path),
                    digest(python_path),
                    str(directory),
                    str(directory),
                    str(python_root),
                ],
                text=True,
            )
        )
        profiles = deployment["host_launch_registry"]["profiles"]
        if len(profiles) != 4 or any(
            profile["code_argument_index"] != 1
            or profile["public_arguments"] != ["-I", "rwkv-lab-worker.pyz"]
            or profile["code_fingerprint"] != digest(first)
            for profile in profiles
        ):
            raise SystemExit("deployment registry drifted from sealed worker artifact")

        deployment_directory = directory / "deployment"
        materialize = [
            sys.executable,
            str(materializer),
            "--trainvm",
            str(trainvm),
            "--python",
            str(python_path),
            "--source-root",
            str(source_root),
            "--output-directory",
            str(deployment_directory),
            "--working-directory",
            str(directory),
            "--trusted-root",
            str(directory),
            "--trusted-root",
            str(python_root),
        ]
        first_materialization = json.loads(
            subprocess.check_output(materialize, text=True)
        )
        second_materialization = json.loads(
            subprocess.check_output(materialize, text=True)
        )
        if first_materialization != second_materialization:
            raise SystemExit("worker deployment materialization is not idempotent")
        for name in (
            "rwkv-lab-worker.pyz",
            "adapters.json",
            "host-launches.json",
            "deployment-receipt.json",
        ):
            if not (deployment_directory / name).is_file():
                raise SystemExit(f"worker deployment omitted {name}")
    print("deterministic isolated rwkv_lab worker artifact passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
