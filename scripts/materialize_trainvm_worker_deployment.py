#!/usr/bin/env python3
"""Materialize one sealed rwkv_lab worker and its native TrainVM registries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from build_trainvm_worker_artifact import build

SCHEMA = "trainvm.rwkv-lab-worker-deployment-materialization/v2"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _document_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish(path: Path, data: bytes, *, replace: bool, mode: int = 0o600) -> None:
    if path.exists() and path.read_bytes() != data and not replace:
        raise FileExistsError(f"refusing to replace changed deployment output: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainvm", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("--trusted-root", action="append", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()

    trainvm = arguments.trainvm.resolve(strict=True)
    python_input = arguments.python.absolute()
    python_metadata = python_input.lstat()
    if stat.S_ISLNK(python_metadata.st_mode) or not stat.S_ISREG(
        python_metadata.st_mode
    ):
        raise ValueError(
            "--python must name a regular interpreter binary, not a venv symlink"
        )
    python = python_input.resolve(strict=True)
    source_root = arguments.source_root.resolve(strict=True)
    working_directory = arguments.working_directory.resolve(strict=True)
    trusted_roots = sorted(
        {str(path.resolve(strict=True)) for path in arguments.trusted_root}
    )
    output_directory = arguments.output_directory.absolute()
    output_directory.mkdir(mode=0o750, parents=True, exist_ok=True)
    output_directory = output_directory.resolve(strict=True)
    artifact = output_directory / "rwkv-lab-worker.pyz"
    runtime_closure = output_directory / "bootstrap-runtime-closure.json"

    with tempfile.TemporaryDirectory(
        prefix=".trainvm-worker-build-", dir=output_directory
    ) as raw:
        closure_builder = (
            Path(__file__).resolve().parent / "build_trainvm_runtime_closure.py"
        )
        temporary_closure = Path(raw) / runtime_closure.name
        runtime_receipt = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-I",
                    str(closure_builder),
                    "--output",
                    str(temporary_closure),
                ],
                text=True,
            )
        )
        temporary_artifact = Path(raw) / artifact.name
        artifact_receipt = build(
            source_root, temporary_artifact, temporary_closure
        )
        closure_bytes = temporary_closure.read_bytes()
        artifact_bytes = temporary_artifact.read_bytes()
        if (
            runtime_closure.exists()
            and runtime_closure.read_bytes() != closure_bytes
            and not arguments.replace
        ):
            raise FileExistsError(
                f"refusing to replace changed runtime closure: {runtime_closure}"
            )
        if artifact.exists() and artifact.read_bytes() != artifact_bytes and not arguments.replace:
            raise FileExistsError(
                f"refusing to replace changed deployment artifact: {artifact}"
            )
        _publish(runtime_closure, closure_bytes, replace=True)
        _publish(artifact, artifact_bytes, replace=True, mode=0o640)

    artifact_receipt["output"] = str(artifact)
    runtime_receipt["output"] = str(runtime_closure)
    python_digest = _digest(python)
    deployment = json.loads(
        subprocess.check_output(
            [
                str(trainvm),
                "inspect-rwkv-lab-deployment",
                str(artifact),
                artifact_receipt["archive_sha256"],
                runtime_receipt["closure_digest"],
                str(python),
                python_digest,
                str(working_directory),
                *trusted_roots,
            ],
            text=True,
        )
    )
    adapters_path = output_directory / "adapters.json"
    launches_path = output_directory / "host-launches.json"
    _publish(
        adapters_path,
        _document_bytes(deployment["adapter_registry"]),
        replace=arguments.replace,
    )
    _publish(
        launches_path,
        _document_bytes(deployment["host_launch_registry"]),
        replace=arguments.replace,
    )
    receipt = {
        "schema": SCHEMA,
        "worker_artifact": artifact_receipt,
        "runtime_closure": runtime_receipt,
        "python": {"path": str(python), "sha256": python_digest},
        "working_directory": str(working_directory),
        "trusted_roots": trusted_roots,
        "adapter_registry": {
            "path": str(adapters_path),
            "sha256": _digest(adapters_path),
            "registry_digest": deployment["adapter_registry_digest"],
        },
        "host_launch_registry": {
            "path": str(launches_path),
            "sha256": _digest(launches_path),
            "registry_digest": deployment["host_launch_registry_digest"],
        },
        "provided_capabilities": deployment["provided_capabilities"],
    }
    receipt_path = output_directory / "deployment-receipt.json"
    _publish(receipt_path, _document_bytes(receipt), replace=arguments.replace)
    directory_descriptor = os.open(output_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
