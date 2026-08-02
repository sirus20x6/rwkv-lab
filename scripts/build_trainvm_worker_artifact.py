#!/usr/bin/env python3
"""Build the deterministic authority-launched rwkv_lab TrainVM zipapp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any


SCHEMA = "trainvm.python-worker-artifact/v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ENTRYPOINT = (
    b"from rwkv_lab.trainvm_adapters.entrypoint import main\n"
    b"raise SystemExit(main())\n"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _source_members(source_root: Path) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = [("__main__.py", ENTRYPOINT)]
    for package in ("rwkv_lab", "trainvm"):
        package_root = source_root / package
        if not package_root.is_dir():
            raise ValueError(f"source package is missing: {package_root}")
        for path in sorted(package_root.rglob("*.py")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"worker source must be a regular file: {path}")
            members.append((path.relative_to(source_root).as_posix(), path.read_bytes()))
    names = [name for name, _ in members]
    if len(names) != len(set(names)):
        raise ValueError("worker artifact contains duplicate archive paths")
    return members


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build(source_root: Path, output: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    output = output.absolute()
    members = _source_members(source_root)
    manifest = {
        "schema": SCHEMA,
        "members": [
            {"path": name, "size": len(data), "sha256": _digest(data)}
            for name, data in members
        ],
    }
    manifest_bytes = _canonical_bytes(manifest) + b"\n"
    archive_members = [
        *members,
        ("TRAINVM_WORKER_MANIFEST.json", manifest_bytes),
    ]
    output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
                for name, data in archive_members:
                    archive.writestr(_zip_info(name), data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            archive_bytes = stream.read()
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema": SCHEMA,
        "output": str(output),
        "archive_sha256": _digest(archive_bytes),
        "source_manifest_sha256": _digest(manifest_bytes),
        "member_count": len(archive_members),
        "size": len(archive_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source_root, arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
