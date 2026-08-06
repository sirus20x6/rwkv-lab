#!/usr/bin/env python3
"""Build the deterministic pre-dispatch runtime closure for TrainVM workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import sysconfig
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

SCHEMA = "trainvm.python-bootstrap-runtime-closure/v2"
DEFAULT_ROOT_DISTRIBUTIONS = (
    "grpcio",
    "pillow",
    "protobuf",
    "torch",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    path = Path(os.path.abspath(path))
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return {
            "kind": "symlink",
            "mode": mode,
            "path": str(path),
            "target": os.readlink(path),
        }
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"runtime closure path is group/world writable: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"runtime closure path is not regular or symlink: {path}")
    return {
        "kind": "regular",
        "mode": mode,
        "path": str(path),
        "sha256": _sha256(path),
        "size": metadata.st_size,
    }


def _add_path(paths: dict[str, dict[str, Any]], path: Path) -> None:
    path = Path(os.path.abspath(path))
    if path.is_dir():
        return
    entry = _entry(path)
    previous = paths.setdefault(str(path), entry)
    if previous != entry:
        raise ValueError(f"runtime closure path changed while scanning: {path}")
    if entry["kind"] == "symlink":
        target = path.resolve(strict=True)
        _add_path(paths, target)


def _distribution_closure(
    root_distributions: tuple[str, ...],
) -> list[importlib.metadata.Distribution]:
    pending = deque(root_distributions)
    selected: dict[str, importlib.metadata.Distribution] = {}
    while pending:
        requested = pending.popleft()
        if requested in selected:
            continue
        distribution = importlib.metadata.distribution(requested)
        name = canonicalize_name(distribution.metadata["Name"])
        selected[name] = distribution
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in selected:
                pending.append(dependency)
    return [selected[name] for name in sorted(selected)]


def _stdlib_files() -> list[Path]:
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    platlib = Path(sysconfig.get_paths()["platlib"]).resolve(strict=True)
    excluded = {purelib, platlib}
    files: list[Path] = []
    for root, directories, names in os.walk(stdlib, followlinks=False):
        root_path = Path(root)
        directories[:] = sorted(
            directory
            for directory in directories
            if root_path / directory not in excluded
            and directory not in {"__pycache__", "site-packages", "dist-packages"}
        )
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = root_path / name
            if path.is_file() or path.is_symlink():
                files.append(path)
    library = sysconfig.get_config_var("LDLIBRARY")
    library_directory = sysconfig.get_config_var("LIBDIR")
    if library and library_directory:
        candidate = Path(library_directory) / library
        if candidate.exists():
            files.append(candidate)
    if sys.prefix != sys.base_prefix:
        configuration = Path(sys.prefix) / "pyvenv.cfg"
        if configuration.is_file():
            files.append(configuration)
    return files


def build(root_distributions: tuple[str, ...]) -> dict[str, Any]:
    paths: dict[str, dict[str, Any]] = {}
    distributions = _distribution_closure(root_distributions)
    for path in _stdlib_files():
        _add_path(paths, path)
    identities = []
    for distribution in distributions:
        name = canonicalize_name(distribution.metadata["Name"])
        identities.append({"name": name, "version": distribution.version})
        files = distribution.files
        if files is None:
            raise ValueError(f"distribution has no installed file inventory: {name}")
        for relative in files:
            path = Path(distribution.locate_file(relative))
            if path.exists() or path.is_symlink():
                _add_path(paths, path)
    body = {
        "api_version": SCHEMA,
        "distributions": identities,
        "files": [paths[name] for name in sorted(paths)],
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "implementation": sys.implementation.name,
            "platform": sysconfig.get_platform(),
            "prefix": sys.prefix,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "root_distributions": list(root_distributions),
    }
    return {**body, "closure_digest": _digest(_canonical(body))}


def _publish(output: Path, data: bytes) -> None:
    output = output.absolute()
    output.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--distribution",
        action="append",
        default=[],
        help="root installed distribution to include recursively (repeatable)",
    )
    arguments = parser.parse_args()
    requested = arguments.distribution or list(DEFAULT_ROOT_DISTRIBUTIONS)
    root_distributions = tuple(
        sorted({canonicalize_name(name) for name in requested})
    )
    if not root_distributions or any(not name for name in root_distributions):
        raise ValueError("runtime closure roots must be nonempty distributions")
    document = build(root_distributions)
    data = _canonical(document) + b"\n"
    _publish(arguments.output, data)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(arguments.output.absolute()),
                "closure_digest": document["closure_digest"],
                "manifest_sha256": _digest(data),
                "root_distributions": list(root_distributions),
                "distribution_count": len(document["distributions"]),
                "file_count": len(document["files"]),
                "total_bytes": sum(
                    entry.get("size", 0) for entry in document["files"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
