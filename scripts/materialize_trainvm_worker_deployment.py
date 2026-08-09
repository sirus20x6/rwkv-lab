#!/usr/bin/env python3
"""Materialize sealed per-adapter rwkv_lab runtimes and TrainVM registries."""

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

SCHEMA = "trainvm.rwkv-lab-worker-deployment-materialization/v3"
REQUIREMENTS_SCHEMA = "trainvm.rwkv-lab-worker-runtime-requirements/v1"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _document_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _runtime_argument(value: str) -> tuple[str, Path]:
    adapter, separator, path = value.partition("=")
    if not separator or not adapter or not path:
        raise argparse.ArgumentTypeError(
            "runtime must be REGISTERED_ADAPTER=/absolute/copied/python"
        )
    return adapter, Path(path)


def _runtime_requirements(
    trainvm: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    document = json.loads(
        subprocess.check_output(
            [str(trainvm), "inspect-rwkv-lab-runtime-requirements"], text=True
        )
    )
    if set(document) != {
        "api_version",
        "profiles",
        "shared_root_distributions",
    } or document.get("api_version") != REQUIREMENTS_SCHEMA:
        raise ValueError("TrainVM returned an unsupported runtime requirements contract")
    shared = tuple(document["shared_root_distributions"])
    if not shared or tuple(sorted(set(shared))) != shared:
        raise ValueError("shared runtime distributions are not canonical")
    profiles: dict[str, tuple[str, ...]] = {}
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("runtime requirements must contain adapter profiles")
    for profile in raw_profiles:
        if not isinstance(profile, dict) or set(profile) != {
            "adapter",
            "root_distributions",
        }:
            raise ValueError("runtime requirement profile has an invalid shape")
        adapter = profile["adapter"]
        distributions = tuple(profile["root_distributions"])
        if (
            not isinstance(adapter, str)
            or not adapter
            or adapter in profiles
            or not distributions
            or tuple(sorted(set(distributions))) != distributions
            or not set(shared).issubset(distributions)
        ):
            raise ValueError("runtime requirement profile is not canonical")
        profiles[adapter] = distributions
    return tuple(profiles), shared, profiles


def _regular_python(path: Path) -> Path:
    path = path.absolute()
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            "runtime Python must be a regular interpreter binary, not a venv symlink"
        )
    return path.resolve(strict=True)


def _runtime_groups(
    adapters: tuple[str, ...],
    adapter_pythons: dict[str, Path],
    adapter_distributions: dict[str, tuple[str, ...]],
) -> dict[tuple[str, tuple[str, ...]], list[str]]:
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for adapter in adapters:
        key = (
            str(adapter_pythons[adapter]),
            tuple(sorted(adapter_distributions[adapter])),
        )
        groups.setdefault(key, []).append(adapter)
    return groups


# How many differing entries a disagreement report names before it stops. The
# report is a diagnosis, not an inventory: a system upgrade moves thousands of
# paths at once, and the first handful identify the cause just as well as all of
# them while keeping the failure readable.
DISAGREEMENT_EXAMPLES = 8


def _by_path(entries: Any) -> dict[str, Any]:
    if not isinstance(entries, list):
        return {}
    return {
        entry["path"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }


def _section_disagreement(name: str, before: Any, after: Any) -> list[str]:
    previous, current = _by_path(before), _by_path(after)
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    changed = sorted(
        path for path in set(previous) & set(current)
        if previous[path] != current[path]
    )
    if not (added or removed or changed):
        return []
    lines = [
        f"  {name}: {len(added)} added, {len(removed)} removed, "
        f"{len(changed)} changed"
    ]
    for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
        for path in paths[:DISAGREEMENT_EXAMPLES]:
            lines.append(f"    {label} {path}")
        if len(paths) > DISAGREEMENT_EXAMPLES:
            lines.append(
                f"    ... and {len(paths) - DISAGREEMENT_EXAMPLES} more {label}"
            )
    return lines


def _closure_disagreement(previous: bytes, current: bytes) -> str:
    """Say which part of the scanned tree moved between two closure computations.

    Both documents describe the same host, computed minutes apart by the same
    interpreter over the same distributions, so a disagreement is a statement
    about the machine rather than about this script: something under the scanned
    roots changed while the deployment was being materialized.

    That is worth naming precisely, because the bare refusal reads like a bug in
    the materializer and sends the next reader looking for nondeterminism in the
    digest. It was checked, on this host, over the real 19,969-file tree: three
    consecutive closure builds and three consecutive materializations into one
    directory produced byte-identical documents, and the whole native suite
    passed. The one observed failure coincided with a `pacman -Syu` that rewrote
    2,373 files under the scanned site-packages and relinked libraries the ELF
    graph pins. The closure was right both times; it described two different
    trees.
    """

    try:
        before = json.loads(previous)
        after = json.loads(current)
    except ValueError:
        return (
            "  the two closure documents are not both readable JSON "
            f"({len(previous)} bytes then {len(current)} bytes)"
        )
    if not isinstance(before, dict) or not isinstance(after, dict):
        return "  the two closure documents are not both JSON objects"
    lines: list[str] = []
    if before.get("api_version") != after.get("api_version"):
        lines.append(
            f"  api_version: {before.get('api_version')} -> "
            f"{after.get('api_version')}"
        )
    if before.get("distributions") != after.get("distributions"):
        lines.append("  distributions: the installed version set moved")
    before_native = before.get("native")
    after_native = after.get("native")
    if isinstance(before_native, dict) and isinstance(after_native, dict):
        if before_native.get("cuda") != after_native.get("cuda"):
            lines.append(
                f"  native.cuda: {before_native.get('cuda')} -> "
                f"{after_native.get('cuda')}"
            )
        lines.extend(
            _section_disagreement(
                "native.objects",
                before_native.get("objects"),
                after_native.get("objects"),
            )
        )
        before_registry = before_native.get("kernel_registry")
        after_registry = after_native.get("kernel_registry")
        if isinstance(before_registry, dict) and isinstance(after_registry, dict):
            lines.extend(
                _section_disagreement(
                    "native.kernel_registry.extensions",
                    before_registry.get("extensions"),
                    after_registry.get("extensions"),
                )
            )
    lines.extend(
        _section_disagreement("files", before.get("files"), after.get("files"))
    )
    if not lines:
        lines.append("  the documents differ outside the sections compared here")
    return "\n".join(
        [
            "the scanned runtime tree moved between the two closure "
            "computations in this run,",
            "so the second scan did not seal the tree the first one sealed. "
            "A package",
            "upgrade or an install landing under the scanned roots mid-run "
            "produces this.",
            *lines,
        ]
    )


def _changed_closure_message(path: Path, previous: bytes, current: bytes) -> str:
    """The whole refusal: what was refused, then why the two scans disagreed."""

    return (
        f"refusing to replace changed runtime closure: {path}\n"
        + _closure_disagreement(previous, current)
    )


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
    parser.add_argument("--python", type=Path)
    parser.add_argument(
        "--adapter-python",
        action="append",
        default=[],
        type=_runtime_argument,
        metavar="ADAPTER=PYTHON",
        help="bind one registered adapter to its copied venv interpreter",
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("--trusted-root", action="append", required=True, type=Path)
    parser.add_argument(
        "--runtime-digest-cache",
        type=Path,
        help=(
            "owner-only persistent inode digest cache shared across runtime "
            "groups and repeated materializations"
        ),
    )
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()

    trainvm = arguments.trainvm.resolve(strict=True)
    adapters, shared_distributions, adapter_distributions = (
        _runtime_requirements(trainvm)
    )
    if (arguments.python is None) == (not arguments.adapter_python):
        raise ValueError(
            "select exactly one of --python or four --adapter-python mappings"
        )
    if arguments.python is not None:
        python = _regular_python(arguments.python)
        adapter_pythons = {adapter: python for adapter in adapters}
        adapter_distributions = {
            adapter: shared_distributions for adapter in adapters
        }
    else:
        adapter_pythons = {}
        for adapter, path in arguments.adapter_python:
            if adapter in adapter_pythons:
                raise ValueError(f"duplicate --adapter-python mapping: {adapter}")
            adapter_pythons[adapter] = _regular_python(path)
        unknown = sorted(set(adapter_pythons).difference(adapters))
        missing = sorted(set(adapters).difference(adapter_pythons))
        if unknown or missing:
            raise ValueError(
                "--adapter-python must bind every registered adapter; missing: "
                + ", ".join(missing)
                + "; unknown: "
                + ", ".join(unknown)
            )
    source_root = arguments.source_root.resolve(strict=True)
    working_directory = arguments.working_directory.resolve(strict=True)
    trusted_roots = sorted(
        {str(path.resolve(strict=True)) for path in arguments.trusted_root}
    )
    output_directory = arguments.output_directory.absolute()
    output_directory.mkdir(mode=0o750, parents=True, exist_ok=True)
    output_directory = output_directory.resolve(strict=True)
    group_members = _runtime_groups(
        adapters, adapter_pythons, adapter_distributions
    )
    runtime_groups = []
    runtime_specs = []
    closure_builder = (
        Path(__file__).resolve().parent / "build_trainvm_runtime_closure.py"
    )
    with tempfile.TemporaryDirectory(
        prefix=".trainvm-worker-build-", dir=output_directory
    ) as raw:
        temporary_directory = Path(raw)
        # Without --runtime-digest-cache the cache still exists, scoped to this
        # build: several runtime groups share a stdlib and most of a package
        # closure, so the second group onwards is already a hit. The flag only
        # changes how long it survives.
        digest_cache = (
            arguments.runtime_digest_cache.absolute()
            if arguments.runtime_digest_cache is not None
            else temporary_directory / "runtime-digests.json"
        )
        for (python_path, distributions), adapters in sorted(
            group_members.items(), key=lambda item: item[1]
        ):
            python = Path(python_path)
            group_identity = hashlib.sha256(
                _document_bytes(
                    {"adapters": adapters, "distributions": distributions}
                )
            ).hexdigest()[:16]
            artifact = output_directory / f"rwkv-lab-worker-{group_identity}.pyz"
            runtime_closure = (
                output_directory
                / f"bootstrap-runtime-closure-{group_identity}.json"
            )
            temporary_closure = temporary_directory / runtime_closure.name
            closure_command = [
                str(python),
                "-I",
                str(closure_builder),
                "--output",
                str(temporary_closure),
            ]
            closure_command.extend(("--digest-cache", str(digest_cache)))
            for distribution in distributions:
                closure_command.extend(("--distribution", distribution))
            runtime_receipt = json.loads(
                subprocess.check_output(closure_command, text=True)
            )
            temporary_artifact = temporary_directory / artifact.name
            artifact_receipt = build(
                source_root, temporary_artifact, temporary_closure
            )
            closure_bytes = temporary_closure.read_bytes()
            artifact_bytes = temporary_artifact.read_bytes()
            previous_closure = (
                runtime_closure.read_bytes() if runtime_closure.exists() else None
            )
            if (
                previous_closure is not None
                and previous_closure != closure_bytes
                and not arguments.replace
            ):
                raise FileExistsError(
                    _changed_closure_message(
                        runtime_closure, previous_closure, closure_bytes
                    )
                )
            if (
                artifact.exists()
                and artifact.read_bytes() != artifact_bytes
                and not arguments.replace
            ):
                raise FileExistsError(
                    f"refusing to replace changed deployment artifact: {artifact}"
                )
            _publish(runtime_closure, closure_bytes, replace=True)
            _publish(artifact, artifact_bytes, replace=True, mode=0o640)
            artifact_receipt["output"] = str(artifact)
            runtime_receipt["output"] = str(runtime_closure)
            python_digest = _digest(python)
            runtime_groups.append(
                {
                    "adapters": adapters,
                    "worker_artifact": artifact_receipt,
                    "runtime_closure": runtime_receipt,
                    "python": {"path": str(python), "sha256": python_digest},
                }
            )
            for adapter in adapters:
                runtime_specs.append(
                    {
                        "adapter": adapter,
                        "code_path": str(artifact),
                        "code_fingerprint": artifact_receipt["archive_sha256"],
                        "bootstrap_runtime_closure_fingerprint": runtime_receipt[
                            "closure_digest"
                        ],
                        "executable_path": str(python),
                        "executable_fingerprint": python_digest,
                        "working_directory": str(working_directory),
                    }
                )
    runtime_specs.sort(key=lambda runtime: runtime["adapter"])
    deployment_input = {
        "api_version": "trainvm.rwkv-lab-worker-runtimes/v1",
        "runtimes": runtime_specs,
        "trusted_roots": trusted_roots,
    }
    deployment = json.loads(
        subprocess.check_output(
            [str(trainvm), "inspect-rwkv-lab-deployment"],
            input=json.dumps(deployment_input, separators=(",", ":")),
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
        "runtime_groups": runtime_groups,
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
