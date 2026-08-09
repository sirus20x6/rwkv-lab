from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .content_authority import (
    InputContentRootIdentity,
    verify_input_content_roots,
)

MAXIMUM_INLINE_CONFIG_BYTES = 40 * 1024
MAXIMUM_MANIFEST_LINE_BYTES = 4 * 1024 * 1024
MAXIMUM_MANIFEST_ROWS = 10_000_000
MAXIMUM_NODE_ID_BYTES = 128

# Probes a worker cannot satisfy and does not need. DeepSpeed's CUDA detection
# shells out to a compiler toolchain that is deliberately absent from the
# worker's filesystem view; left enabled it turns an optional capability check
# into a hard import failure. This is a library default, not authority: it is
# applied with setdefault so a sealed invocation can still say otherwise.
FIXED_LIBRARY_ENVIRONMENT_DEFAULTS = (("DS_IGNORE_CUDA_DETECTION", "1"),)


class AdapterInputError(ValueError):
    pass


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _absolute_normalized(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise AdapterInputError(f"{label} is not a path string")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise AdapterInputError(f"{label} must be an absolute normalized path")
    return path


def _workspace_roots(workspace: Mapping[str, Any], name: str) -> tuple[Path, ...]:
    values = workspace.get(name)
    if not isinstance(values, (list, tuple)) or not values:
        raise AdapterInputError(f"worker workspace has no {name}")
    roots: list[Path] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise AdapterInputError(f"workspace {name}[{index}] is not a path")
        path = _absolute_normalized(value, f"workspace {name}[{index}]")
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise AdapterInputError(
                f"workspace {name}[{index}] is unavailable"
            ) from error
        if not path.is_dir():
            raise AdapterInputError(f"workspace {name}[{index}] is not a directory")
        roots.append(path)
    return tuple(roots)


@dataclass(frozen=True, slots=True)
class WorkspacePathAuthority:
    """Resolve trainer paths only inside authority-declared workspace roots."""

    run_directory: Path
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    input_content_roots: tuple[InputContentRootIdentity, ...]

    @classmethod
    def from_workspace(
        cls, workspace: Mapping[str, Any], *, require_content: bool = False
    ) -> WorkspacePathAuthority:
        run_value = workspace.get("run_directory")
        if not isinstance(run_value, str):
            raise AdapterInputError("worker invocation has no run directory")
        read_roots = _workspace_roots(workspace, "allowed_read_roots")
        write_roots = _workspace_roots(workspace, "allowed_write_roots")
        raw_content_roots = workspace.get("input_content_roots")
        if raw_content_roots is None:
            if require_content:
                raise AdapterInputError(
                    "worker workspace has no verified input_content_roots"
                )
            input_content_roots: tuple[InputContentRootIdentity, ...] = ()
        else:
            try:
                input_content_roots = verify_input_content_roots(
                    list(raw_content_roots)
                    if isinstance(raw_content_roots, (list, tuple))
                    else raw_content_roots
                )
            except (OSError, ValueError) as error:
                raise AdapterInputError(
                    "worker input content identity verification failed"
                ) from error
            for identity in input_content_roots:
                content_path = Path(identity.path)
                if not _within(content_path, read_roots):
                    raise AdapterInputError(
                        "worker input content root is outside declared read roots"
                    )
        run_directory = cls._resolve_write_target(
            _absolute_normalized(run_value, "workspace run_directory"), write_roots
        )
        return cls(run_directory, read_roots, write_roots, input_content_roots)

    @staticmethod
    def _resolve_write_target(path: Path, roots: tuple[Path, ...]) -> Path:
        existing = path
        missing: list[str] = []
        while not existing.exists():
            if existing.parent == existing:
                raise AdapterInputError("write path has no existing ancestor")
            missing.append(existing.name)
            existing = existing.parent
        try:
            resolved = existing.resolve(strict=True)
        except OSError as error:
            raise AdapterInputError("write path ancestor is unavailable") from error
        if not resolved.is_dir():
            raise AdapterInputError("write path ancestor is not a directory")
        for part in reversed(missing):
            resolved /= part
        if not _within(resolved, roots):
            raise AdapterInputError("write path is outside declared write roots")
        return resolved

    def read_path(
        self,
        value: str,
        *,
        label: str,
        kind: str,
        require_content_identity: bool = True,
    ) -> Path:
        path = _absolute_normalized(value, label)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise AdapterInputError(f"{label} is unavailable") from error
        if not _within(resolved, self.read_roots + self.write_roots):
            raise AdapterInputError(f"{label} is outside declared read roots")
        if (
            require_content_identity
            and self.input_content_roots
            and not any(
                resolved == Path(identity.path)
                or (
                    identity.kind == "directory"
                    and Path(identity.path) in resolved.parents
                )
                for identity in self.input_content_roots
            )
        ):
            raise AdapterInputError(
                f"{label} is not covered by a verified input content root"
            )
        if kind == "file" and not resolved.is_file():
            raise AdapterInputError(f"{label} is not a file")
        if kind == "directory" and not resolved.is_dir():
            raise AdapterInputError(f"{label} is not a directory")
        if kind not in {"file", "directory"}:
            raise AdapterInputError("adapter requested an unknown path kind")
        return resolved

    def write_directory(self, value: str, *, label: str) -> Path:
        path = self._resolve_write_target(
            _absolute_normalized(value, label), self.write_roots
        )
        if path.exists() and not path.is_dir():
            raise AdapterInputError(f"{label} is not a directory")
        return path

    def exact_run_directory(self, configured: str) -> Path:
        resolved = self.write_directory(configured, label="trainer output directory")
        if resolved != self.run_directory:
            raise AdapterInputError(
                "trainer output directory disagrees with worker workspace authority"
            )
        return resolved

    def node_run_directory(self, node_id: object) -> Path:
        """Derive an isolated, retry-stable trainer directory from node authority."""
        if (
            not isinstance(node_id, str)
            or not node_id.isascii()
            or not 0 < len(node_id.encode("utf-8")) <= MAXIMUM_NODE_ID_BYTES
            or not node_id[0].isalnum()
            or any(
                not (character.isalnum() or character in "._-")
                for character in node_id
            )
        ):
            raise AdapterInputError("worker node ID cannot name a run directory")
        return self.write_directory(
            str(self.run_directory / "nodes" / node_id),
            label="node trainer output directory",
        )

    def verify_jsonl_file_references(
        self,
        manifest: Path,
        *,
        fields: tuple[str, ...],
        label: str,
    ) -> None:
        """Require every path referenced by a frozen JSONL to be content-bound."""

        rows = 0
        try:
            with manifest.open("rb") as stream:
                line_number = 0
                while encoded := stream.readline(MAXIMUM_MANIFEST_LINE_BYTES + 1):
                    line_number += 1
                    if len(encoded) > MAXIMUM_MANIFEST_LINE_BYTES:
                        raise AdapterInputError(
                            f"{label} line {line_number} exceeds its size bound"
                        )
                    if not encoded.strip():
                        continue
                    rows += 1
                    if rows > MAXIMUM_MANIFEST_ROWS:
                        raise AdapterInputError(f"{label} exceeds its row bound")
                    try:
                        row = json.loads(encoded)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise AdapterInputError(
                            f"{label} line {line_number} is not valid JSON"
                        ) from error
                    if not isinstance(row, dict):
                        raise AdapterInputError(
                            f"{label} line {line_number} is not an object"
                        )
                    values = [row.get(field) for field in fields if row.get(field)]
                    if len(values) != 1 or not isinstance(values[0], str):
                        raise AdapterInputError(
                            f"{label} line {line_number} has no unique image path"
                        )
                    referenced = Path(values[0])
                    if not referenced.is_absolute():
                        referenced = manifest.parent / referenced
                    self.read_path(
                        str(referenced),
                        label=f"{label} line {line_number} image",
                        kind="file",
                    )
        except OSError as error:
            raise AdapterInputError(f"{label} could not be read") from error
        if rows == 0:
            raise AdapterInputError(f"{label} is empty")

    def verify_json_relative_file_reference(
        self,
        directory: Path,
        *,
        manifest_name: str,
        field: str,
        label: str,
    ) -> Path:
        manifest = self.read_path(
            str(directory / manifest_name),
            label=f"{label} manifest",
            kind="file",
        )
        try:
            encoded = manifest.read_bytes()
            if not encoded or len(encoded) > MAXIMUM_MANIFEST_LINE_BYTES:
                raise AdapterInputError(f"{label} manifest exceeds its size bound")
            document = json.loads(encoded)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterInputError(f"{label} manifest is not valid JSON") from error
        value = document.get(field) if isinstance(document, dict) else None
        if not isinstance(value, str) or not value:
            raise AdapterInputError(f"{label} manifest {field} is invalid")
        referenced = (directory / value).resolve()
        if directory != referenced and directory not in referenced.parents:
            raise AdapterInputError(f"{label} manifest escapes its packed directory")
        return self.read_path(
            str(referenced),
            label=f"{label} manifest {field}",
            kind="file",
        )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise AdapterInputError("inline adapter config contains an unsupported value")


def read_inline_config(
    inputs: Mapping[str, Any], *, name: str = "config"
) -> dict[str, Any]:
    """Decode config already frozen into the content-addressed invocation.

    A path to a mutable JSON file is deliberately rejected: binding only its
    pathname would let its contents change after plan/submission locking.
    Larger configurations must arrive as a verified immutable artifact.
    """

    if set(inputs) != {name} or not isinstance(inputs.get(name), Mapping):
        raise AdapterInputError(
            f"adapter requires exactly one inline object input named {name!r}"
        )
    value = _thaw(inputs[name])
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdapterInputError("inline adapter config is not finite JSON") from error
    if not encoded or len(encoded) > MAXIMUM_INLINE_CONFIG_BYTES:
        raise AdapterInputError("inline adapter config exceeds its size bound")
    return value


def require_run_directory(configured: str, workspace: Mapping[str, Any]) -> Path:
    return WorkspacePathAuthority.from_workspace(workspace).exact_run_directory(
        configured
    )


def bind_worker_process_environment(
    workspace: Mapping[str, Any],
    *,
    environment: MutableMapping[str, str] | None = None,
) -> Path:
    """Bind the worker's process environment to its authorized run directory.

    hostd execs the worker with an empty environment on purpose -- that
    emptiness *is* the isolation, and `hostd_linux_stopped_launcher.cpp` passes
    a literal `{nullptr}` envp to make it unconditional. Libraries underneath
    the trainer nevertheless resolve a home directory for their per-run caches,
    and a worker without HOME dies after the model is already resident and
    before the first optimizer step, reporting only an exception type name.

    HOME is therefore derived from the run directory the controller already
    authorized for this invocation. It is deliberately not read from
    `/etc/passwd`, and deliberately not inherited from whatever launched hostd:
    either would hand the worker a writable location the plan never granted it,
    which is the thing the empty environment buys.

    A HOME that is already present and points outside that run directory is
    refused rather than quietly corrected, so that restoring the launcher's
    environment fails loudly instead of appearing to work.
    """

    environment = os.environ if environment is None else environment
    for name, value in FIXED_LIBRARY_ENVIRONMENT_DEFAULTS:
        environment.setdefault(name, value)
    authorized = WorkspacePathAuthority.from_workspace(workspace).run_directory
    # hostd provisions and chowns this directory to the worker principal before
    # the launch (provision_authorized_run_directory), so its absence here means
    # the worker is not standing in the workspace the plan authorized.
    if not authorized.is_dir():
        raise AdapterInputError(
            "authorized run directory is not a provisioned directory"
        )
    declared = environment.get("HOME")
    if declared is None:
        environment["HOME"] = str(authorized)
        return authorized
    home = _absolute_normalized(declared, "worker HOME")
    try:
        resolved = home.resolve(strict=True)
    except OSError as error:
        raise AdapterInputError("worker HOME is unavailable") from error
    if not _within(resolved, (authorized,)):
        raise AdapterInputError(
            "worker HOME is outside the authorized run directory"
        )
    return resolved
