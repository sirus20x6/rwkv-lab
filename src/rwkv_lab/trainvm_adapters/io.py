from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAXIMUM_INLINE_CONFIG_BYTES = 40 * 1024


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
            raise AdapterInputError(f"workspace {name}[{index}] is unavailable") from error
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

    @classmethod
    def from_workspace(cls, workspace: Mapping[str, Any]) -> WorkspacePathAuthority:
        run_value = workspace.get("run_directory")
        if not isinstance(run_value, str):
            raise AdapterInputError("worker invocation has no run directory")
        read_roots = _workspace_roots(workspace, "allowed_read_roots")
        write_roots = _workspace_roots(workspace, "allowed_write_roots")
        run_directory = cls._resolve_write_target(
            _absolute_normalized(run_value, "workspace run_directory"), write_roots
        )
        return cls(run_directory, read_roots, write_roots)

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

    def read_path(self, value: str, *, label: str, kind: str) -> Path:
        path = _absolute_normalized(value, label)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise AdapterInputError(f"{label} is unavailable") from error
        if not _within(resolved, self.read_roots + self.write_roots):
            raise AdapterInputError(f"{label} is outside declared read roots")
        if kind == "file" and not resolved.is_file():
            raise AdapterInputError(f"{label} is not a file")
        if kind == "directory" and not resolved.is_dir():
            raise AdapterInputError(f"{label} is not a directory")
        if kind not in {"file", "directory"}:
            raise AdapterInputError("adapter requested an unknown path kind")
        return resolved

    def write_directory(self, value: str, *, label: str) -> Path:
        path = self._resolve_write_target(_absolute_normalized(value, label), self.write_roots)
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
