from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAXIMUM_INLINE_CONFIG_BYTES = 40 * 1024


class AdapterInputError(ValueError):
    pass


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
    declared = workspace.get("run_directory")
    if not isinstance(declared, str):
        raise AdapterInputError("worker invocation has no run directory")
    configured_path = Path(configured)
    declared_path = Path(declared)
    if (
        not configured_path.is_absolute()
        or not declared_path.is_absolute()
        or configured_path != Path(os.path.normpath(configured_path))
        or declared_path != Path(os.path.normpath(declared_path))
        or configured_path != declared_path
    ):
        raise AdapterInputError(
            "trainer output directory disagrees with worker workspace authority"
        )
    return declared_path
