"""Small wrappers for safer torch serialization loads."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


UNSAFE_LOAD_ENV = "MOE_MLA_ALLOW_UNSAFE_TORCH_LOAD"


def safe_torch_load(path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """Load torch files without enabling pickle execution by default.

    Set MOE_MLA_ALLOW_UNSAFE_TORCH_LOAD=1 only for legacy checkpoints that you
    created and trust. That fallback uses pickle-backed loading.
    """
    kwargs.pop("weights_only", None)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, *args, weights_only=True, **kwargs)
    except Exception as exc:
        if os.environ.get(UNSAFE_LOAD_ENV) == "1":
            return torch.load(path, *args, weights_only=False, **kwargs)
        raise RuntimeError(
            f"Refusing unsafe torch.load for {path!s}. If this is a trusted "
            f"legacy checkpoint, rerun with {UNSAFE_LOAD_ENV}=1."
        ) from exc
