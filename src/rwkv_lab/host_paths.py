"""Resolve host-specific artifact paths instead of baking one machine's in.

Model weights, conversion patches and token streams are not in the repository,
so the trainers and loaders that consume them used to carry the maintainer's
absolute ``/thearray/...`` paths as field defaults. Every config built on any
other host inherited a path the caller never chose and failed on it.

The policy, first applied to ``TerminalExpertTrainConfig.model_path`` and
recorded in the README under "Host-specific path defaults":

  1. an environment variable, for hosts that keep the artifact elsewhere;
  2. the historical path, but *only* when it actually exists on this host, so
     the maintainer's runs behave exactly as before;
  3. ``None`` — nothing is assumed, and validation fails naming the
     configuration field and the environment variable rather than echoing a
     bare absolute path the caller never typed.

Step 2 is existence-checked on purpose: an absolute default that is merely
absent should not be an error, because the caller never asked for it.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_host_path(
    env_var: str, historical: str, *, probe: str | None = None
) -> str | None:
    """Best-effort local path for an artifact, or None when this host has none.

    ``probe`` names the path whose existence stands in for the artifact's, for
    outputs that do not exist yet: an output directory is judged by the run
    root it would be created under, not by itself.
    """
    configured = os.environ.get(env_var)
    if configured:
        return configured
    if Path(probe if probe is not None else historical).expanduser().exists():
        return historical
    return None


def require_host_path(
    value: str | None, *, field: str, env_var: str, flag: str | None = None
) -> str:
    """Return ``value``, or raise naming the field that is not configured."""
    if value:
        return value
    how = f"Pass --{flag}" if flag else f"Set {field} explicitly"
    raise ValueError(
        f"{field} is not configured on this host. {how}, or set ${env_var} to "
        f"where it lives here. There is deliberately no built-in default: the "
        f"historical path belongs to one machine, and silently inheriting it "
        f"is how a run ends up reading something nobody chose."
    )
