#!/usr/bin/env python3
"""Run a command with accelerator discovery disabled before the process starts.

Pytest selects on markers *after* it has imported the test modules, so a
`-m "not gpu"` run still executes every module-level expression in every test
file it collects.  Eight of those expressions in this repository are
``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``, and the argument to
``skipif`` is evaluated when the decorator is applied -- at import.  A test that
is never selected has therefore already initialized the driver, and on a
workstation whose GPU also drives the display that is a user-visible cost.

Deselection cannot fix this, and neither can a fixture: both run too late.  The
only thing that is reliably early enough is the environment the process starts
with, which is why this is a launcher rather than a plugin.  The root
``tests/conftest.py`` applies the same mask for direct ``pytest`` invocations --
conftest import precedes test-module import, so it is early enough for anything
that starts in Python, but it cannot help a native binary or a subprocess tree.
The two cover disjoint failure modes and are both wanted.

Verify the effect physically rather than by asking the library that was masked::

    python scripts/non_gpu_environment.py python -c \\
      "import torch; torch.cuda.is_available()"

and confirm no descriptor in ``/proc/self/fd`` resolves to ``/dev/nvidia*`` or
``/dev/dri/*``.  ``open_accelerator_device_files`` below is that check, and
``tests/test_non_gpu_environment.py`` asserts on it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


ACCELERATOR_ACCESS_ENV = "TRAINVM_TEST_ACCELERATOR_ACCESS"

# Keep these explicit rather than inheriting an operator's shell.  An empty
# ordinal list is the documented "no devices" value for the CUDA and HIP
# runtimes; `-1` is a widely copied folk remedy with worse-defined behaviour, so
# it is deliberately not used here.  NVIDIA's container runtime reads `void` as
# "inject neither devices nor driver capabilities".  JAX and the portable
# runtimes are pinned to their CPU implementations by their own selectors,
# because they do not consult the CUDA ordinal list.
NON_GPU_ENVIRONMENT: Mapping[str, str] = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "HSA_VISIBLE_DEVICES": "",
    "GPU_DEVICE_ORDINAL": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_PLATFORM_NAME": "cpu",
    "ONEAPI_DEVICE_SELECTOR": "*:cpu",
    "SYCL_DEVICE_FILTER": "cpu",
}

# What an initialized driver leaves behind: the control device, the unified
# memory device, per-GPU character devices, and the DRM render nodes.
ACCELERATOR_DEVICE_PREFIXES = ("/dev/nvidia", "/dev/dri/", "/dev/kfd")


def accelerator_access_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return true only for the one documented, explicit test opt-in."""

    source = os.environ if environment is None else environment
    return source.get(ACCELERATOR_ACCESS_ENV) == "1"


def mask_accelerators(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child environment that cannot discover an accelerator."""

    masked = dict(os.environ if environment is None else environment)
    masked.update(NON_GPU_ENVIRONMENT)
    masked[ACCELERATOR_ACCESS_ENV] = "0"
    return masked


def open_accelerator_device_files(pid: str = "self") -> list[str]:
    """Return the accelerator device files this process currently has open.

    This is the physical statement of "no accelerator access", and it is the
    one worth asserting on.  ``torch.cuda.is_available() is False`` is necessary
    but weak: it is also False when torch was built without CUDA, when the
    driver is a different version than the runtime, and in several other cases
    that have nothing to do with whether this mask worked.  An open descriptor
    to ``/dev/nvidia0`` cannot be explained away.
    """

    descriptors = f"/proc/{pid}/fd"
    if not os.path.isdir(descriptors):  # not Linux, or no procfs
        return []
    opened: list[str] = []
    for entry in sorted(os.listdir(descriptors)):
        try:
            target = os.readlink(os.path.join(descriptors, entry))
        except OSError:
            continue  # raced with a close; it is not open now, which is the claim
        if target.startswith(ACCELERATOR_DEVICE_PREFIXES):
            opened.append(target)
    return opened


def main(arguments: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if arguments is None else arguments)
    if not command:
        print("usage: non_gpu_environment.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    os.execvpe(command[0], command, mask_accelerators())
    raise AssertionError("os.execvpe returned")


if __name__ == "__main__":
    raise SystemExit(main())
