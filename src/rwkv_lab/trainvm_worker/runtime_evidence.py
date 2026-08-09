"""Worker-measured runtime, device and ABI evidence for cache namespaces.

The authority decides which bytes may be reused. It cannot decide it alone: the
facts that make a compiled artifact reusable -- which loader graph is actually
mapped, which driver the kernel is running, which device the process was handed,
which ABI the host presents -- exist only inside the worker process. So the
worker measures them and the authority admits them.

Admission, not transport, is where trust is decided, and the shape of this
module reflects that:

  * the report carries no identity the authority derives for itself. There is
    no host id, no boot id, no launch spec digest, no inventory receipt digest,
    no resource binding digest, and no receipt name. A worker therefore cannot
    write a document that names the receipt its cache lookup would read;
  * the identity fields it does carry -- run, node, attempt, launch nonce,
    concurrency key, lease, fencing token -- are echoes of the sealed bootstrap.
    The authority compares them, and refuses a report from a superseded attempt
    rather than believing it;
  * the runtime closure fingerprint is not measured here. It is whatever the
    pre-import guard verified for this deployment, passed in by the caller. A
    second implementation that re-derived it would be a second answer to a
    question the guard already answered under stricter conditions.

Everything else in the report is measured from the running host with the
standard library only, because this runs on the same side of the trust boundary
as the guard and must not depend on the third-party stack it is describing.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import sysconfig
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# The guard's driver identity, not a second reading of the same file. The
# closure pins whatever that function returned when the deployment was sealed
# and refuses the host if it has moved since, so a probe that parsed
# /proc/driver/nvidia/version its own way could report a driver the closure had
# already accepted as unchanged, or the reverse. The `driver_version` field
# below keeps its name -- it is the wire field of the worker evidence report,
# shared with the proto and the C++ claim -- but its value is the guard's
# identity, which is a version token and a module build ID rather than the
# whole first line of /proc/driver/nvidia/version.
from ..trainvm_runtime_guard import driver_identity
from ._canonical import canonical_dumps, is_bounded_text, is_digest, sha256_digest
from .bootstrap import WorkerBootstrap

REPORT_API_VERSION = "trainvm.worker-runtime-evidence/v1"
NVIDIA_GPU_DIRECTORY = "/proc/driver/nvidia/gpus"
CPU_INFORMATION_FILE = "/proc/cpuinfo"
MAXIMUM_RUNTIME_VERSIONS = 64
# Distributions whose version changes what a compiled artifact is. Absent ones
# are omitted rather than recorded as null: "not installed" and "installed at
# version none" are different runtimes and must not share a namespace entry.
RUNTIME_VERSION_DISTRIBUTIONS = (
    "bitsandbytes",
    "deepspeed",
    "flash-attn",
    "flash-attn-3",
    "flash-linear-attention",
    "numpy",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "torch",
    "torchao",
    "transformer-engine",
    "triton",
    "xformers",
)
# The absent-accelerator driver identity. A portable namespace still has to
# name one, and an empty string is not a name.
NO_DRIVER = "none"
_PCI_ADDRESS = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]")


class RuntimeEvidenceError(RuntimeError):
    pass


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="strict") as stream:
            return stream.read()
    except OSError:
        return None


def _nvidia_device_identity(pci_address: str) -> dict[str, str]:
    """Device UUID and model, read from the driver rather than from CUDA.

    /proc exposes this without initialising a compute context, so the probe
    does not perturb the device it is describing and does not need the CUDA
    runtime it may be about to declare incompatible.
    """
    # The address reaches a path, so its shape is checked rather than trusted.
    # It arrives from the authority's inventory today; a probe that let it name
    # an arbitrary path would be one caller change away from reading one.
    if not _PCI_ADDRESS.fullmatch(pci_address):
        raise RuntimeEvidenceError(f"malformed PCI address: {pci_address!r}")
    information = _read_text(
        os.path.join(NVIDIA_GPU_DIRECTORY, pci_address, "information")
    )
    identity: dict[str, str] = {}
    if information is None:
        return identity
    for line in information.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        name = key.strip().lower()
        if name in {"model", "gpu uuid", "device minor"}:
            identity[name] = value.strip()
    return identity


def _cpu_features() -> list[str]:
    text = _read_text(CPU_INFORMATION_FILE)
    if text is None:
        return []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "flags":
            return sorted(set(value.split()))
    return []


def host_abi_identity() -> dict[str, Any]:
    """The ABI a compiled artifact was linked against, as this host presents it."""
    uname = os.uname()
    try:
        libc = os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError):
        libc = None
    return {
        "libc": libc or "unknown",
        "machine": uname.machine,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": sys.implementation.name,
        "python_platform": sysconfig.get_platform(),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "sysname": uname.sysname,
        # The kernel release, not the full version string: the build host and
        # timestamp in `uname -v` move on every distribution rebuild without
        # changing anything a compiled kernel binds to.
        "release": uname.release,
    }


def runtime_versions() -> list[dict[str, str]]:
    """Installed versions of the distributions a compiled artifact binds to."""
    from importlib import metadata

    identities = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
    }
    for name in RUNTIME_VERSION_DISTRIBUTIONS:
        try:
            identities[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception as error:  # pragma: no cover - defensive
            raise RuntimeEvidenceError(
                f"runtime version identity could not be read: {name}"
            ) from error
    if len(identities) > MAXIMUM_RUNTIME_VERSIONS:
        raise RuntimeEvidenceError("runtime version identity set is unbounded")
    return [
        {"name": name, "version": identities[name]} for name in sorted(identities)
    ]


def _compute_identity(
    selected_devices: Sequence[dict[str, str]],
) -> dict[str, Any]:
    if not selected_devices:
        return {
            "compute_device_vendor": "cpu",
            "compute_architecture": platform.machine() or "unknown",
            "driver_version": NO_DRIVER,
            "compute_device_uuid": None,
            "compute_device_pci_address": None,
            "attributes": {"cpu_features": _cpu_features()},
        }
    vendors = {device.get("vendor") for device in selected_devices}
    if len(vendors) != 1:
        raise RuntimeEvidenceError("selected devices do not share one vendor")
    vendor = next(iter(vendors))
    if vendor != "nvidia":
        raise RuntimeEvidenceError(f"unsupported accelerator vendor: {vendor}")
    version = driver_identity()
    if not is_bounded_text(version, 256):
        raise RuntimeEvidenceError(
            "an NVIDIA device was selected but the kernel exposes no usable "
            "driver identity"
        )
    architectures = {device.get("architecture", "") for device in selected_devices}
    if len(architectures) != 1 or not next(iter(architectures)):
        raise RuntimeEvidenceError(
            "selected devices do not share one compute architecture"
        )
    models: dict[str, dict[str, str]] = {}
    for device in selected_devices:
        address = device.get("pci_address", "")
        if not address:
            raise RuntimeEvidenceError("a selected device has no PCI address")
        models[address] = _nvidia_device_identity(address)
    exact = selected_devices[0] if len(selected_devices) == 1 else None
    return {
        "compute_device_vendor": "nvidia",
        "compute_architecture": next(iter(architectures)),
        "driver_version": version,
        # Measured from the driver, never taken from the caller's device
        # description: the caller may say which device it was fenced to, but
        # not what that device's identity is.
        "compute_device_uuid": (
            models[exact["pci_address"]].get("gpu uuid") if exact else None
        ),
        "compute_device_pci_address": exact["pci_address"] if exact else None,
        "attributes": {"devices": models},
    }


def measure_worker_runtime_evidence(
    bootstrap: WorkerBootstrap,
    *,
    runtime_closure_fingerprint: str,
    selected_devices: Iterable[dict[str, str]] = (),
    placement_specific: bool = False,
) -> dict[str, Any]:
    """Measure this runtime and bind the measurement to the sealed bootstrap.

    `runtime_closure_fingerprint` is the digest the pre-import guard verified,
    not a value measured here; see the module docstring. `selected_devices` are
    the devices the authority fenced this launch to, echoed back so the
    authority can check that what the worker sees is what it granted -- a
    device the worker names but was not fenced to is refused on admission.
    """
    if not is_digest(runtime_closure_fingerprint):
        raise RuntimeEvidenceError("verified runtime closure digest is invalid")
    devices = list(selected_devices)
    compute = _compute_identity(devices)
    attributes = compute.pop("attributes")
    versions = runtime_versions()
    abi = host_abi_identity()
    report: dict[str, Any] = {
        "api_version": REPORT_API_VERSION,
        "run_id": bootstrap.run_id,
        "node_id": bootstrap.node_id,
        "attempt_id": bootstrap.attempt_id,
        "launch_nonce": bootstrap.launch_nonce,
        "concurrency_key": bootstrap.concurrency_key,
        "lease_id": bootstrap.lease_id,
        "fencing_token": bootstrap.fencing_token,
        "compute_device_vendor": compute["compute_device_vendor"],
        "compute_architecture": compute["compute_architecture"],
        "driver_version": compute["driver_version"],
        "runtime_versions": versions,
        "runtime_closure_fingerprint": runtime_closure_fingerprint,
        "host_abi_digest": sha256_digest(canonical_dumps(abi)),
        # Facts that decide reuse but appear nowhere else in the claim: the
        # host's instruction set, or the exact device attributes behind an
        # architecture name that several parts share.
        "compute_compatibility_digest": sha256_digest(
            canonical_dumps(
                {
                    "architecture": compute["compute_architecture"],
                    "attributes": attributes,
                    "driver_version": compute["driver_version"],
                    "vendor": compute["compute_device_vendor"],
                }
            )
        ),
    }
    # Placement identity exists only in a placement-specific namespace. A
    # portable report that carried it would be refused, so it is not written:
    # the two namespaces are different questions, not one answer with a flag.
    if placement_specific:
        uuid = compute["compute_device_uuid"]
        address = compute["compute_device_pci_address"]
        if not is_bounded_text(uuid, 256) or not is_bounded_text(address, 256):
            raise RuntimeEvidenceError(
                "a placement-specific launch has no exact device identity"
            )
        report["compute_device_uuid"] = uuid
        report["compute_device_pci_address"] = address
    return report


def worker_runtime_evidence_bytes(report: dict[str, Any]) -> bytes:
    """The canonical wire form the authority decodes with an exact schema."""
    return canonical_dumps(report)


def write_worker_runtime_evidence(path: Path, report: dict[str, Any]) -> None:
    path.write_bytes(worker_runtime_evidence_bytes(report))
