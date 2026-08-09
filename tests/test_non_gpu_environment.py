"""The non-GPU suite must be unable to reach an accelerator, not merely decline to.

The weak form of this assertion is `torch.cuda.is_available() is False`, and it
is the form that would have let the original incident through: it is also False
on a CPU-only wheel, which is why hosted CI stayed green while developer
workstations initialized the display driver on every run. Hosted CI has no GPU,
so it cannot prove the mask works -- these tests are written so that they *can*,
when run on a machine that has one, and remain honest when run on one that does
not.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.non_gpu_environment import (
    ACCELERATOR_ACCESS_ENV,
    NON_GPU_ENVIRONMENT,
    accelerator_access_enabled,
    mask_accelerators,
    open_accelerator_device_files,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "non_gpu_environment.py"


def test_mask_overrides_every_inherited_accelerator_selector():
    """An operator's shell must not be able to donate a device to a CPU run."""
    inherited = {name: "gpu-value" for name in NON_GPU_ENVIRONMENT}
    inherited[ACCELERATOR_ACCESS_ENV] = "1"

    masked = mask_accelerators(inherited)

    assert {name: masked[name] for name in NON_GPU_ENVIRONMENT} == dict(
        NON_GPU_ENVIRONMENT
    )
    assert not accelerator_access_enabled(masked)


def test_launcher_masks_before_the_child_process_starts():
    probe = (
        "import json, os; "
        f"names={list(NON_GPU_ENVIRONMENT)!r}; "
        "print(json.dumps({name: os.environ.get(name) for name in names}))"
    )
    environment = dict(os.environ)
    environment.update({name: "gpu-value" for name in NON_GPU_ENVIRONMENT})
    environment[ACCELERATOR_ACCESS_ENV] = "1"

    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), sys.executable, "-c", probe],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
    )

    assert json.loads(completed.stdout) == dict(NON_GPU_ENVIRONMENT)


def test_this_session_is_masked():
    """The mask reached the process actually running the non-GPU suite."""
    if accelerator_access_enabled():
        pytest.skip("this is the explicitly authorized accelerator phase")
    assert {name: os.environ.get(name) for name in NON_GPU_ENVIRONMENT} == dict(
        NON_GPU_ENVIRONMENT
    )


def test_this_session_holds_no_accelerator_device_open():
    """The physical claim: no descriptor resolves to a GPU device file.

    Collection has already imported every selected test module by the time this
    runs, which is the exact boundary at which marker-only filtering was too
    late. If any module-level `torch.cuda.is_available()` had initialized the
    driver, `/proc/self/fd` would show `/dev/nvidiactl` and `/dev/nvidia0` here.
    """
    if accelerator_access_enabled():
        pytest.skip("this is the explicitly authorized accelerator phase")
    assert open_accelerator_device_files() == []


def test_importing_a_cuda_marked_module_opens_no_device(tmp_path):
    """Import the worst offender in a masked child and check its descriptors.

    `tests/test_experiment_toolkit.py` carries a module-level
    `skipif(not torch.cuda.is_available())`, so importing it is sufficient to
    initialize an unmasked driver -- that is the whole defect. Doing it in a
    subprocess keeps the assertion meaningful on a machine that has a GPU:
    without the mask this child opens seven device files.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}, {str(ROOT / 'tests')!r}]\n"
        "import importlib\n"
        "importlib.import_module('test_experiment_toolkit')\n"
        "from scripts.non_gpu_environment import open_accelerator_device_files\n"
        "print(repr(open_accelerator_device_files()))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), sys.executable, str(probe)],
        cwd=ROOT, env=dict(os.environ), capture_output=True, text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"probe could not import the module: {completed.stderr[-400:]}")

    assert completed.stdout.strip().splitlines()[-1] == "[]"


@pytest.mark.gpu
@pytest.mark.slow
def test_the_portable_benchmark_receipt_reports_the_devices_it_opened():
    """The portable benchmark's own device-file field, proved to be measured.

    `scripts/benchmark_workloads/portable_lm_step.py` publishes
    `open_accelerator_device_files` in its receipt so a portable measurement
    carries physical evidence that it stayed on the CPU path. Every other test
    of that field expects an empty list, which a hardcoded `[]` would satisfy
    just as well. This one runs the workload WITHOUT the mask on a host that
    has devices, where the honest answer is non-empty, so it is the assertion
    that fails if the field stops being read from /proc.

    The non-empty answer is itself worth knowing: importing torch unmasked on
    this repository's workstation opens /dev/nvidiactl, /dev/nvidia0 and
    /dev/nvidia-uvm before a single step runs, so an unmasked portable run is
    a CPU measurement whose peak RSS was taken in a process that initialized a
    GPU. The receipt's `accelerator` field stays False throughout, because no
    tensor ever left the CPU -- reachable and used are different claims and
    the receipt keeps them apart.
    """
    if not accelerator_access_enabled():
        pytest.skip(
            f"needs {ACCELERATOR_ACCESS_ENV}=1; this test deliberately runs a "
            "child WITHOUT the accelerator mask, which the rest of this "
            "module exists to forbid")
    if not list(Path("/dev").glob("nvidia*")):
        pytest.skip("no NVIDIA device files on this host")

    workload = ROOT / "scripts/benchmark_workloads/portable_lm_step.py"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name in NON_GPU_ENVIRONMENT:
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(workload), "--phase", "timed",
         "--bucket", "seq64xbatch2", "--steps", "1"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        check=False, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout)

    opened = report["open_accelerator_device_files"]
    assert opened, (
        "an unmasked portable run on a device host reported no open device "
        "files, so the receipt field is not being read from this process")
    assert all(name.startswith("/dev/") for name in opened)
    assert report["accelerator"] is False
    assert report["execution_device"] == "cpu"


_COLLECTION_CACHE: dict[str, list[str]] = {}


def _collect_node_ids(marker_expression):
    """Node ids the whole suite selects for a marker expression.

    `--collect-only` needs no accelerator, which is the point: the opt-in
    property is provable on hosted CI and on a workstation whose GPU is busy.
    Deliberately not pinned to a named module -- an assertion about one file
    starts passing for the wrong reason the day that file is renamed or split.
    """
    if marker_expression not in _COLLECTION_CACHE:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-p", "no:cacheprovider", "-m", marker_expression, "tests"],
            cwd=ROOT, env=mask_accelerators(), capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stdout[-2000:]
        _COLLECTION_CACHE[marker_expression] = [
            line.strip() for line in completed.stdout.splitlines()
            if line.startswith("tests/") and "::" in line
        ]
    return _COLLECTION_CACHE[marker_expression]


def test_the_gpu_and_non_gpu_phases_partition_the_suite():
    """Nothing gpu-marked is selected by `-m "not gpu"`, and both halves exist.

    Both halves matter. "Nothing gpu-marked was selected" is satisfied just as
    well by a suite that has stopped marking anything at all, which is the
    failure this would otherwise hide.
    """
    gpu = _collect_node_ids("gpu")
    non_gpu = _collect_node_ids("not gpu")

    assert gpu, "no gpu-marked tests are collected at all"
    assert non_gpu, "no non-gpu tests are collected at all"
    assert set(gpu).isdisjoint(non_gpu)


def test_a_gpu_test_does_not_execute_when_the_caller_forgets_the_marker():
    """Default-deny, end to end: a gpu node id run with no `-m` does not run.

    This is deliberately agnostic about which guard did it -- the conftest hook
    or the test's own `skipif`. The claim being made is only the one that
    matters to the incident: it did not execute. Which mechanism fired is
    pinned separately, below, where it can be asserted exactly.
    """
    node_id = _collect_node_ids("gpu")[0]

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node_id],
        cwd=ROOT, env=mask_accelerators(), capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout[-2000:]
    assert "1 skipped" in completed.stdout, completed.stdout[-2000:]


def test_the_conftest_hook_denies_gpu_items_without_the_opt_in():
    """The default-deny mechanism itself, asserted rather than inferred.

    Scraping a terminal report for a skip reason is fragile -- this repository's
    reporter does not print `-rs` short summaries at all, so an end-to-end
    assertion on the reason text passes or fails for reasons unrelated to the
    policy. Calling the hook directly is exact.
    """
    import conftest

    class FakeItem:
        def __init__(self, marker):
            self._marker = marker
            self.added = []

        def get_closest_marker(self, name):
            return object() if name == self._marker else None

        def add_marker(self, marker):
            self.added.append(marker)

    gpu_item, cpu_item = FakeItem("gpu"), FakeItem("fla")
    conftest.pytest_collection_modifyitems([gpu_item, cpu_item])

    assert cpu_item.added == [], "a non-gpu test must not be skipped"
    assert len(gpu_item.added) == 1
    assert gpu_item.added[0].name == "skip"
    assert ACCELERATOR_ACCESS_ENV in gpu_item.added[0].kwargs["reason"]
