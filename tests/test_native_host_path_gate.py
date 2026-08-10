"""A path that exists only on the deployment host cannot reach native sources.

`scripts/ci_native_host_path_gate.py` exists because a test hardcoding
`/thearray/git/moe-mla` passes for whoever writes it -- they are on the machine
the path names -- and fails only in CI, often on somebody else's branch. PR #124
found seven such sites across six files when the assumption was two.

This gate's expected state is green, and that is the hard thing to test. A gate
that starts green and stays green is indistinguishable from one that checks
nothing, so the tests below are arranged around what would make it hollow
rather than around what it currently reports:

    a planted violation reddens     the gate can fail at all
    each forbidden root reddens     every deny-list entry is live, not decorative
    production source is in scope   the scope is not silently tests-only
    legitimate absolute paths pass  the rule is a deny-list, not "no absolute
                                    paths" -- /proc, /dev/null, /opt/trainvm and
                                    JSON pointers must never trip it
    an empty scope FAILS            checking nothing is not passing
    the tally is verdict-neutral    the last line must not read as a pass when
                                    it failed
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GATE = REPOSITORY / "scripts/ci_native_host_path_gate.py"

_spec = importlib.util.spec_from_file_location("ci_native_host_path_gate", GATE)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["ci_native_host_path_gate"] = gate
_spec.loader.exec_module(gate)


def run(repository: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--repository", str(repository)],
        capture_output=True, text=True, check=False)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal native scope, not a copy of the repository.

    Copying `trainvm/` would make every test here read 253 files. The gate only
    walks the directory and matches text, so a two-file scope exercises it
    identically and stays fast.
    """
    root = tmp_path / "repository"
    (root / "trainvm/tests").mkdir(parents=True)
    (root / "trainvm/src").mkdir(parents=True)
    (root / "trainvm/tests/recipe_tests.cpp").write_text(
        '#include <string>\nconst char* kWorkspace = "/opt/trainvm/work";\n',
        encoding="utf-8")
    (root / "trainvm/src/service.cpp").write_text(
        '#include <string>\nconst char* kNull = "/dev/null";\n', encoding="utf-8")
    return root


def test_the_checked_in_tree_is_clean():
    """The invariant this gate ships enforcing, asserted against the real tree."""
    result = run(REPOSITORY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_a_planted_host_path_reddens(tree):
    """The failure PR #124 had to find by hand, reproduced forward."""
    assert run(tree).returncode == 0, "the fixture must start clean"

    target = tree / "trainvm/tests/recipe_tests.cpp"
    target.write_text(
        target.read_text() + 'const char* kModel = "/thearray/git/moe-mla/m.safetensors";\n',
        encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert "recipe_tests.cpp:3" in result.stdout
    assert "/thearray" in result.stdout
    assert "FAILED" in result.stdout


@pytest.mark.parametrize(
    "planted",
    ["/thearray/x", "/home/sirus/notes.md", "/Users/someone/x", "/root/x"])
def test_every_forbidden_root_is_live(tree, planted):
    """Each deny-list entry must actually redden.

    Parametrised rather than written once because a deny-list is exactly the
    kind of structure that grows an entry whose pattern never matches anything,
    and nothing would say so.
    """
    target = tree / "trainvm/src/service.cpp"
    target.write_text(target.read_text() + f'// {planted}\n', encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1, f"{planted} did not redden the gate"


def test_production_source_is_in_scope_not_only_tests(tree):
    """The card asked for test sources; the scope is deliberately wider.

    A host path is strictly worse in production source than in a test, so a
    gate that only watched `trainvm/tests/` would leave the worse case open.
    """
    (tree / "trainvm/src/service.cpp").write_text(
        'const char* p = "/thearray/git/moe-mla";\n', encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert "src/service.cpp" in result.stdout


@pytest.mark.parametrize(
    "legitimate",
    ["/proc/self/fd", "/sys/fs/cgroup", "/dev/null", "/usr/bin/true",
     "/opt/trainvm/python", "/srv/trainvm", "/var/lib/trainvm",
     "/spec/workflow/nodes/train/invoke", "/rootfs/x", "/homework/x"])
def test_legitimate_absolute_paths_are_not_flagged(tree, legitimate):
    """The rule is a deny-list of host-specific roots, not "no absolute paths".

    `/proc` and `/sys` are the subject matter of the session-authority tests and
    `/spec/...` is a JSON pointer that merely looks like a path. A blanket rule
    would flag all of these, be wrong more often than right, and get bypassed.

    `/rootfs` and `/homework` are here because the patterns for `/root/` and
    `/home/<user>` are one careless edit away from matching them.
    """
    target = tree / "trainvm/src/service.cpp"
    target.write_text(target.read_text() + f'const char* p = "{legitimate}";\n',
                      encoding="utf-8")

    result = run(tree)
    assert result.returncode == 0, f"{legitimate} was wrongly flagged:\n{result.stdout}"


def test_an_empty_scope_fails_rather_than_passing(tmp_path):
    """Checking nothing is not passing.

    Without this the gate would print PASSED on any tree with no `trainvm/` at
    all -- a renamed directory would silently retire the check, which is the
    mode that turns a gate into decoration.
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    result = run(empty)
    assert result.returncode == 1
    assert "nothing was checked" in result.stdout


def test_the_tally_reads_true_on_a_failing_run(tree):
    """The last line must not describe a clean tree when the gate just failed.

    `verdict_line` supplies PASSED/FAILED; the detail it is handed has to be
    verdict-neutral. An earlier draft of this gate ended `... carry no path
    rooted at ...`, which is a false sentence on the run that found one -- and
    the last line is the one most likely to be read alone.
    """
    (tree / "trainvm/src/service.cpp").write_text(
        'const char* p = "/thearray/x";\n', encoding="utf-8")

    verdict = run(tree).stdout.strip().splitlines()[-1]
    assert "FAILED" in verdict
    assert "carry no path" not in verdict
    assert "scanned for paths rooted at" in verdict


def test_build_output_is_not_scanned(tree):
    """`trainvm/build/` is gitignored, generated, and holds vendored sources.

    Reporting a violation there would name a file nobody wrote and nobody can
    fix, which is how a gate teaches people to ignore it.
    """
    generated = tree / "trainvm/build/vendor"
    generated.mkdir(parents=True)
    (generated / "third_party.cpp").write_text(
        'const char* p = "/home/someone/vendored";\n', encoding="utf-8")

    assert run(tree).returncode == 0
