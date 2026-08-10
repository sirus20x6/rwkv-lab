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
    # Both Python scopes, clean. The gate refuses an empty scope on either half
    # -- a scan that matched nothing is indistinguishable from a clean tree --
    # so a fixture without these would fail for a reason unrelated to the case
    # under test.
    (root / "src/rwkv_lab").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "src/rwkv_lab/lever.py").write_text(
        'OUT_DIR = "/opt/trainvm/runs"\n', encoding="utf-8")
    (root / "scripts/tool.py").write_text(
        'CACHE = "/var/lib/trainvm/cache"\n', encoding="utf-8")
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


# --- the Python half: module-level constants only ---------------------------


def test_a_planted_module_constant_reddens(tree):
    """The failure this half exists to catch, in the scope it was added for."""
    assert run(tree).returncode == 0, "the fixture must start clean"

    target = tree / "src/rwkv_lab/lever.py"
    target.write_text('MODEL = "/thearray/git/moe-mla/Qwen3.5-9B-Base"\n',
                      encoding="utf-8")
    result = run(tree)
    assert result.returncode == 1
    assert "src/rwkv_lab/lever.py:1" in result.stdout
    assert "MODEL" in result.stdout
    assert "evaluated at import" in result.stdout


def test_a_historical_path_constant_is_the_policy_not_a_violation(tree):
    """`*_HISTORICAL_PATH` is what step 2 of the policy requires.

    Flagging it would make the gate demand the removal of the value that makes
    the historical fallback work, which is the opposite of the intent.
    """
    (tree / "scripts/tool.py").write_text(
        'CACHE_ENV = "MOE_MLA_CACHE"\n'
        'CACHE_HISTORICAL_PATH = "/thearray/git/moe-mla/cache"\n'
        'OUT_HISTORICAL_ROOT = "/thearray/git/moe-mla/runs"\n',
        encoding="utf-8")
    assert run(tree).returncode == 0


def test_an_environment_fallback_is_not_flagged(tree):
    """`os.environ.get(VAR, "/thearray/...")` is already overridable.

    It passes because the assignment's value is a Call, not a string literal,
    so the module-constant restriction excludes it -- not because of any
    environment-specific check. An explicit one was written and deleted:
    mutation testing showed removing it reddened nothing, and it matched the
    source *line*, so a comment mentioning os.environ beside a bare constant
    would have excused it.
    """
    (tree / "scripts/tool.py").write_text(
        'import os\n'
        'ZTOK = os.environ.get("ZTOK", "/thearray/git/ztok/zig-out/bin/ztok")\n',
        encoding="utf-8")
    assert run(tree).returncode == 0


def test_a_docstring_naming_a_host_path_is_documentation(tree):
    """Prose is not configuration -- the gate's own docstring quotes these roots.

    It passes because a docstring's parent is an Expr rather than an Assign,
    so the module-constant restriction already excludes it. An explicit
    docstring-exclusion helper was written and deleted for the same reason as
    the environment case above: deleting it reddened nothing.
    """
    (tree / "scripts/tool.py").write_text(
        '"""Reads /thearray/git/moe-mla by default on the maintainer\'s box."""\n'
        'OUT = "/opt/trainvm/out"\n',
        encoding="utf-8")
    assert run(tree).returncode == 0


def test_an_argparse_default_is_deliberately_out_of_scope(tree):
    """24 of these exist and they are a separate card.

    Their failure mode is different -- the flag is visible in `--help` and only
    an omitted flag fails -- and gating them needs an allowlist in the dozens.
    This asserts the narrower scope on purpose, so widening it is a deliberate
    edit rather than an accident.
    """
    (tree / "scripts/tool.py").write_text(
        'import argparse\n'
        'def main():\n'
        '    p = argparse.ArgumentParser()\n'
        '    p.add_argument("--data", default="/thearray/git/babyllm/data")\n',
        encoding="utf-8")
    assert run(tree).returncode == 0


def test_a_nested_python_file_is_not_scanned(tree):
    """Both disposition scopes declare `recursive: false`.

    A recursive walk would report subpackages the policy was never applied to,
    which is a population nobody has classified.
    """
    nested = tree / "src/rwkv_lab/training_runtime"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text('X = "/thearray/git/moe-mla"\n',
                                    encoding="utf-8")
    assert run(tree).returncode == 0


def test_an_empty_python_scope_fails_rather_than_passing(tmp_path):
    """The same argument the native half already makes.

    A scan that matched nothing prints the same PASSED as a clean tree, so an
    absent scope has to be a failure or the verdict describes a scan that never
    happened.
    """
    root = tmp_path / "repository"
    (root / "trainvm/src").mkdir(parents=True)
    (root / "trainvm/src/service.cpp").write_text(
        'const char* kNull = "/dev/null";\n', encoding="utf-8")
    result = run(root)
    assert result.returncode == 1
    assert "no Python sources found" in result.stdout


def test_the_verdict_states_both_populations(tree):
    """Two scopes, two counts. One number cannot evidence both scans."""
    result = run(tree)
    assert "255" not in result.stdout  # not the repository's own count
    assert "2 native sources" in result.stdout
    assert "2 top-level Python sources" in result.stdout
