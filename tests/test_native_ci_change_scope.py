import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/classify_native_ci_changes.py"
SPEC = importlib.util.spec_from_file_location("classify_native_ci_changes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
classify = MODULE.classify
git_changed_paths = MODULE.git_changed_paths
path_mode = MODULE.path_mode


def test_python_and_documentation_changes_keep_real_catalog_validation() -> None:
    receipt = classify(
        [
            "src/rwkv_lab/trainvm_adapters/handlers.py",
            "docs/experiment-vm/WORKFLOW_COVERAGE.md",
        ]
    )
    assert receipt["mode"] == "catalog"


def test_native_and_native_fixture_changes_require_full_ctest() -> None:
    for path in (
        "trainvm/src/document.cpp",
        "trainvm/tests/trainvm_tests.cpp",
        ".github/docker/trainvm-ci.Dockerfile",
        "docs/experiment-vm/examples/vision-representation-ab.json",
    ):
        assert path_mode(path)[0] == "full"


def test_unrecognized_or_malformed_paths_fail_closed() -> None:
    for path in (
        "new-language/kernel.mojo",
        "../outside.cpp",
        "/absolute/path.py",
        "bad\\windows.cpp",
        "double//separator.py",
    ):
        mode, reason = path_mode(path)
        assert mode == "full"
        assert "unrecognized" in reason or "malformed" in reason


def test_python_tests_do_not_need_a_native_build_by_themselves() -> None:
    assert classify(["tests/test_coconut.py"])["mode"] == "none"


def test_highest_required_tier_wins_independent_of_input_order() -> None:
    paths = ["tests/test_coconut.py", "src/rwkv_lab/coconut.py", "trainvm/src/main.cpp"]
    assert classify(paths)["mode"] == "full"
    assert classify(reversed(paths))["mode"] == "full"


def test_empty_change_set_is_no_native_work() -> None:
    assert classify([]) == {
        "api_version": "trainvm.native-ci-change-scope/v1",
        "mode": "none",
        "paths": [],
    }


def test_unavailable_git_diff_forces_full_tier(monkeypatch) -> None:
    class FailedDiff:
        returncode = 128
        stdout = b""
        stderr = b"missing base\n"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: FailedDiff())
    assert classify(git_changed_paths("missing", "head"))["mode"] == "full"


REPOSITORY = Path(__file__).resolve().parents[1]
WORKFLOW = REPOSITORY / ".github/workflows/tests.yml"


def test_every_tracked_top_level_entry_classifies_without_a_default_skip() -> None:
    """No path in the repository as it stands today reaches ``none`` by accident.

    The classifier's ``none`` tier is the only one that can lose coverage, so it
    is the only one that must be reachable exclusively on purpose. Anything the
    rules do not name -- including the ``deploy/``, ``experiments/``,
    ``patches/`` and ``proto/`` trees that grew after the classifier was
    written -- has to land on ``full``.
    """

    import subprocess

    tracked = subprocess.run(
        ["git", "-C", str(REPOSITORY), "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    paths = [item.decode() for item in tracked if item]
    assert paths, "expected a tracked file list"

    skipped = {path for path in paths if path_mode(path)[0] == "none"}
    # Only the two named non-native trees may skip: the Python suite and the
    # four explicitly listed root documents.
    unexpected = {
        path
        for path in skipped
        if not path.startswith("tests/") and path not in MODULE.NONE_FILES
    }
    assert unexpected == set(), f"paths skip the native job by default: {sorted(unexpected)}"


def test_new_top_level_trees_are_not_silently_skipped() -> None:
    for path in (
        "deploy/trainvm.service",
        "experiments/run.toml",
        "patches/fla.patch",
        "proto/hostd.proto",
        "requirements.txt",
        "uv.lock",
    ):
        assert path_mode(path)[0] == "full", path


def test_workflow_spells_every_narrowing_decision_against_the_narrow_tier() -> None:
    """A future edit back to ``== 'full'`` has to fail rather than be noticed.

    Both gates that can do less work compare against ``catalog``/``none``, so an
    unrecognized mode string buys the whole suite instead of losing it. This is
    the shape, not the wording, and it is the property the card asked to be
    demonstrated rather than asserted.
    """

    text = WORKFLOW.read_text(encoding="utf-8")
    assert '!= \'catalog\'' in text, "exclusion gate must be written against the narrow tier"
    assert '!= "catalog"' in text, "ctest gate must be written against the narrow tier"
    assert "outputs.mode == 'full'" not in text, (
        "a positive == 'full' comparison fails open on an unexpected mode")
    assert "set -euo pipefail" in text, (
        "the classifier step must not let tee mask a crash")
