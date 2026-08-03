"""CI must install what package metadata declares, not a hand-kept list.

Four workflow steps used to `pip install` packages that pyproject.toml did not
mention: jsonschema in the schema job, and zstandard/duckdb/opencv-python-headless
repeated verbatim in three others. That meant no documented command reproduced
CI from a clean checkout, the versions were unbounded, and the three copies had
to be kept in step by hand.

These assert the declaration is the single source of truth. A package installed
ad hoc must either move into an extra or be named here with a reason, so the
gap cannot reopen quietly.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = REPOSITORY / ".github/workflows/tests.yml"
PYPROJECT = REPOSITORY / "pyproject.toml"

# Installed outside the metadata on purpose. Both are CUDA-stack packages the
# base dependencies deliberately exclude, and the reason is not "nobody got
# round to declaring it".
ALLOWED_AD_HOC = {
    # A specific CPU wheel index, which an extra cannot express.
    "torch": "installed from the CPU wheel index; an extra cannot pin an index",
    # The fla job exists precisely to run against the real published package,
    # so declaring it would defeat the point of the job.
    "flash-linear-attention": "the fla job exists to test the real package",
}

# Packages the suite imports at module scope, which therefore have to be
# installed for their coverage to be real rather than a directory of skips.
MODULE_SCOPE_IMPORTS = {
    "jsonschema", "zstandard", "duckdb", "opencv-python-headless",
}


def _test_extra() -> list[str]:
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return document["project"]["optional-dependencies"]["test"]


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].strip()
        for entry in requirements
    }


def test_the_test_extra_declares_every_module_scope_import():
    missing = MODULE_SCOPE_IMPORTS - _requirement_names(_test_extra())
    assert not missing, (
        f"{sorted(missing)} are imported at module scope by the suite but the "
        f"test extra does not declare them, so a clean checkout cannot "
        f"reproduce CI")


def test_every_declared_test_dependency_carries_a_lower_bound():
    """An unbounded dependency lets a release change what the gates prove."""
    unbounded = [
        entry for entry in _test_extra()
        if not re.search(r"[<>=~]=?\s*\d", entry)
    ]
    assert not unbounded, (
        f"{unbounded} have no version bound, so a future release can change "
        f"the suite's behaviour with nothing recording what it was written "
        f"against")


def test_jsonschema_is_bound_to_the_draft_the_schema_declares():
    """Validator behaviour is draft-specific, so the bound is the schema's.

    Not a style rule: draft 2020-12 support arrived in jsonschema 4.0, and the
    gate selects its validator from the document's own $schema. A lower bound
    would silently pick a different validator class.
    """
    schema = (REPOSITORY / "docs/experiment-vm/experiment-v1.schema.json")
    assert "2020-12" in schema.read_text(encoding="utf-8"), (
        "the schema no longer declares draft 2020-12; revisit the jsonschema "
        "lower bound before changing this test")
    declared = [e for e in _test_extra() if e.startswith("jsonschema")]
    assert declared == ["jsonschema>=4.0"], (
        f"expected jsonschema>=4.0 for draft 2020-12, found {declared}")


def test_no_workflow_step_installs_an_undeclared_package():
    """The gate against reopening the gap.

    Matches `pip install foo bar`, but not `pip install -e '.[extra]'` and not
    `pip install --upgrade pip`, since those are the declared forms.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    declared = _requirement_names(_test_extra())
    offenders: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "pip install" not in stripped:
            continue
        arguments = stripped.split("pip install", 1)[1].split()
        for argument in arguments:
            if argument.startswith("-") or argument.startswith("."):
                break  # an extra install or a flag-led form, e.g. -e '.[test]'
            if argument in ("pip",):
                continue
            if argument in ALLOWED_AD_HOC:
                continue
            if argument in declared:
                offenders.append(
                    f"{argument} is installed ad hoc although the test extra "
                    f"already declares it")
            else:
                offenders.append(
                    f"{argument} is installed ad hoc and is declared nowhere; "
                    f"add it to the test extra or to ALLOWED_AD_HOC with a "
                    f"reason")
    assert not offenders, "\n".join(offenders)


def test_the_allowlist_itself_states_reasons():
    """An allowlist without reasons becomes a place to hide things."""
    for package, reason in ALLOWED_AD_HOC.items():
        assert len(reason.split()) >= 4, (
            f"{package} is allowed ad hoc with no real explanation")
