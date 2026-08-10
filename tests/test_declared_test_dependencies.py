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
    # The proto-binding gate's only Python need. uv is a launcher for the
    # pinned grpcio-tools generator, not something the suite imports, and the
    # binding bytes are fixed by that pin rather than by uv's own version --
    # so declaring it in the test extra would install it into every test job
    # to no effect and imply a version bound that governs nothing.
    "uv": "launches the pinned grpcio-tools generator; the suite never imports it",
}

# Packages the suite imports at module scope, which therefore have to be
# installed for their coverage to be real rather than a directory of skips.
MODULE_SCOPE_IMPORTS = {
    "jsonschema", "zstandard", "duckdb", "opencv-python-headless",
}


def _test_extra() -> list[str]:
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return document["project"]["optional-dependencies"]["test"]


def _installed_name(argument: str) -> str:
    """The package an argument names, without shell quoting or version pin.

    Both sides of this file have to agree on what "the same package" means.
    A pyproject entry is written `jsonschema>=4.0`; the workflow writes the
    same package as `'flash-linear-attention==0.4.1'`, quotes included,
    because that is how a pin survives the shell. Comparing either form
    literally against a bare name never matches, so both go through here.
    """
    unquoted = argument.strip("'\"")
    return re.split(r"[<>=!~\[]", unquoted, maxsplit=1)[0].strip()


def _requirement_names(requirements: list[str]) -> set[str]:
    return {_installed_name(entry) for entry in requirements}


def _ad_hoc_offenders(text: str, declared: set[str]) -> list[str]:
    """The offender list, over workflow text supplied by the caller.

    Split out from the test so the branches below can be exercised against
    constructed lines. Reading the real file is one caller of this, not the
    only way to reach it.
    """
    offenders: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "pip install" not in stripped:
            continue
        arguments = stripped.split("pip install", 1)[1].split()
        for argument in arguments:
            unquoted = argument.strip("'\"")
            if unquoted.startswith("-") or unquoted.startswith("."):
                break  # an extra install or a flag-led form, e.g. -e '.[test]'
            name = _installed_name(argument)
            if name in ("pip",):
                continue
            if name in ALLOWED_AD_HOC:
                continue
            if name in declared:
                offenders.append(
                    f"{name} is installed ad hoc although the test extra "
                    f"already declares it")
            else:
                offenders.append(
                    f"{name} is installed ad hoc and is declared nowhere; "
                    f"add it to the test extra or to ALLOWED_AD_HOC with a "
                    f"reason")
    return offenders


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
    offenders = _ad_hoc_offenders(
        WORKFLOW.read_text(encoding="utf-8"),
        _requirement_names(_test_extra()),
    )
    assert not offenders, "\n".join(offenders)


def test_a_pinned_allowlisted_install_is_not_an_offender():
    """The allowlist must survive someone pinning the package it names.

    This is not hypothetical. PR #224 pinned flash-linear-attention, which is
    in ALLOWED_AD_HOC, and the gate went red -- it compared the raw shell
    token `'flash-linear-attention==0.4.1'`, quotes and pin included, against
    the bare allowlist key. The allowlist stopped working at the moment
    someone did the responsible thing.
    """
    assert "flash-linear-attention" in ALLOWED_AD_HOC, (
        "this test is written around a real allowlist entry; pick another "
        "one rather than deleting the test")
    for form in (
        "flash-linear-attention",
        "flash-linear-attention==0.4.1",
        "'flash-linear-attention==0.4.1'",
        '"flash-linear-attention>=0.4,<0.5"',
    ):
        assert not _ad_hoc_offenders(f"          pip install {form}", set()), (
            f"{form} names an allowlisted package and must not be reported")


def test_a_pinned_declared_install_reports_the_duplicate_declaration():
    """The pinned form must reach the same branch the bare form reaches.

    Before the fix a pinned declared package fell past the `in declared`
    branch into the else, so the gate said it "is declared nowhere" about
    something the test extra declares -- sending the reader off to add a
    second declaration to fix a message that was simply wrong.
    """
    offenders = _ad_hoc_offenders(
        "          pip install 'jsonschema>=4.0'", {"jsonschema"})
    assert len(offenders) == 1
    assert "already declares it" in offenders[0], offenders[0]
    assert offenders[0].startswith("jsonschema "), (
        f"the offender should name the package, not the shell token: "
        f"{offenders[0]}")


def test_a_pinned_undeclared_install_is_still_an_offender():
    """The whole risk of normalising: it must not smuggle anything past.

    triton is the live example -- the fla job installs it ad hoc and nothing
    declares it. Stripping the pin has to leave that finding intact, and has
    to report the package name rather than the quoted token.
    """
    offenders = _ad_hoc_offenders(
        "          pip install 'triton==3.7.1'", {"jsonschema"})
    assert len(offenders) == 1
    assert "declared nowhere" in offenders[0], offenders[0]
    assert offenders[0].startswith("triton "), offenders[0]


def test_a_flag_led_install_still_stops_at_the_flag():
    """Normalising must not turn the declared forms into findings.

    `pip install -e '.[test]'` and `pip install --upgrade pip` are how the
    workflow is supposed to install things. Stripping quotes made the
    leading-dot check see `.[test]` where it used to see `'.[test]'`, so
    this pins that the guard still fires on both spellings.
    """
    for line in (
        "          pip install -e '.[vision,test,trainvm-worker]'",
        "          pip install '.[test]'",
        "          pip install --upgrade pip",
        "          pip install .[test]",
    ):
        assert not _ad_hoc_offenders(line, set()), line


def test_the_allowlist_itself_states_reasons():
    """An allowlist without reasons becomes a place to hide things."""
    for package, reason in ALLOWED_AD_HOC.items():
        assert len(reason.split()) >= 4, (
            f"{package} is allowed ad hoc with no real explanation")
