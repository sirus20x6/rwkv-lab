"""The step-zero arming pin's verdict line must not overstate its own checking.

`scripts/print_step_zero_arming_pin.py --check` is the ctest half of the arming
authority: it compares the pinned registry facts against the registry a built
binary carries. Running it needs GCC 16 and a built `trainvm`, which this job
does not have -- so what is asserted here is the part that needs neither, the
sentence the gate prints about what it compared.

That sentence is worth its own file. `differences()` field-compares only the
contracts present on *both* sides; a contract the registry declares and the pin
does not is reported and then skipped. The line claimed every registry profile
was "compared field by field", which is false exactly on the failing path --
the path whose number gets quoted into a card.

These assert the message, never the exit code. The computation was right and
only the sentence was wrong, so the gate reads identically either way and an
exit-code assertion cannot tell the two apart.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts.print_step_zero_arming_pin import (  # noqa: E402
    PIN,
    population_summary,
)


def profiles(*contracts: str) -> dict:
    return {"profiles": [{"contract": name} for name in contracts]}


def test_only_the_compared_contracts_are_called_compared():
    """The regression proof: this sentence would have caught the original slip.

    Two registry contracts, one of them absent from the pin. `differences()`
    reports the absent one and never field-compares it, while the old sentence
    called both "compared field by field".
    """
    summary = population_summary(profiles("a"), profiles("a", "b"))
    assert "2 registry profiles" in summary, summary
    assert f"1 compared field by field against {PIN}" in summary, summary
    assert "1 absent from the pin and reported without being compared" in summary
    assert "2 compared field by field" not in summary, summary


def test_the_compared_and_uncompared_counts_sum_to_the_registry():
    """A reader must be able to check the arithmetic, so it has to hold."""
    summary = population_summary(profiles("a"), profiles("a", "b", "c"))
    assert "3 registry profiles; 1 compared field by field" in summary, summary
    assert "2 absent from the pin" in summary, summary


def test_a_pinned_contract_the_registry_dropped_is_its_own_population():
    """It is not a registry profile, so it cannot be folded into that total."""
    summary = population_summary(profiles("a", "gone"), profiles("a"))
    assert "1 registry profile;" in summary, summary
    assert "1 compared field by field" in summary, summary
    assert "the pin holds 1 contract the registry no longer declares" in summary


def test_a_pin_with_no_profiles_key_compares_nothing():
    """The pin can be missing its list entirely; the sentence must still hold."""
    summary = population_summary({}, profiles("a", "b"))
    assert "2 registry profiles; 0 compared field by field" in summary, summary
    assert "2 absent from the pin" in summary, summary


@pytest.mark.parametrize(
    ("pinned", "observed", "expected"),
    [
        (
            (), (),
            "0 registry profiles; 0 compared field by field against "
            f"{PIN}, 0 absent from the pin and reported without being "
            "compared; the pin holds 0 contracts the registry no longer declares",
        ),
        (
            ("a",), ("a",),
            "1 registry profile; 1 compared field by field against "
            f"{PIN}, 0 absent from the pin and reported without being "
            "compared; the pin holds 0 contracts the registry no longer declares",
        ),
        (
            ("a", "b", "c"), ("a", "b", "c"),
            "3 registry profiles; 3 compared field by field against "
            f"{PIN}, 0 absent from the pin and reported without being "
            "compared; the pin holds 0 contracts the registry no longer declares",
        ),
    ],
    ids=["none", "one", "several"],
)
def test_the_verdict_line_reads_correctly_at_every_count(pinned, observed, expected):
    """Singular and plural, at 0, 1 and n.

    "1 registry profiles" is the shape of error that makes a reader distrust
    the rest of the line, and this count moves whenever a contract is added.
    """
    assert population_summary(profiles(*pinned), profiles(*observed)) == expected


def test_a_single_orphaned_pin_entry_is_singular():
    summary = population_summary(profiles("gone"), profiles())
    assert "the pin holds 1 contract the registry no longer declares" in summary


# --- the sentence `--check` actually prints -------------------------------
#
# The tests above drive `population_summary` directly, which leaves the call
# site untested: reverting `main()` to the old sentence reddened nothing. That
# survivor is why these exist. `--check` needs a built trainvm, so the registry
# read is replaced and everything downstream of it is the real code path.

REGISTRY_PROFILE = {
    "key": {"contract": "example.contract", "adapter": "example_adapter"},
    "lifecycle": {"stateful": True},
    "training_composition": {"slots": {"scorer": "evaluator"}},
    "authoring": {"outputs": {}},
}


def registry(*contracts: str) -> list[dict]:
    built = []
    for name in contracts:
        profile = {**REGISTRY_PROFILE, "key": dict(REGISTRY_PROFILE["key"])}
        profile["key"]["contract"] = name
        built.append(profile)
    return built


def check(tmp_path, monkeypatch, capsys, pinned: list[str], live: list[str]) -> str:
    """Run `--check` end to end with the registry read replaced, return stdout."""
    import scripts.print_step_zero_arming_pin as script

    # The pin is generated by the same code path, over the contracts it should
    # hold, so a difference between the two sides is only ever membership.
    monkeypatch.setattr(script, "registry_profiles", lambda binary: registry(*pinned))
    document = script.build("trainvm")
    pin = tmp_path / script.PIN
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(document, indent=2), encoding="utf-8")

    monkeypatch.setattr(script, "registry_profiles", lambda binary: registry(*live))
    monkeypatch.setattr(
        sys, "argv",
        ["print_step_zero_arming_pin.py", "trainvm",
         "--repository", str(tmp_path), "--check"])
    script.main()
    return capsys.readouterr().out


def test_check_prints_the_population_summary(tmp_path, monkeypatch, capsys):
    """A clean run: every registry profile really was compared field by field."""
    output = check(tmp_path, monkeypatch, capsys, ["a", "b"], ["a", "b"])
    verdict = next(line for line in output.splitlines()
                   if line.startswith("step-zero arming pin:"))
    assert "PASSED" in verdict, output
    assert f"2 registry profiles; 2 compared field by field against {PIN}" in verdict
    assert "0 absent from the pin and reported without being compared" in verdict


def test_check_does_not_claim_it_compared_a_contract_the_pin_lacks(
    tmp_path, monkeypatch, capsys
):
    """The failing path, which is the one whose number gets quoted into a card."""
    output = check(tmp_path, monkeypatch, capsys, ["a"], ["a", "b"])
    verdict = next(line for line in output.splitlines()
                   if line.startswith("step-zero arming pin:"))
    assert "FAILED" in verdict, output
    assert "the registry has b and the pin does not" in output, output
    assert "2 registry profiles; 1 compared field by field" in verdict, verdict
    assert "1 absent from the pin and reported without being compared" in verdict
    assert "2 registry profiles compared field by field" not in verdict, verdict
