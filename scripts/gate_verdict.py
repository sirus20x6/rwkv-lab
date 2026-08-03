"""The final line a CI gate prints must state its own verdict.

Several gates in this repository printed their failures first and a neutral
tally last:

    NO COVERAGE: tests/test_x.py contributes no tests to 'not gpu' ...
    coverage gate: 128 covered, 9 explained exclusions, 138 test files on disk

The exit code was correct, so CI caught the failure. But that last line reads
the same whether the gate passed or failed, so anyone reading the tail of a log
-- or piping through `tail` -- sees a plausible summary and concludes the check
passed. That is how a real PR got pushed with a failing coverage gate: the gate
was run locally first, and its output was truncated to the line that says
nothing about the outcome.

A summary that does not state its own verdict is a measurement that looks valid
while describing the wrong thing, which is the failure shape this repository has
repeatedly had to dig out of receipts, fixtures and pins. This puts the verdict
in the one line most likely to be read alone.

`scripts/validate_native_ci_exclusions.py` and `scripts/acceptance.sh` already
end with an unambiguous verdict and deliberately do not use this helper.
"""

from __future__ import annotations

from collections.abc import Sequence


def verdict_line(label: str, problems: Sequence[object], detail: str) -> str:
    """Return the single line a gate should print last.

    The two forms differ in more than a word. A reader who sees only this line,
    with no context and no colour, must not be able to mistake one for the
    other -- so the failing form leads with FAILED and carries a count, while
    the passing form leads with PASSED and carries only the tally.
    """
    if problems:
        count = len(problems)
        noun = "problem" if count == 1 else "problems"
        return f"{label}: FAILED — {count} {noun} above; {detail}"
    return f"{label}: PASSED — {detail}"
