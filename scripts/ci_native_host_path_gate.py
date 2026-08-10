#!/usr/bin/env python3
"""Fail when a native source names a path that exists only on one machine.

`registered_component_names_are_finite_recipe_choices` in
`trainvm/tests/recipe_profile_tests.cpp` hardcoded `/thearray/git/moe-mla`. It
passed in hosted CI, because it called `expand_json` and never `validate_plan`,
and nothing on that path checks whether a path exists -- the string was inert.

Its sibling `lm_recipe_profiles_tests` did the same thing and *did* reach
catalog validation, which calls `filesystem::exists` on every field of type
`path`. It passed on the deployment host, where `/thearray/git/moe-mla` exists,
and aborted in CI with `TrainingComponentResolutionError: training component
configuration path is unavailable`.

That is the failure this gate exists to prevent, and its shape is the reason a
gate is worth the friction: the test passes for whoever writes it, because they
are on the machine the path names, and fails only in CI -- often on somebody
else's branch, who then has to diagnose a change they did not make. PR #124
found seven such sites across six files when the assumption was that there were
two.

What is forbidden, and what is deliberately not
----------------------------------------------
Only paths rooted at a *host-specific* prefix. Absolute paths are otherwise
completely fine and this gate must never object to them: `/proc` and `/sys` are
the subject matter of the session-authority tests, `/dev/null`, `/usr/bin/true`,
`/opt/trainvm`, `/srv/trainvm` and `/var/lib/trainvm` are deployment locations
the tests legitimately assert on, and strings like
`/spec/workflow/nodes/train/invoke` are JSON pointers that merely look like
paths.

A blanket "no absolute paths" rule would flag all of those. It would be wrong
far more often than right, and a guard that is wrong most of the time gets
bypassed -- the same argument `CLAUDE.md` makes for why the worktree hazard has
no automated guard. So the rule is an explicit deny-list of roots that cannot
exist on an arbitrary machine, and it grows only when a new one is observed.

Scope: `trainvm/` -- sources, headers and tests alike
----------------------------------------------------
The card that prompted this asked for the test sources. The scope here is all
of `trainvm/`, because the hazard is strictly worse in production source than
in a test, and because it costs nothing: the whole directory holds **zero**
host-specific roots today, so this gate ships enforcing a real invariant with
no allowlist and no exceptions file.

That last point is the entire design. `docs/experiment-vm/unwired-module-exclusions.v1.json`
is a countdown that can only shrink, and it is affordable because it is small.
An exceptions list here would start at zero and only ever grow, which is not a
countdown -- it is a place to put the next violation. There is deliberately no
way to record an exception: a host-specific path in `trainvm/` is always a
defect, and if that ever stops being true the honest change is to argue it
here, in this file, rather than to add a line to a JSON document.

Python: module-level constants only
-----------------------------------
`src/rwkv_lab` and `scripts` are now in scope for **module-level string
constants** and nothing else, on the same no-allowlist terms, because their
honest count is zero as of the conversion in PR #256.

The wider Python populations are deliberately excluded, and that is a
measurement rather than an oversight. Across those two scopes there are **24**
argparse defaults naming a host path and **49** further occurrences in ordinary
code, several of which are prose inside log lines and f-strings. A gate over
those needs an allowlist in the dozens -- the instrument-tuned-until-comfortable
failure that keeps the Python unwired-symbol gate unshipped at 79 entries.

(An earlier version of this file put the whole-repository figure at 40 files.
That was measured without looking at `scripts/`, which holds 71 of them; the
real total is 110 files. The correction is on card-ffc44ac4.)

Module constants earn their place because they differ in kind, not degree: they
are evaluated at **import**, so the file fails on another host before `--help`
can print anything naming the field or its environment variable. An argparse
default fails only when the caller omits the flag, and is visible in `--help`.
That distinction is what makes one population worth a zero-allowlist gate and
the other worth a card.

`dashboard` (Go) and `tests` remain out of scope entirely.

Usage:
    python scripts/ci_native_host_path_gate.py [--repository .]
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

SCOPE = "trainvm"
SUFFIXES = (".cpp", ".hpp", ".h", ".cc", ".cxx")

# Roots that cannot be assumed to exist on a machine that is not this one.
#
# `/home/` and `/Users/` are matched with a following name character so that a
# bare mention of the word in prose does not trip the gate; `/root/` likewise
# requires the trailing separator so `/rootfs` is untouched.
HOST_SPECIFIC = (
    ("/thearray", re.compile(r"/thearray\b")),
    ("/home/<user>", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("/Users/<user>", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("/root/", re.compile(r"/root/")),
)


# The Python half. Deliberately narrower than the native half: it checks
# module-level string constants only, and nothing else.
#
# The wider populations were measured and refused. Across `src/rwkv_lab` and
# `scripts` there are 24 argparse defaults naming a host path and 49 further
# occurrences in ordinary code, several of which are prose inside log lines.
# A gate over those needs an allowlist in the dozens, which is an instrument
# tuned until the number looks comfortable -- the same objection that keeps
# the Python unwired-symbol gate unshipped at 79 entries.
#
# Module-level constants are different in two ways. They are evaluated at
# import, so the file fails on another host before `--help` can print anything
# naming the field; and their honest count here is **zero**, so this half ships
# with no allowlist and no exceptions file, exactly like the native half.
PYTHON_SCOPES = ("src/rwkv_lab", "scripts")

# `*_HISTORICAL_PATH` and `*_HISTORICAL_ROOT` are the policy working, not
# violating it: step 2 of the host-path policy is "the historical path, but
# only when it exists on this host", and that value has to be written down
# somewhere. The existing sweep tests in tests/test_host_path_defaults.py skip
# the same substring for the same reason.
HISTORICAL = re.compile(r"HISTORICAL")


def python_sources(repository: pathlib.Path) -> list[pathlib.Path]:
    """Top-level `.py` files in each Python scope.

    Not recursive, matching the disposition catalogs' own `source_scope`,
    which declare `recursive: false` for both of these prefixes. A recursive
    walk would scan subpackages the policy has never been applied to and
    report a population nobody has classified.
    """
    found = []
    for scope in PYTHON_SCOPES:
        root = repository / scope
        if not root.is_dir():
            continue
        found.extend(sorted(p for p in root.glob("*.py") if p.is_file()))
    return found


def python_constant_violations(
        repository: pathlib.Path) -> tuple[list[str], int]:
    """Module-level string constants naming a host-specific path."""
    problems: list[str] = []
    scanned = python_sources(repository)
    for path in scanned:
        relative = path.relative_to(repository)
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        parents: dict[int, "ast.AST"] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            assignment = parents.get(id(node))
            enclosing = parents.get(id(assignment)) if assignment else None
            # Module-level `NAME = "literal"` and nothing else. That one
            # restriction is what excludes the forms an exemption list would
            # otherwise have to enumerate: a docstring's parent is an Expr,
            # not an Assign; `os.environ.get(VAR, "/thearray/...")` assigns a
            # Call, so its string is not the assignment's value; and an
            # argparse default is a keyword inside a call. Mutation testing
            # established this -- explicit checks for the docstring and
            # environment cases were written, and deleting them reddened
            # nothing, because they could never fire. The environment one was
            # worse than dead: it matched the source *line*, so a comment
            # mentioning os.environ beside a bare constant would have excused
            # it.
            if not (isinstance(assignment, ast.Assign)
                    and isinstance(enclosing, ast.Module)):
                continue
            target = assignment.targets[0]
            name = target.id if isinstance(target, ast.Name) else ""
            if HISTORICAL.search(name):
                continue
            for label, pattern in HOST_SPECIFIC:
                match = pattern.search(node.value)
                if match is None:
                    continue
                problems.append(
                    f"{relative}:{node.lineno}: {name or '<constant>'} is a "
                    f"module-level constant naming {match.group(0)}, a path "
                    f"rooted at {label} which exists only on one machine. It "
                    f"is evaluated at import, so on any other host this file "
                    f"fails before it can print anything naming the field. "
                    f"Resolve it through scripts' host_paths.resolve_host_path "
                    f"with a named environment variable, as PR #103 did -- see "
                    f"README 'Host-specific path defaults'.")
                break
    return problems, len(scanned)


def sources(repository: pathlib.Path) -> list[pathlib.Path]:
    """Every native source in scope, with build output excluded.

    `trainvm/build/` is gitignored and holds generated translation units and
    vendored third-party sources; scanning it would report violations nobody
    wrote and cannot fix.
    """
    root = repository / SCOPE
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if any(part.startswith("build") for part in path.relative_to(root).parts[:-1]):
            continue
        found.append(path)
    return found


def violations(repository: pathlib.Path) -> tuple[list[str], int]:
    """Return the problems and how many files were actually read.

    The file count is returned so the verdict line can state it. A gate over a
    scope that silently matched nothing would print PASSED forever, and this is
    a gate whose expected state is green -- so the number of files it read is
    the only evidence in its output that it looked at anything.
    """
    problems: list[str] = []
    scanned = sources(repository)
    for path in scanned:
        relative = path.relative_to(repository)
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in HOST_SPECIFIC:
                match = pattern.search(line)
                if match is None:
                    continue
                problems.append(
                    f"{relative}:{number}: names {match.group(0)}, a path rooted at "
                    f"{label} which exists only on one machine. A test that reaches "
                    f"filesystem::exists on it passes for you and fails in CI. "
                    f"Redirect the already-expanded plan and recompile, the way "
                    f"checked_in_qwen_example_expands_without_source_changes does, "
                    f"rather than editing the path.")
    return problems, len(scanned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse")
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()

    problems, scanned = violations(repository)
    python_problems, python_scanned = python_constant_violations(repository)
    problems = problems + python_problems
    for problem in problems:
        print(f"FAIL: {problem}")

    if scanned == 0:
        # Not a pass. An empty scope means the gate checked nothing, which is
        # indistinguishable from a clean tree in every other respect.
        print(f"FAIL: no native sources found under {SCOPE}/, so nothing was checked")
        problems = problems + ["empty scope"]

    if python_scanned == 0:
        # The same argument, for the half added later. Both scopes have to be
        # non-empty or the verdict describes a scan that did not happen.
        print("FAIL: no Python sources found under "
              f"{', '.join(PYTHON_SCOPES)}, so nothing was checked")
        problems = problems + ["empty python scope"]

    print(verdict_line(
        "native host path gate",
        problems,
        # Neutral tally only. verdict_line supplies PASSED/FAILED, so this half
        # must read true either way -- "carry no path rooted at ..." would be a
        # false sentence on the run that just found one.
        f"{scanned} native sources under {SCOPE}/ and {python_scanned} "
        f"top-level Python sources under {', '.join(PYTHON_SCOPES)} scanned "
        f"for paths rooted at "
        f"{', '.join(label for label, _ in HOST_SPECIFIC)}; the Python half "
        f"checks module-level constants only",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
