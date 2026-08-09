"""Resolve — and always report — the foreign `ztok` this checkout can see.

`ztok` lives in a different repository (github.com/sirus20x6/ztok) and is never
built from this tree. Two tests could see it, and both used to let its presence
or absence change a verdict without saying so:

* `tests/test_world_vocab.py` ran `ztok encode` and asserted our in-process
  tokenizer matched its live output. Whoever last built `/thearray/git/ztok`
  therefore decided whether this repository's vocabulary parity passed. Quiet in
  both directions: a stale ztok passes an assertion that should fail, a newer one
  fails an assertion that is fine, and no message named a version either way.
* `tests/test_benchmark_runner.py` skips on `importlib.util.find_spec("ztok")`
  returning None — the same dependency failing open instead of failing loudly.

The parity assertion no longer asks ztok anything: it grades our tokenizer
against `tests/data/world_vocab_parity.json`, which is committed here (see that
file's `generated_by` block for the build that produced it). ztok, when present,
is graded *by* the same committed expectations rather than being the authority.

What survives is the rule this file exists to enforce, borrowed verbatim from
`tests/trainvm_binary.py`: **whichever ztok is visible is reported, on every run,
including the case where none is.** An unrecorded tool choice makes a surprising
verdict unreproducible by construction.

Unlike trainvm there is no in-checkout candidate to prefer, so the host path is
kept as a labelled fallback rather than removed — it is safe here only because
nothing it says can decide a result. `ZTOK` overrides it, and a `ZTOK` that
cannot be used raises instead of quietly degrading into a skip.

Regenerate the vendored expectations (needs a ztok build) with:

    python -m ztok_binary --write     # from tests/
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

# The historical location of the sibling checkout, kept because
# src/rwkv_lab/generate.py already defaults to it. It is a *foreign* build by
# construction; the report line says so.
HOST_ZTOK = pathlib.Path("/thearray/git/ztok/zig-out/bin/ztok")

PARITY_FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "world_vocab_parity.json"

VENDORED_VOCAB = REPOSITORY / "src" / "rwkv_lab" / "assets" / "rwkv_vocab_v20230424.txt"


class ZtokBinaryError(RuntimeError):
    """An explicitly requested ztok binary could not be used."""


def resolve_ztok() -> tuple[str | None, str]:
    """Return (binary path or None, a sentence naming how it was resolved).

    Raises ZtokBinaryError when ZTOK names something unusable: an explicit
    request that cannot be honoured must not quietly become a skip.
    """
    requested = os.environ.get("ZTOK")
    if requested:
        path = pathlib.Path(requested)
        if not path.is_file():
            raise ZtokBinaryError(f"ZTOK={requested} is not a file")
        if not os.access(path, os.X_OK):
            raise ZtokBinaryError(f"ZTOK={requested} is not executable")
        return str(path), "ZTOK"
    if HOST_ZTOK.is_file() and os.access(HOST_ZTOK, os.X_OK):
        return str(HOST_ZTOK), "host checkout, not built from this tree"
    return None, (
        "not found (set ZTOK to a ztok binary to cross-check it against the "
        "vendored expectations; nothing requires it)"
    )


def ztok_version(binary: str) -> str:
    """`ztok --version`, or a sentence explaining why it could not be asked."""
    try:
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30)
    except OSError as error:
        return f"unknown ({error})"
    if completed.returncode != 0:
        return f"unknown (--version exited {completed.returncode})"
    return completed.stdout.strip() or "unknown (--version printed nothing)"


def binary_report_line() -> str:
    """One line for the pytest header, on every run, whatever the outcome."""
    try:
        binary, description = resolve_ztok()
    except ZtokBinaryError as error:
        return f"ztok binary: unusable -- {error}"
    if binary is None:
        return f"ztok binary: none -- {description}"
    return f"ztok binary: {binary} [{ztok_version(binary)}] ({description})"


def python_binding_report_line() -> str:
    """The importable `ztok` package, which is a separate dependency again.

    `test_benchmark_runner.py` skips on this one. Naming it here means a run
    that skipped is diagnosable without re-deriving which interpreter looked
    where.
    """
    try:
        spec = importlib.util.find_spec("ztok")
    except (ImportError, ValueError) as error:
        return f"ztok python binding: unimportable -- {error}"
    if spec is None or not spec.origin:
        return "ztok python binding: none (not importable on sys.path)"
    return f"ztok python binding: {spec.origin}"


def report_lines() -> list[str]:
    return [binary_report_line(), python_binding_report_line()]


def _regenerate(write: bool) -> int:
    """Rebuild tests/data/world_vocab_parity.json from a live ztok.

    Both producers must agree before anything is written: ztok's live output and
    this repository's in-process tokenizer. A fixture that recorded only one of
    them would pin whichever was wrong.
    """
    import hashlib
    import sys

    sys.path.insert(0, str(REPOSITORY / "src"))
    from rwkv_lab.generate import WorldVocab  # noqa: PLC0415  (regeneration only)

    binary, description = resolve_ztok()
    if binary is None:
        print(f"cannot regenerate: {description}")
        return 1

    existing = json.loads(PARITY_FIXTURE.read_text())
    vocab = WorldVocab(str(VENDORED_VOCAB))
    samples = []
    for sample in existing["samples"]:
        text = sample["text"]
        completed = subprocess.run(
            [binary, "encode", "--model", str(VENDORED_VOCAB), text],
            capture_output=True, text=True, check=True)
        external = [int(x) for x in completed.stdout.split()]
        in_process = vocab.encode(text)
        if external != in_process:
            print(f"ztok and WorldVocab disagree on {text!r}:\n"
                  f"  ztok:       {external}\n"
                  f"  WorldVocab: {in_process}")
            return 1
        samples.append({"text": text, "ids": external})

    document = dict(existing)
    document["generated_by"] = {
        "tool": "ztok",
        "version": ztok_version(binary),
        "binary": binary,
        "cross_checked_against": "rwkv_lab.generate.WorldVocab",
    }
    document["vocabulary"] = {
        "path": str(VENDORED_VOCAB.relative_to(REPOSITORY)),
        "sha256": hashlib.sha256(VENDORED_VOCAB.read_bytes()).hexdigest(),
    }
    document["samples"] = samples
    serialized = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if write:
        PARITY_FIXTURE.write_text(serialized)
        print(f"wrote {PARITY_FIXTURE}")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":  # pragma: no cover - developer entry point
    import sys

    raise SystemExit(_regenerate(write="--write" in sys.argv[1:]))
