"""RWKV World tokenizer parity, graded against committed content.

This file used to shell out to a `ztok` binary from `$ZTOK` or from a hardcoded
`/thearray/git/ztok/...` path and assert that our in-process tokenizer matched
its live output. That made another repository's build the authority on whether
this repository's vocabulary parity passed, and nothing recorded which build
answered. It was quiet in both directions — a stale ztok passes an assertion
that should fail, a newer one fails an assertion that is fine — and there was no
in-checkout alternative to fall back to, so the test simply skipped off-box.

The expectations are vendored instead, in `tests/data/world_vocab_parity.json`,
against the vocabulary this repository already ships at
`src/rwkv_lab/assets/rwkv_vocab_v20230424.txt`. Three facts made that the right
option rather than pinning a ztok version:

* the vocabulary is frozen by name and by history — `rwkv_vocab_v20230424` is
  the canonical RWKV World table, dated in its own filename, and the file has
  been touched twice ever in the ztok repository;
* this checkout's copy is byte-identical to the one the test used to read out of
  `/thearray/git/ztok/bench/vocabs/`, so nothing is lost by reading our own;
* `WorldVocab.encode` has been a pure in-process trie for some time. Nothing in
  the assertion needs a subprocess at all.

So a committed digest cannot churn on someone else's build, and the parity test
now runs and means something on a machine that has never heard of ztok.

ztok is not thereby ignored. When one is visible it is cross-checked against the
same committed expectations — graded *by* this repository rather than grading it
— and `tests/ztok_binary.py` reports which one answered, path and version, in
the pytest header on every run including the runs where there is none.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import ztok_binary
from rwkv_lab.generate import WorldVocab

PARITY = json.loads(ztok_binary.PARITY_FIXTURE.read_text())
SAMPLES = [(sample["text"], sample["ids"]) for sample in PARITY["samples"]]


def _vocabulary_path() -> Path:
    return ztok_binary.REPOSITORY / PARITY["vocabulary"]["path"]


def test_vendored_vocabulary_is_the_file_the_expectations_were_generated_from():
    """The expectations are only meaningful against the exact table they used.

    Without this, replacing the vendored vocabulary would silently change what
    every assertion below is asserting about.
    """
    digest = hashlib.sha256(_vocabulary_path().read_bytes()).hexdigest()
    assert digest == PARITY["vocabulary"]["sha256"], (
        f"{PARITY['vocabulary']['path']} is not the vocabulary "
        f"tests/data/world_vocab_parity.json was generated from; regenerate "
        f"the fixture (cd tests && python -m ztok_binary --write) rather than "
        f"editing the pinned digest")


@pytest.mark.parametrize("text,expected", SAMPLES)
def test_world_vocab_encode_matches_the_vendored_expectations(text, expected):
    """The parity assertion. No subprocess, no foreign binary, no skip."""
    vocab = WorldVocab(str(_vocabulary_path()))
    assert vocab.encode(text) == expected


@pytest.mark.parametrize("text,expected", SAMPLES)
def test_world_vocab_decode_round_trips_the_vendored_expectations(text, expected):
    vocab = WorldVocab(str(_vocabulary_path()))
    assert vocab.decode(expected) == text


@pytest.mark.parametrize("text,expected", SAMPLES)
def test_ztok_agrees_with_the_vendored_expectations_when_one_is_present(text, expected):
    """Cross-check, deliberately the weaker direction.

    A disagreement here means the visible ztok and this repository's committed
    expectations have diverged — which is a fact about that build, and the skip
    and failure messages both name it, rather than a verdict on this checkout.
    """
    binary, description = ztok_binary.resolve_ztok()
    if binary is None:
        pytest.skip(f"no ztok to cross-check: {description}")
    version = ztok_binary.ztok_version(binary)
    completed = subprocess.run(
        [binary, "encode", "--model", str(_vocabulary_path()), text],
        capture_output=True, text=True, check=True)
    assert [int(x) for x in completed.stdout.split()] == expected, (
        f"{binary} [{version}] ({description}) disagrees with the expectations "
        f"committed in tests/data/world_vocab_parity.json, which were generated "
        f"by {PARITY['generated_by']['version']}")
