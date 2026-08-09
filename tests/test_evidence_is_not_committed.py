"""`evidence/` is generated output and must never enter a commit.

Two failures motivated this, and they fail differently, so both are asserted.

The quiet one: `evidence/acceptance.json` and `evidence/python-cpu.xml` were
committed by accident in #43/#44 and sat in the tree for six days. An acceptance
receipt names the commit it ran against, but a *committed* receipt is served to
every reader at whatever commit they have checked out, so the two diverge on the
very next commit. The stale pair described a 2026-08-03 run of a dirty worktree
while claiming to be the repository's acceptance evidence. That is worse than no
receipt, because it reads as evidence and is precise about the wrong thing.

The bulk one: a GPU qualification run left 187 files under
`evidence/generic-hf-multimodal-sft-cfb83cd/` with nothing ignoring the
directory. The repository's existing `*.pt`/`*.safetensors`/`*.bin` rules did
cover its checkpoint payloads, so this was never mainly about size — but 150
files (~4.3MB of receipts, configs, manifests and logs) were still stageable by
any `git add -A`. Relying on extension rules to keep covering a directory whose
contents nobody controls is the fragile part; ignoring it by shape is not.

The ignore rule is what prevents both, so these tests exercise `git` itself
rather than parsing `.gitignore`: the question is what git actually does, not
what the file appears to say.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


# Paths a real run produces. The qualification directory name embeds a commit
# sha, so the rule has to cover a directory that does not exist yet — these use a
# deliberately different sha from the one observed, to catch a rule accidentally
# written against the single instance that prompted it.
GENERATED_PATHS = (
    "evidence/acceptance.json",
    "evidence/python-cpu.xml",
    "evidence/python-fla.xml",
    "evidence/gpu.xml",
    "evidence/native-configure.log",
    "evidence/go.log",
    "evidence/generic-hf-multimodal-sft-0000000/receipts/qualification-receipt.json",
    "evidence/generic-hf-multimodal-sft-0000000/artifacts/published/"
    "checkpoint-step-1-publication-4/payload/optimizer.pt",
    "evidence/generic-hf-multimodal-sft-0000000/inputs/model/model.safetensors",
    "evidence/some-future-qualification-deadbeef/logs/engine.log",
)


@pytest.mark.parametrize("path", GENERATED_PATHS)
def test_generated_evidence_path_is_ignored(path: str) -> None:
    """git itself must refuse to stage this path."""
    result = _git("check-ignore", "-q", "--no-index", path)
    assert result.returncode == 0, (
        f"{path} is NOT ignored, so `git add -A` would stage it. "
        "Add or widen the /evidence/ rule in .gitignore."
    )


def test_no_evidence_file_is_tracked() -> None:
    """Nothing under evidence/ may be tracked, ignore rule or not.

    `.gitignore` does not apply to already-tracked files, so a rule alone would
    not have removed the two stale receipts — it would only have hidden the
    problem from `git status` while they stayed in every checkout.
    """
    tracked = _git("ls-files", "evidence").stdout.split()
    assert tracked == [], (
        "tracked files under evidence/: "
        + ", ".join(tracked)
        + ". evidence/ is generated output; a committed receipt claims the "
        "commit it is checked out at rather than the one it ran against. "
        "Remove with `git rm --cached`."
    )
