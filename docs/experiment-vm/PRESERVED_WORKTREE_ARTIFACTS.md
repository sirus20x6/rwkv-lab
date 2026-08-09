# Preserved worktree artifacts

Eighteen files in this commit were, until it landed, untracked working-tree files
in three stale git worktrees. None of them existed on any ref. A single
`git worktree remove` on any of those three directories would have destroyed
them, and worktree cleanup is an ordinary, well-intentioned thing to do — an
autopilot sweep removed sixty worktrees on 2026-08-09 and reached these three
before an untracked-file check was added to the rule.

This file is the provenance record: where each file came from, what it is, and
what is known about whether it still matters. Nothing was deleted, moved, or
edited at its source. Every file here is a byte-for-byte copy; the SHA-256 below
is the digest of both the original and the copy.

## Why these are in the repository rather than in a scratch directory

Several are **locked documents** — `trainvm lock-input-content` output that pins
the exact content of every input root a run was authorized to consume. A locked
document is the authority record for what a run was permitted to do. Having one
exist only as an untracked file in a stale worktree is the same category of
problem as a receipt naming a commit nobody can find.

The rest are experiment specifications whose runs left output in
`/thearray/git/moe-mla/runs/`. The run output is gitignored, so the
specification is the only durable statement of what produced it.

## What this commit does NOT claim

Preserving a document is not endorsing it, scheduling it, or authorizing it.
Three of these describe GPU runs that were never started, two of them gated on
operator authorization that has not been given. Read the "status" column before
acting on anything here. In particular, nothing in this directory is standing
authorization to start a training run.

## Source A — `/thearray/git/moe-mla-parity-integration`

Branch `integration/parity-candidate`, HEAD `1a906c9`, unmerged. That worktree
also carries 32 modified tracked files which are **not** part of this commit;
they remain there untouched. They were snapshotted separately onto the branch
`preserve/parity-integration-worktree-20260809` — see "The 32 modified tracked
files" at the end of this document.

### The Qwen3.6 caption-distillation bundle

This is one coherent sealed bundle and is preserved whole. The two
`.locked.json` seals are content pins over the other four documents; a seal
without its subject is not useful, so the inputs travel with it even though four
of them also exist on the unmerged branch `experiment/qwen36-caption-lora`
(commit `8559473`, byte-identical — verified).

| file | sha256 | mtime | in git before? |
| --- | --- | --- | --- |
| `docs/qwen36_caption_finetune.md` | `3e34aa1340cc01e878d0d3d6cb0d892b7545192aadd9b2a8352c16c382177ce7` | 2026-08-05 04:35 | `8559473`, identical |
| `experiments/qwen36_caption_distill_lora_v1.json` | `ae83adce01edb36b41498773bca152ce0b4523a61bdfdcef094ecf3253d98ad8` | 2026-08-05 04:35 | `8559473`, identical |
| `experiments/qwen36_caption_distill_lora_v1.input-roots.json` | `72970140fb01caeeb470d165b00471152c7beee7964beef0a87b084d0c2310bf` | 2026-08-05 04:35 | `8559473`, identical |
| `experiments/qwen36_caption_distill_lora_v1.targets.json` | `f501a5d8a91e8324182556f6cc3d2d900c9d63b5627fa7efa88c961d1e4c2456` | 2026-08-05 04:35 | `8559473`, identical |
| `experiments/qwen36_caption_distill_lora_v1.trainvm.json` | `cb4f5636f9adc3f2ea3eedaeb5d6d75c3fd75f3640fb06abc2333f60913c7f6d` | 2026-08-05 04:35 | `8559473`, identical |
| `experiments/qwen36_caption_distill_lora_v1.locked.json` | `a4778f4ac1d11159a5f4cf7ee3b2d682561acbdfec2b74778d2271f288e7b5db` | 2026-08-05 05:24 | **no ref, anywhere** |
| `experiments/qwen36_caption_distill_lora_r256_pipeline.locked.json` | `8a7fad61b70192b8a03e253493d67fc63515958087e7e7bc2852ca501d4ce5ca` | 2026-08-05 10:24 | **no ref, anywhere** |

- `docs/qwen36_caption_finetune.md` — the run contract: frozen dataset digest
  `4869008d…`, 11,909/680/674 split, rank-256 LoRA alpha 512 over exactly 311
  enumerated language/output modules (402,751,488 adapter parameters), one epoch
  at accumulation 16 to 745 optimizer steps, and the untouched-baseline →
  adapter → blinded three-way 200-image audit evaluation order.
- `..._v1.json` — the standalone resolved trainer config for that contract.
- `..._v1.input-roots.json` — the three content roots to seal (dataset,
  `src/rwkv_lab`, base checkpoint).
- `..._v1.targets.json` — the audited 311-module LoRA target manifest,
  digest `b7fff9ea…`.
- `..._v1.trainvm.json` — the declarative TrainVM graph, unsealed.
- `..._v1.locked.json` — **locked seal**, identity
  `qwen36-caption-distill-lora-r256-v2`. Pins 13,271 files / 1,970,803,801 bytes
  of dataset, 408 files of `src/rwkv_lab`, and 27 files / 110,625,494,895 bytes
  of base checkpoint.
- `..._r256_pipeline.locked.json` — **locked seal**, identity
  `qwen36-caption-distill-lora-r256-v2-pipeline-resume`. Same three roots, same
  digests, resume-shaped workflow and a 120 s lease timeout.

**Status: superseded as a plan, retained as evidence.** `docs/experiment-vm/QWEN_CAPTION_DECLARATIVE_MIGRATION.md`
on main carries this run's decisions forward into
`examples/qwen-caption-lora-r256.recipe-instance.v1.json` and states in its own
words that "the old bespoke implementation and its artifacts remain historical
evidence until a real declarative run passes". These two seals are the `-v2`
generation that preceded the `corrected-v3` run whose 674 adapter-test
generations failed on a CUDA timeout; they are the only surviving record of what
`-v2` was authorized to consume.

### The ztok modded-nanogpt search trials

Six trial specifications from the `ztok-enhancement-search` campaign, all
authored 2026-08-04. **They are a deliberate series, not disposable sweep
output** — the card that filed this work warned against assuming otherwise, and
the warning was right.

| file | sha256 | mtime |
| --- | --- | --- |
| `experiments/ztok_modded_search_20260804_q16_nsys_profile_replay_full.json` | `5e9f92e670690bb694abf6fab6f459364c4520f9df1fe807a6a8b11eb4f72114` | 2026-08-04 21:29 |
| `experiments/ztok_modded_search_20260804_q23_batch8_to16_middle.json` | `4d640d37ee9c2a22101bcf43eabde317862749b644c8536715cc1b10a716087b` | 2026-08-04 10:47 |
| `experiments/ztok_modded_search_20260804_q24_batch8_to16_to24_middle.json` | `a66facf447da8acf3aa71d212778df9f08a80737c47a5ea38e53f4605a67e0aa` | 2026-08-04 10:54 |
| `experiments/ztok_modded_search_20260804_q25_context_batch_curriculum_middle.json` | `b1b54b742ec035b4dcbde3f3d54672e9e631d6279450a61f6bafc22b24a8e0c0` | 2026-08-04 11:05 |
| `experiments/ztok_modded_search_20260804_q26_late_linear_cooldown_full.json` | `727ed614f9626b9e1e80b15cdfb2d3463b5ab62416e27fff17b56a6d49dc6a19` | 2026-08-04 11:14 |
| `experiments/ztok_modded_search_20260804_q27_late_head_split_full.json` | `351ba4e49f782d3179884b7f21945947d856d07a0f0d0833ce30074ee924bda4` | 2026-08-04 11:33 |

None was ever committed on any ref. Each varies exactly one lever against a
shared baseline:

- **q16** — replays the already-characterised full rung unchanged, purely to
  capture an nsys GPU trace (`execution.gpu_trace`, 8 warmup / 128 skip / 32
  capture steps, required `gpu_trace` artifact). The only one with tracing.
- **q23** — batch ramp 8 → 16 at half progress (`batch_size_schedule`).
- **q24** — batch ramp 8 → 16 → 24 in thirds. Direct A/B against q23.
- **q25** — joint context+batch curriculum via `train_shape_schedule`:
  (24, 512) → (12, 1024) → (8, 1904), invariant source-byte budget.
- **q26** — `lr_decay_kind: late_linear` with `cooldown_fraction: 0.2`.
- **q27** — late embedding/head untying at `head_split_fraction: 0.666…`.

What makes them a series: a shared `campaign: ztok-enhancement-search` label, a
shared `metadata.description`, a shared compile/input-content search cache, a
campaign-level `concurrency_key` of `ztok-modded-search-20260804` on q23–q26,
and `source_byte_budget` values (`227844989` middle, `455689978` final) that are
exactly the rungs declared by the campaign design document. The q-numbering is
dense — run directories exist for q00 through q27 — so these six are the
surviving specifications of a 28-trial series whose earlier specs were not kept.

**Status: retained, and five of six describe runs that actually happened.**
`/thearray/git/moe-mla/runs/ztok-modded-search-20260804-q{23,24,25,26,27}-*/`
each hold `summary.json`, `metrics.jsonl`, `train.jsonl` and checkpoints; that
output is gitignored, so these specifications are the only durable statement of
what produced it. The q16 nsys replay directory is empty — specified, never run.

Two caveats worth knowing before treating these as reproducible:

- They are generated by `scripts/materialize_ztok_search_trial.py`, which was
  itself uncommitted and evolving during the campaign. q26 carries
  `timeout_seconds: 600` where today's generator would emit `900` for its
  budget, so it is **not** byte-reproducible from the current script. The
  documents are the authority, not the generator.
- The campaign's parent design document
  (`experiments/ztok_modded_nanogpt_search_v1.json`, schema
  `rwkv-lab.optimization-campaign-design.v1`, added in `59b8588`), its baseline
  run document, and the prose docs `ZTOK_MODDED_NANOGPT_SEARCH.md` and
  `ZTOK_SEARCH_TRIALS.md` all live only on unmerged branches and are **not**
  brought forward here. They are tracked, so they were not at risk under this
  card, but the six trials are less interpretable without them. Filed
  separately.

### `docs/experiment-vm/preserved/trainvm_normal_dashboard_bridge.sh`

`43f249c726f9978d3566dbf03ea3f08030e85574f2cd0a00e1eaf5b79b8cbd03`, 2026-08-05
06:23, never committed on any ref. Its original location was
`scripts/trainvm_normal_dashboard_bridge.sh`.

A 40-line bash bridge that tails a caption run's `baseline-test.jsonl` and
appends `phase: "baseline_eval"` progress records to `train.jsonl` so the
baseline-generation phase is visible on the dashboard before optimizer step 1.
It polls `/proc/<worker_pid>` rather than `kill -0` because hostd runs the worker
under an isolated UID — the comment in the file explains that this is sound only
because the bridge emits progress and never controls the worker.

**It is deliberately not restored to `scripts/`.** `scripts/` is an exhaustively
enumerated disposition scope (`source-dispositions.scripts.v1.json`, prefix
`scripts`, non-recursive, `.py`/`.sh`), so adding a shell file there requires a
catalog entry, moves `source_tree_digest`, and moves three pinned counts in
`trainvm/tests/source_disposition_catalog_tests.cpp` — and the catalog entry
would mean assigning a disposition class, canonical entry point, effect set and
resume relevance to a script this commit's author did not write and cannot
classify honestly. Preserving the bytes and classifying the script are two
different jobs; this commit does the first. `docs/experiment-vm/preserved/` is
outside every enumerated scope, so nothing here is claimed to be an active
script. Promoting it to `scripts/` is filed as its own card.

## Source B — `/thearray/git/moe-mla-card-mageflow-continuation`

Branch `card/mageflow-continuation-step3952`, HEAD `e1fa07e`, unmerged (99 ahead
/ 154 behind main). These three files were the *entire* untracked content of
that worktree, and none was ever committed on any ref — including on that
branch, whose 99 commits are hostd/TrainVM C++ infrastructure and contain none
of them.

| file | sha256 | mtime |
| --- | --- | --- |
| `docs/experiment-vm/MAGEFLOW_CONTINUATION_STEP3952.md` | `dcb06ca1c4ef43fbf18fd90ddb9eb0b3e1289e9f5e7e49dd6665c987ee4ba474` | 2026-08-04 00:02 |
| `experiments/trainvm_mageflow_terminal_continuation_step3952.json` | `ff870fddbd5cbd30e17117113c4e53858b7a5976d70451bdac44587821e25f97` | 2026-08-03 22:12 |
| `experiments/trainvm_mageflow_terminal_continuation_step3952.input-roots.json` | `8d004f800e68ac90b2ba8ba5d31340d8fe85a6959a23b334352d73877596438a` | 2026-08-03 22:12 |

- `MAGEFLOW_CONTINUATION_STEP3952.md` — an operator runbook for resuming the
  routed photo/animation terminal-expert run from a copy-on-write snapshot of
  optimizer step 3952 without mutating the legacy checkpoint: current-head
  acceptance and native rebuild, boot-scoped GPU authority, worker deployment,
  cache reflink, content lock, submission, then a startup-gallery gate and an
  optimizer-progress gate.
- `trainvm_mageflow_terminal_continuation_step3952.json` — the TrainVM
  Experiment document it submits (`mageflow-terminal-continuation-step3952`,
  `rwkv-lab.mageflow-terminal-expert`, `resume_from …/checkpoint-00003952`,
  `max_steps: 12228`, `eval_on_resume: true`).
- `…input-roots.json` — the ten roots to content-hash when sealing it.

**Status: not superseded — strictly newer than anything on main.** The closest
document on main, `experiments/mageflow_terminal_legacy_restored_step2500.json`,
is a flat trainer config with the identical 82-key set; only five values differ,
all of them lineage/path advances (`resume_from`, the expert checkpoints,
`output_dir`, `encoder_cache_dir`). Every hyperparameter is byte-identical. The
step2500 run's own output directory ends at `checkpoint-00003952` with
`status.json` reporting `"state": "interrupted", "step": 3952` — so step 3952 is
that run's last coherent step and this document resumes from a snapshot of it.
Nothing on main covers terminal-expert continuation past step 2500, and nothing
on main is a TrainVM Experiment document for this family at all.

**It has not been run, and it is gated on the operator.** The runbook requires a
free machine ("only in the free-machine window", "after the machine is
released"), an explicitly reviewed journal identity after the 2026-08-03 reboot,
and a GPU UUID the operator chooses — the document ships the placeholder
`GPU-replace-with-reviewed-display-device-uuid` and says in as many words not to
copy it literally. `RUN_ID` is still a placeholder throughout.

## Source C — `.claude/worktrees/qwen36-multimodal-dpo`

Branch `card/qwen36-multimodal-dpo`, HEAD `af67529`, **merged**, which is what
made that worktree a cleanup candidate. Tracked separately as `card-74d0f9e5`.

| file | sha256 | mtime |
| --- | --- | --- |
| `docs/experiment-vm/QWEN36_DPO_LIVE_LAUNCH_HANDOFF.md` | `0d3076bba4ad3e60fe20d98bbefcda05717357085957f6d81039bf49e0121360` | 2026-08-06 11:39 |

269 lines, dated 2026-08-06 11:36 US/Central, never committed on any ref. Its
original location was the repository root; it is filed under
`docs/experiment-vm/` here because the root holds only `README.md`,
`TRAINING_LEVERS.md`, `CLAUDE.md` and `LICENSE`.

It is a live-launch handoff for a rank-256 cached-reference DPO LoRA run on the
frozen preference dataset and the baked caption model: branch state and
cherry-pick status, the exact sudoers-permitted install commands, the frozen
dataset and target-manifest digests, the run's hyperparameters, the `AuthorRun`
dry-run-then-commit call sequence, and seven completion checks ending "Do not
hide a failure behind the systemd wrapper's `active` state."

**Status: preserved; parts of it are stale; it is not authorization.** Its
opening records that "the user explicitly said start it" — on 2026-08-06. There
is no evidence the run was ever started. The entire "Current conflict state"
section is finished work: no cherry-pick or rebase is in progress, no conflict
markers remain, and HEAD moved on to `af67529` on 2026-08-07. Do not redo it.
The launch procedure and completion checks are not stale. An explicit
instruction three days old, relayed through a document, is not standing
authorization to start a long GPU run — see `card-74d0f9e5`.

### The uncommitted recipe edit is already safe

That worktree also carries an uncommitted `+7/-4` modification to
`docs/experiment-vm/examples/qwen36-caption-dpo-lora-r256.recipe-instance.v1.json`
which `card-74d0f9e5` flagged as unexplained and unpreserved. It is neither.
The modified working-tree file is **byte-for-byte identical to the version on
`origin/main` today** — both hash to
`67c3c58d5d0da9b7f861cbe210fcf662328a055b9113b9689ca865610444b3aa`. The change
landed in `e2a4163` ("Let a cooperative plan use a GPU that is driving a
display", PR #100); the worktree simply has a working-tree copy of a newer
version than its own HEAD.

Its content, for the record: two cosmetic reformattings (the `recipe` object
expanded across lines, `0.000005` rewritten as `5e-06`), a removed trailing
newline, and one substantive change — `data.qualitative_manifest_sha256` moved
from `sha256:e5cfb9fc…` to `sha256:9d4423f5…`, which is the digest main's
`hf-multimodal-sft.recipe-profiles.v1.json` also carries. Nothing is at risk and
nothing needs to be recovered.

## Verification

Each of the three source worktrees was left byte-for-byte unchanged; a SHA-256
manifest of every non-`.git` file was taken before and after and compared. No
file was moved or deleted, and no worktree was removed.

## The 32 modified tracked files (`card-247ac07a`)

The section above notes that `/thearray/git/moe-mla-parity-integration` also
held 32 modified tracked files — 1,536 insertions, 142 deletions — outside the
scope of that commit. They were in a worse position than the untracked files
were: uncommitted changes have no reflog, so unlike a deleted branch they are
not recoverable, and a single `git checkout .` would have ended them.

They are now on **`preserve/parity-integration-worktree-20260809`**, commit
`39efceb` (a second commit, `6748da0`, carries the same tree's untracked files
so the branch is a complete picture of that working tree). The branch is a
snapshot, **not a proposal to merge** — see the pins caveat below. It was built
with `git write-tree` against an alternate index, so the source working tree was
never modified; it remains dirty, registered, and byte-for-byte as it was.

### What the diff is

Three unrelated pieces of work sharing one tree:

1. **A ztok enhancement-search campaign** — `+267` lines extending
   `scripts/materialize_ztok_search_trial.py` and `+231` extending
   `src/rwkv_lab/trainvm_adapters/ztok_superposition.py`, with `+367` lines of
   new tests. Adds batch and train-shape schedules, `late_linear` LR decay with
   a cooldown fraction, late head/embedding untying, an explicit
   `source_byte_budget`, five named runtime levers, a SpectralMuon optimizer
   path, and an nsys GPU-trace path that drives the governed child trainer over
   a pair of pipes.
2. **A Qwen3.6 caption fine-tune adapter** — a 27th worker profile
   `rwkv-lab.qwen-caption-finetune` in the native contract, its
   `_qwen_caption_finetune` handler, a `qwen-caption` extra and console script,
   family predicates in the qualification and acceptance scripts, and a
   disposition catalog entry.
3. **A host-ledger authority snapshot** — `+170` lines in
   `trainvm/src/host_ledger.cpp` adding `authority_snapshot()`, one
   transactionally coherent view replacing five separate reads in
   `hostd_daemon_runtime.cpp`; `active_unlaunched_grants()`, which lets terminal
   release recovery reclaim a grant acquired but never launched; and a
   verification-revision cache so frequent status polls stop replaying the whole
   tamper-evident ledger per field.

### What is already on main, and what is not

Checked by content — grepping distinctive identifiers against
`git show origin/main:<path>` — rather than by ancestry, which this repository's
squash merges make unreliable.

**Already on main, verbatim:** the four `deploy/` files that carry the
unprivileged-hostd migration — `trainvm-hostd.service`,
`trainvm-controller.service`, `trainvm-gpu-fault-observer.service` and
`trainvm-hostd.json` — hash identically to `origin/main` today. Those 95 lines
of the diff are a working-tree copy newer than the worktree's own HEAD, the same
pattern the DPO recipe edit above turned out to be. `deploy/install-hostd-sudoers.sh`
is main's version plus four host-local grant lines.

**Not on main by any route:** everything else. No identifier from pieces 1, 2 or
3 appears anywhere on main. Six of the modified files do not exist on main *at
all*, including the entire ztok superposition adapter and its generator.

### The part worth acting on

The six `experiments/ztok_modded_search_20260804_q*.json` trial documents
preserved by the commit above are on main and **cannot run there**. All six
declare `source_byte_budget`, `runtime_levers` and `telemetry_interval_steps`;
main's `ZtokSuperpositionTrainConfig` accepts none of the three — and in fact
main has no `src/rwkv_lab/trainvm_adapters/ztok_superposition.py`, no
`scripts/materialize_ztok_search_trial.py`, and no `rwkv-lab.ztok-superposition`
adapter profile. This branch holds the implementation those documents describe.
`card-2005089a` records the missing design document; the missing implementation
is the larger half of the same gap.

### Why this is not a merge candidate as it stands

`integration/parity-candidate` is 166 commits ahead of main and 147 behind, and
most of this diff modifies files that exist only on that branch. It is not a
patch that can be rebased onto main; porting it means porting its base.

Its `trainvm/tests/source_disposition_catalog_tests.cpp` pins were computed
against that branch's catalog — 170 entries going to 171, a six-class role
split, `direct_gap` 0. Main's catalog today has 165 entries, a five-class split,
and `direct_gap` 1. The two lineages diverged rather than one lagging, so
**these pin values are wrong for main and must be regenerated, never carried
across.** The same applies to `source-dispositions.rwkv-lab.v1.json`.

One behavioural change needs a decision before any port:
`hostd_daemon_runtime.cpp` now hardcodes `ledger_verified = true` and drops the
`mutation_disabled_reason` branch for a failed ledger verification, because
`authority_snapshot()` throws instead of reporting. Whether an unverifiable
ledger should surface as a degraded status or as an exception out of the status
call is a real question, and the diff answers it by removing the degraded path.
