# Required final evaluation is a completion barrier

`worker.completed` is a proposal, not evidence that evaluation succeeded. The
controller may accept it only after the immutable invocation's finalization
policy has been satisfied by durable journal observations and one canonical
closure artifact with schema `rwkv-lab.final-evaluation.v1`.

This rule exists because the Qwen caption-distillation production run reached
its terminal checkpoint and published scalar/gallery material while all 674
required adapter generations failed. A trainer-level “done” receipt concealed
that result. Missing, partial, all-error, stale-step, wrong-checkpoint, and
context-poisoned evidence are now distinct non-complete verdicts.

## Registry-driven inventory

`FinalizationPolicyRegistry` enumerates the live adapter registry. It does not
pin a profile count. Every profile whose lifecycle is `stateful` must resolve
to exactly one operation policy, and every declared output must have exactly
one classification:

| Declaration | Finalization role | Completion requirement |
|---|---|---|
| checkpoint | terminal anchor | durable, nonempty, exact optimizer step |
| scalar result/report | scalar parent evidence | exact step and checkpoint bound when required |
| image gallery | example parent evidence | nonempty, full frozen membership, zero unresolved errors, exact step/checkpoint |
| semantic `eval_examples` | example parent evidence | canonical multimodal examples, full frozen membership, zero unresolved errors, exact step/checkpoint |
| test report | test parent evidence | full frozen membership, zero unresolved errors, exact step/checkpoint |
| opaque/other report | audit parent evidence | durable and nonempty when required |
| `rwkv-lab.final-evaluation.v1` report | semantic closure | controller completion authority |

An optional adapter output becomes a finalization requirement when the
immutable invocation declares that output. Required outputs are always
requirements. The inventory JSON reports `closure_output_name`; a missing value
is accompanied by `migration_pending: true`; that is an explicit adapter
migration gap, never evidence that finalization is optional. A migrated
stateful profile must declare exactly one closure output.

Scalar requirements are different from artifact outputs. The controller
derives their metric names, step domain, and final cadence from the immutable
training/evaluation composition. The closure lists those expectations, and the
controller cross-checks them against exact-terminal-step `metric.sampled`
journal events. A worker cannot satisfy a scalar requirement by writing a
report with a scalar-looking type name.

## Canonical closure

The canonical closure is a bounded immutable report. Existing galleries,
reports, and audits are parent artifacts; their type names or directories are
not opened as semantic authority. The closure contains:

- the registry policy digest;
- terminal optimizer step and immutable checkpoint identity/fingerprint;
- frozen member IDs, their digest and count;
- required parent-output receipts;
- scalar metric expectations for controller-side journal matching;
- resolved and failed counts;
- a cumulative append-only member ledger; and
- optional eval-only recovery evidence.

The closure does not assert its own artifact identity, fingerprint, journal
sequence, or durability. Those would be circular and worker-controlled. The
controller derives them from the accepted `artifact.published` event and wraps
the strictly decoded `FinalEvaluationManifest` in a separate
`FinalEvaluationReceipt`. Both reflected documents round-trip independently;
the semantic manifest rejects controller-owned envelope fields.
Untrusted closure bytes enter through `decode_final_evaluation_manifest()`
only. It rejects oversized input before JSON allocation, strictly decodes the
reflected schema, validates bounded semantic shape, and requires the original
bytes to equal the one canonical serialization.

The reducer receives a controller-derived `OperationFinalizationPolicy` and a
`FinalEvaluationExpectation`. The latter fixes the exact policy digest,
terminal checkpoint and optimizer fingerprints, canonical member list/digest/
count, and canonical scalar requirements before worker evidence is inspected.
The first worker receipt cannot shrink or otherwise establish any of these
sets. An optional closure port remains migration-pending; only a required
closure producer clears that state, and migration-pending operations can never
reduce to complete.

Each member ledger row binds a member ID and context digest to a monotonically
increasing attempt. A success carries a nonempty result digest; an error carries
a bounded error code. Published errors are never erased. A later valid success
resolves that member, so recovery preserves incident evidence while allowing a
good result to win.

The controller validates canonical closure bytes and requires the checkpoint
and every output receipt named by the closure to be durable prior artifacts of
the run. It independently checks required scalar observations. It does not
trust a worker's summary counts: the reducer recomputes them from the ledger.

## Eval-only recovery

Partial and all-error closures remain `finalization_pending`; malformed,
stale-step, wrong-checkpoint, changed-membership, rewritten-history, and
context-poisoned closures fail closed. A recovery receipt must:

1. resume the immutable terminal checkpoint at the same terminal step;
2. request every and only member still unresolved by prior durable history;
3. append records without rewriting prior rows; and
4. prove identical optimizer-state fingerprints before and after evaluation.

The before and after fingerprints must both equal the terminal optimizer
fingerprint already held by the controller. Equality between two worker-chosen
values is not evidence of immutability. Every retry appends exactly one record
for every and only currently unresolved member, in canonical member order, and
one new receipt for the selected evaluation output.

No training loop or optimizer update is permitted in recovery. Completion is
allowed only after the reducer finds every frozen member resolved, every
required output receipted, every required scalar durably sampled at the exact
terminal step, and the declared zero-unresolved-error policy satisfied.

The reducer emits a bounded `trainvm.finalization-verdict/v1` document. Its
`cause` and unresolved member list are the dashboard-facing explanation; a
generic worker exit code is not an adequate substitute.

Receipt count, per-receipt collections, aggregate records, aggregate collection
entries, and aggregate identity bytes all have hard limits. Canonical member,
scalar, initial-record, and initial-output ordering prevents distinct wire
encodings from representing the same semantic closure.
