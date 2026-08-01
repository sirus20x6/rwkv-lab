# Data, evaluation, and artifact authority

Status: implementation contract for dataset, cache, evaluation, benchmark, and publication ownership

Scope: model-family-neutral acquisition, inventory, materialization, review, cache reuse, evaluation,
export, promotion, retention, and deletion

This contract complements [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`HOST_RESOURCE_AUTHORITY.md`](HOST_RESOURCE_AUTHORITY.md). TrainVM owns logical workflow and
artifact authority; hostd owns physical resources and launched processes. A worker may produce bytes,
but bytes do not become a reusable artifact until TrainVM validates and records an immutable
publication receipt.

## Decision

Treat data, evaluations, caches, and exports as typed, content-bound artifacts rather than paths
discovered by convention. Every mutable external-effect phase writes only to an attempt-scoped
staging namespace. Every reusable result is promoted through an authority-owned validation and
publication transaction.

The design separates six concepts that the legacy workflows often collapse:

1. **acquisition state** records resumable interaction with mutable or remote sources;
2. **inventory revisions** freeze observed membership and source identity without claiming every
   payload has been downloaded or licensed for every use;
3. **materializations** deterministically select and arrange content from one sealed inventory;
4. **cache qualifications** state exactly how strongly cached bytes are bound to inputs and producer;
5. **evaluation evidence** binds held-out inputs, generated outputs, metrics, and comparisons;
6. **publication state** records immutable promotion, aliases, retention, quarantine, and deletion.

An existing file, successful exit, row count, filename, symlink, dashboard card, or legacy manifest
is evidence only. None of them independently grants reuse, promotion, export, or deletion authority.

## Authority split

| Concern | Authority | Worker role |
|---|---|---|
| source request, inventory revision, and use policy | TrainVM | enumerate or fetch under a registered operation |
| network, credentials, and provider budget | registered operation plus site policy | use sealed secret references; report spend and responses |
| staging files and partial downloads | current node attempt | write only within the granted staging root |
| immutable artifact publication | TrainVM artifact authority | submit a candidate manifest and terminal receipt |
| cache construction | registered producer operation | build bytes and report geometry/content evidence |
| filesystem and signal effects | TrainVM/hostd through a typed operation | never infer a handoff by polling or send an undeclared signal |
| review decisions and scoring policy | authenticated human or registered scorer | submit an append-only decision or score artifact |
| evaluation generation and metrics | registered evaluation operation | publish bounded outputs against sealed held-out inputs |
| promotion and aliases | TrainVM policy plus explicit operator/decision artifact | never rewrite a parent artifact or serving alias directly |
| retention, quarantine, and deletion | TrainVM storage authority | release handles and acknowledge terminal cleanup |
| physical process and accelerator ownership | hostd | consume the exact active host grant |

The Go dashboard is a protocol client. It may request review, promotion, alias, retention, or deletion
commands and render their receipts; it does not mutate manifests, move files, signal workers, or
decide that a path is safe to reuse.

## Mandatory invariants

1. Mutable staging and immutable publication use different roots and different artifact states.
2. A published artifact has one canonical manifest digest and never changes in place.
3. A path is never an artifact identity; identity includes canonical manifest and content evidence.
4. Acquisition never silently becomes deterministic materialization.
5. A partial download is never represented as the completed object's content hash.
6. Storage exhaustion, quota pressure, and truncated sources are structured terminal or resumable
   states, not successful short datasets.
7. License and use constraints travel with every derived artifact and can only become equally or more
   restrictive unless a new reviewed attestation explicitly permits otherwise.
8. Cache reuse requires the consumer's declared qualification level; presence alone is never enough
   when producer or content identity matters.
9. Overlay and handoff operations cannot modify their immutable base inputs.
10. Human review, scoring, compaction, trash, restoration, and policy publication are append-only
    decisions over immutable revisions.
11. Eval galleries publish generated, target, and held-out identities together; mutable preview files
    never appear in historical scrubber state.
12. Export always creates a new immutable bundle and never overwrites a checkpoint, adapter,
    compressor, tokenizer, dataset, or prior export.
13. Promotion changes a decision or alias pointer, never the promoted artifact's bytes or manifest.
14. Deletion is explicit, referentially checked, receipted, and recoverable before irreversible purge.
15. Benchmark promotion requires isolated baseline/candidate evidence and parity for every required
    declared bucket, not only a favorable aggregate.

## Typed artifact model

C++26 reflection may generate descriptors and strict decoders, but persisted records use explicit
schema versions and canonical serialization. The sketches below define semantic requirements rather
than final field tags.

```cpp
enum class ArtifactState {
  staging,
  candidate,
  published,
  quarantined,
  tombstoned,
  purged,
};

enum class ArtifactSensitivity {
  public_data,
  internal,
  restricted,
  secret_derived,
};

enum class Completeness {
  complete,
  partial_resumable,
  partial_terminal,
  unknown,
};

struct ContentObject {
  std::string logical_name;
  std::string media_type;
  std::uint64_t size_bytes;
  std::string sha256;
  std::optional<std::string> tree_path;
};

struct ArtifactManifest {
  std::string api_version;
  std::string artifact_id;
  std::string artifact_type;
  std::string schema_version;
  ArtifactState state;
  ArtifactSensitivity sensitivity;
  Completeness completeness;
  std::string producer_operation_digest;
  std::string producer_attempt_id;
  std::string plan_digest;
  std::vector<std::string> parent_manifest_digests;
  std::vector<ContentObject> objects;       // canonical logical-name order
  std::string use_policy_digest;
  std::map<std::string, std::string> attributes;
  std::string canonical_manifest_digest;
};
```

Tree artifacts use normalized relative paths. Absolute paths, `..`, empty segments, non-normalized
Unicode names, device nodes, sockets, FIFOs, and symlinks are rejected unless an artifact type has a
separately audited link contract. Publication walks already-open directory descriptors without
following links, hashes regular files, verifies sizes, and atomically makes the completed tree
read-only or moves it into an authority-owned immutable store.

The content list may be represented by a Merkle tree for large artifacts. The root digest, leaf
ordering, normalization rules, chunking algorithm, and manifest schema are part of the identity.
Filesystem metadata such as mtime, inode, hardlink count, UID, or staging location is not content
identity.

## Acquisition and inventory

### Stateful acquisition

Acquisition covers remote listing, paginated APIs, archives, object stores, crawling, captioning
providers, and any source whose observations can change between calls. It is a stateful process
operation with explicit external effects. Its checkpoint must include, as applicable:

- canonical request specification and provider/model/revision identity;
- source terms/use-policy snapshot and the attestation that selected the allowed purpose;
- ordered pagination, shard, archive, or item cursor;
- per-item source identity, request digest, response metadata, retry count, and disposition;
- append-only spend and rate-limit accounting;
- completed byte ranges and partial-object identities;
- rejected, duplicate, unavailable, malformed, and policy-blocked records;
- staging tree digest or durable object receipts;
- RNG state where sampling affects membership;
- storage reservation, bytes used, low-space threshold, and terminal reason.

Resuming acquisition revalidates the checkpoint's source/request identity and reconciles each item
receipt. It does not repeat a billable request or overwrite a completed object merely because the
staging filename exists. A changed remote listing creates a new inventory revision or an explicitly
versioned continuation; it never mutates a sealed revision.

### Inventory revision

An inventory is an immutable ordered observation table. Each row has a stable item key, source
locator digest, observed metadata, disposition, optional completed payload digest, optional caption
or annotation references, constraints, and error evidence. The revision records:

- source and query identities;
- ordering and tie-break rules;
- row schema and canonical serialization;
- row count by disposition;
- payload-complete count and bytes;
- duplicate/equivalence relationships without silently deleting members;
- acquisition attempt and checkpoint lineage;
- content digest for the inventory table itself.

An inventory may truthfully contain unavailable or incomplete rows. A downstream materializer must
declare which dispositions it accepts. `downloaded`, `complete`, `captioned`, `eligible`, and similar
terms are closed enum values defined by the inventory schema, not arbitrary source strings.

### Deterministic materialization

Materialization consumes only sealed inputs: one or more inventory revisions, a typed selection
policy, required annotations, and an output-layout contract. It is `replay_safe` and stateless only
when membership and output bytes are a pure function of those inputs.

The materialization receipt records:

- inventory manifest digests and accepted dispositions;
- canonical filters, joins, sorting, tie breaks, deduplication, sampling algorithm, and seed;
- exact selected item keys in final order or a content-addressed selection manifest;
- source payload and annotation hashes for every selected item;
- link/copy/re-encode mode and output logical names;
- expected and observed counts, bytes, extensions/media types, geometry buckets, and rejection reasons;
- output tree digest and validation report.

Hardlinking and copying are storage implementations, not identity. A hardlinked source must already
be immutable under authority or must be copied into immutable publication before the result can be
trusted. Re-encoding is a distinct producer operation with codec/version/options in its identity.

### Storage pressure and partial downloads

Before acquisition or materialization dispatch, TrainVM reserves a bounded staging budget and checks
site watermarks. The worker emits monotonic byte/accounting progress. Crossing a soft watermark may
stop scheduling new objects; crossing a hard watermark requests a safe checkpoint and terminates
with `storage_pressure`, preserving resumable state.

A partial object identity contains the remote object key, request identity, observed version/ETag or
last-modified evidence, expected length when known, verified byte ranges, per-range hashes, and
temporary object ID. Resume is allowed only if the remote identity still matches. Otherwise the
partial is quarantined and a fresh object attempt begins. Multipart completion computes the final
content hash from completed bytes before changing disposition to complete.

Unknown length, missing ETag, server-side transformation, or a provider that cannot honor ranges
does not make resume impossible, but it makes the operation restart that object rather than append to
uncertain bytes. Quota exhaustion, provider budget exhaustion, permission denial, and source removal
are separate terminal reasons and remain visible in inventory statistics.

## License, consent, and use constraints

TrainVM records policy evidence; it does not manufacture legal permission. Every source and artifact
has a typed use-policy reference with:

- policy/terms document digest and observed effective date;
- provenance category and source/provider identity;
- declared allowed purposes such as research evaluation, internal training, redistribution, or
  commercial deployment;
- redistribution and derivative constraints;
- attribution, notice, deletion, geographic, age/sensitive-content, and access restrictions;
- consent or data-governance attestation reference where required;
- reviewer/authority, decision time, reason, and expiry/review time;
- sensitivity and audience labels.

Policy evaluation is fail-closed for a requested purpose. An unknown or expired decision cannot be
treated as unrestricted. A derived dataset, cache, gallery, trace, model bundle, or benchmark report
inherits the intersection of parent constraints. Redacted summaries may receive a different policy
only through a registered transformation and reviewed policy decision.

Credentials, access tokens, cookies, and raw secret values never enter manifests or events. They are
runtime secret references. Provider response metadata is redacted before publication when it may
contain credentials or private user data.

## Cache qualification and reuse

Cache identity includes its semantic type, ordered source inputs, transforms, geometry, dtype,
layout, producer, runtime dependencies where relevant, and qualification evidence. Consumers declare
the minimum accepted level:

| Level | Meaning | Evidence | Permitted claim |
|---|---|---|---|
| `unqualified` | legacy or staging bytes only | path/file observation | diagnostic display only |
| `presence_geometry` | expected members, shapes, dtypes, lengths, and ordering exist | complete inventory plus structural validation | consume only when producer/content equivalence is irrelevant and plan allows this level |
| `producer_bound` | geometry plus exact registered producer, code, configuration, parents, and dependency identities | sealed producer receipt and input manifests | reuse for the same declared producer contract |
| `content_verified` | producer-bound plus per-object or Merkle content integrity and semantic spot/full parity required by the cache type | immutable hashes and qualification report | authoritative cross-run reuse within matching policy and compatibility rules |

Qualification is monotonic only through a new receipt; it never edits the cache manifest. A consumer
requiring `content_verified` cannot downgrade because rebuilding is inconvenient. Same dimensions,
row count, filenames, or source paths do not prove producer or content identity.

Cache-type descriptors define semantic validators. Examples include source membership/order, tensor
geometry and dtype, finite values, image decode and dimensions, tokenizer/version compatibility,
feature-to-source alignment, bucket coverage, padding/mask behavior, and bounded numerical parity
against direct computation. Sampling-based parity records its deterministic sample manifest and does
not claim full content equality.

### Overlay and handoff

An overlay is a new immutable artifact with an immutable base-manifest parent and an ordered set of
replacement/addition objects. Resolution rules, collision policy, tombstones, and the resolved tree
digest are part of the overlay identity. Builders write a fresh staging tree; they never patch the
base cache or repoint a shared symlink.

A cache handoff is a graph transition caused by a validated publication event, not by `pgrep`, log
text, a status-file string, directory existence, or a magic exit code. The handoff receipt binds the
producer attempt, cache manifest, qualification level, coverage/span, consumer node, and plan
revision.

Legacy workers that require a filesystem rewrite or Unix signal use separate registered effect
operations:

- `filesystem_projection` materializes a consumer-private view under an allowed root, with
  before/after tree receipts and no mutation of immutable parents;
- `worker_signal` validates the exact active attempt/process receipt through hostd, names one
  allowlisted semantic command, and waits for a typed acknowledgement;
- `legacy_status_import` may observe a file but cannot authorize publication or transition without
  authority validation.

All three are explicit in `trainvm plan`, have idempotency classes, and are rejected in an
exact-recovery graph unless their adapter contract supplies the required reconciliation evidence.

## Human review, scoring, and policy publication

### Review log

Review operates over an immutable review-pack manifest. Each action appends a signed/authenticated
decision containing review-pack digest, item ID, prior decision revision, action, optional bounded
reason code, reviewer identity, and command idempotency key. Concurrent stale decisions are rejected
or retained as explicit conflicts; last-writer-wins is not an authority rule.

Review actions may label, rate, accept, reject, quarantine, restore, or request exclusion. They do
not edit source files, captions, scores, or manifests. Bulk actions record an explicit ordered item
selection artifact so the command is replayable.

### Recoverable trash and restoration

A review `delete` request first creates a quarantine/tombstone decision and moves or projects the
mutable working copy into an authority-owned trash namespace. The receipt binds the item/content
hash, source and trash locations, reason, review decision, retention deadline, and reverse operation.
The immutable source or published artifact remains addressable for lineage and is not unlinked.

Restoration appends a receipt and creates a new working projection. Irreversible purge occurs only
after the retention deadline, reference and legal-hold checks, a separate explicit authorization,
and a verified purge receipt. A failed or partial purge remains quarantined and cannot be reported as
purged.

### Manifest compaction

Compaction creates a new immutable manifest that applies accepted decisions and tombstones to a named
parent revision. It records included/excluded item IDs, ordering, decision-log digest, policy, counts,
and parent lineage. It does not rewrite the parent manifest or erase rejected rows. Deterministic
compaction of the same inputs produces the same manifest digest.

### Scoring and classification

Automated quality, aesthetic, safety, routing, color, caption, or other scores are separate columnar
artifacts keyed by item identity. Their manifest binds model/checkpoint, code, preprocessing,
precision, hardware-relevant numerical profile, label/score schema, and source inventory. Missing,
errored, or low-confidence results remain explicit dispositions.

Combining scores with an inventory is a deterministic join/materialization. A scorer cannot silently
change dataset membership or overwrite a human label. Human decisions, source annotations, model
scores, and derived policy buckets remain distinct namespaces with declared precedence.

### Policy publication

A selection/routing/review policy is an immutable versioned artifact containing typed predicates,
thresholds, precedence, target purpose, input schemas, author/reviewer, evaluation evidence, and
effective scope. Publishing a policy requires a semantic diff from its parent and an explicit
decision receipt. Applying it creates a new materialization; it never retroactively changes a prior
dataset or run.

## Evaluation and gallery publication

Evaluation consumes immutable checkpoint/model, evaluator, held-out dataset, prompt/template,
sampling, precision/runtime, and seed manifests. Held-out membership is never inferred from the
training directory at eval time. Split-audit evidence binds train and held-out manifests and records
identity, exact-content, and any declared near-duplicate checks.

Workers may create mutable previews beneath an attempt-scoped gallery staging root. The dashboard may
show them as **live/unpublished**, but they are not historical evidence and disappear when the
attempt is reconciled. Publication validates decodability, size limits, expected pair cardinality,
and policy before creating an immutable gallery revision.

```cpp
struct EvalGalleryItem {
  std::string item_id;
  std::string heldout_item_id;
  std::string heldout_manifest_digest;
  std::string prompt_or_condition_digest;
  std::string generated_object_sha256;
  std::string generated_object_uri;
  std::optional<std::string> target_object_sha256;
  std::optional<std::string> target_object_uri;
  std::optional<std::string> source_object_sha256;
  std::optional<std::string> source_object_uri;
  std::uint64_t seed;
  std::map<std::string, std::string> sampling_attributes;
};

struct EvalGalleryManifest {
  std::string api_version;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::uint64_t step;
  std::string step_domain;
  std::string checkpoint_manifest_digest;
  std::string evaluator_profile_digest;
  std::vector<EvalGalleryItem> items;
  std::string use_policy_digest;
  std::string canonical_manifest_digest;
};
```

Generated, target/original, and held-out identities are mandatory as applicable. A target path
without a target content hash is not a valid side-by-side pair. Captions, prompts, masks, controls,
audio, or video conditioning use typed sibling artifacts and hashes rather than unbounded strings in
the gallery manifest. Sensitive text may be encrypted/restricted while its digest remains bound.
Object URIs are locators, never identity or read authority: the dashboard resolves only `file:`
objects beneath configured data roots and verifies the corresponding SHA-256 before returning bytes.
The published artifact fingerprint similarly binds the exact gallery-manifest bytes. Moving an
object therefore requires republishing a new immutable revision even when its content hash is stable.

Each published revision is append-only and indexed by `(run, eval node, step domain, step, attempt)`.
The scrubber reads these revisions, never rescans run directories. Re-evaluation at the same step
creates another revision with its evaluator/attempt identity; it does not replace history. Metric
artifacts reference the exact gallery/held-out/checkpoint manifests used for their computation.

## Immutable export

Export is a registered verification-and-publication graph, not a copy button. It consumes immutable
parent artifacts and an export profile, verifies policy and compatibility, writes a fresh staging
bundle, validates it in an isolated reader, and atomically publishes a new manifest.

Every export binds, when applicable:

- source checkpoint, architecture/topology/freeze map, optimizer-stripping decision, and precision;
- tokenizer/processor/template/config and custom-code identities;
- adapters, routing experts, reward/process heads, conversion patches, or representation components;
- training/eval dataset and promotion-decision manifests required by site policy;
- dependency/runtime requirements and safe loading format;
- file hashes, tree digest, size, use policy, notices, and a verification report;
- parent and promotion lineage.

No export operation overwrites, edits, or adds files to an existing source or bundle. Output names
are presentation metadata; the manifest digest is identity. Symlinks and external absolute
references are rejected unless fully resolved and copied under an explicitly audited format.

### Representation and vision-compressor export

A representation compressor, projection head, native vision head, or similar portable component is
an exportable artifact only when the bundle additionally binds:

- source teacher/representation cache manifests and their qualification levels;
- compressor architecture/configuration and exact weights;
- input normalization, image/sequence geometry, dtype, layout, padding/mask, and output contract;
- training parent, baseline, selected/best checkpoint, and transfer-evaluation manifests;
- reconstruction, relational, caption/grounding, or other declared parity gates;
- an isolated load-and-forward validation over a sealed sample manifest.

A convenient `state_dict` or file found in a run directory is not an export. Failed validation keeps
the candidate in staging or publishes restricted failure evidence; it cannot receive a production
alias.

## Run evidence, promotion, aliases, and retention

### Run-evidence bundle

A terminal run publishes one evidence manifest that references rather than copies all durable facts:

- experiment source and canonical plan;
- adapter registry lock, code, executable, dependency, and host-profile identities;
- node attempts, external-effect receipts, resource grants, controls, and terminal states;
- input inventories/materializations, caches and qualification receipts;
- checkpoints, metrics, galleries, traces, benchmark/parity reports, and exports;
- policy/use decisions, warnings, failures, rejected candidates, and promotion receipts.

Missing expected evidence remains an explicit incomplete field. `Completed` lifecycle state does not
by itself mean `qualified`, `promotable`, or `exportable`.

### Promotion and aliases

Promotion is an append-only decision over exact candidate and evidence manifest digests. The receipt
contains policy/version, gate results, reviewer or automatic decision identity, comparison baseline,
reason, and rollback target. Promotion never modifies the candidate.

Human-readable aliases such as `candidate`, `best`, `production`, or a task-specific channel are
authority-owned, revisioned pointers to immutable manifest digests. Updating an alias requires its
expected prior revision, a promotion receipt, and policy compatibility. Rollback is a new alias
revision pointing to an earlier immutable artifact. Filesystem symlinks may be generated as
non-authoritative projections after the alias transaction commits.

### Retention and deletion

Retention policy is evaluated over the artifact reference graph, not filename age. Policies may keep
latest aliases, promoted artifacts, best checkpoints, failure evidence, legal holds, or a bounded
history, but every policy evaluation publishes a candidate action set before mutation.

Deletion proceeds through:

```text
requested -> reference_checked -> quarantined -> grace_elapsed -> purging -> purged
                         \-> blocked
```

Reference checking includes parent/child lineage, aliases, active runs, resumes, exports, legal/use
holds, and open worker handles. Quarantine is reversible and records exact objects. Purge uses
descriptor-relative, no-follow operations within an authority-owned root, verifies each target
against the quarantine receipt, and publishes bytes/files removed plus failures. Shared CAS objects
are removed only when their authoritative reference count is zero. Journal manifests, tombstones,
and purge receipts remain even when payload bytes are gone.

## Isolated benchmark map and parity decision

Performance qualification is a typed map/reduce graph. It never benchmarks a candidate inside the
training process that produced it and never compares profiled candidate timing with unprofiled
baseline timing.

The **map** expands a sealed benchmark matrix over required shape/geometry buckets, dtypes,
precision modes, device/topology profiles, and seed/input samples. Each baseline and candidate cell
runs in a fresh isolated attempt with:

- identical immutable checkpoint/input/cache manifests;
- exact adapter, runtime, compiler, device, and dependency identities;
- declared cold compile and disposable warmup phases;
- bounded repetitions and synchronization/timing protocol;
- output, gradient/update/state/resume parity fields appropriate to the operation effect;
- end-to-end latency/throughput, memory, input stall, compile, and warmup evidence;
- raw evidence and a terminal cell receipt, including failures and out-of-memory results.

The **reduce** verifies matrix completeness and first evaluates parity/non-regression per required
cell. Only parity-passing cells contribute to speed aggregation. It reports distributions and tail
latency, not only a mean. A machine-readable decision contains:

- matrix/input/baseline/candidate manifest digests;
- required, completed, failed, and excluded cells with reasons;
- per-cell parity thresholds and results;
- paired performance deltas and confidence rule;
- memory and quality gates;
- adopted/rejected/inconclusive disposition;
- scope of adoption, because a candidate may qualify for only a subset of buckets;
- decision policy/reviewer and immutable receipt digest.

Any missing required bucket, parity failure, changed input, mixed machine profile, or policy mismatch
makes the overall decision fail or narrow its explicit scope. It cannot be hidden by a faster
aggregate. Promotion updates a versioned adapter/optimization alias; it never rewrites benchmark
history.

## External-effect protocol

All state-changing filesystem, network, provider, review, alias, quarantine, and purge operations use
the journal intent/receipt protocol:

1. validate policy, active logical lease, and any required host grant;
2. commit an intent with canonical request digest and idempotency key;
3. create or reconcile an attempt-scoped staging/effect record;
4. perform the bounded effect through a registered operation;
5. validate output and commit its immutable receipt;
6. publish or transition only after the receipt is durable.

`replay_safe` operations must prove deterministic outputs. `receipt_required` operations reconcile
provider request IDs, partial objects, filesystem moves, or aliases before retry. `at_most_once`
operations stop for operator reconciliation after dispatch uncertainty. Network spend, destructive
purge, and human-facing external publication are never guessed successful from timeout or path
existence.

## Required implementation surfaces

TrainVM must provide reflected descriptors and protocol types for:

- source/use-policy, acquisition checkpoint, inventory, materialization, and selection manifests;
- artifact candidate validation and immutable publication;
- cache geometry/producer/content qualification and overlay resolution;
- filesystem projection, semantic worker command, and legacy observation effects;
- review action, quarantine/restore, compaction, score column, and policy publication;
- split audit, evaluation metric, gallery staging/publication, and run-evidence manifests;
- export and isolated export-validation receipts;
- promotion decision, alias revision, retention plan, quarantine, and purge receipts;
- benchmark matrix/cell/parity/reduction decision.

The dashboard projections must show artifact state, completeness, qualification level, sensitivity,
use restrictions, lineage, aliases, retention holds, and terminal errors. Staging/live previews are
visually distinct from immutable history. Every destructive or externally publishing command shows
its exact scope from the compiled plan or retention action set.

## Acceptance scenarios

1. Interrupt a paginated acquisition after a partial object, resume without repeating completed
   billable requests, and reject the partial when the remote object identity changes.
2. Exhaust the staging soft and hard watermarks; preserve an honest resumable checkpoint and never
   publish the shortened inventory as a complete dataset.
3. Materialize the same sealed inventory and selection policy twice and obtain identical membership,
   ordering, captions/annotations, and tree digest.
4. Mutate bytes at a reused source path and prove producer-bound/content-verified cache reuse fails
   while presence/geometry evidence alone cannot satisfy the consumer.
5. Build an overlay, crash during filesystem projection, and recover without modifying its base or
   advancing the consumer before a complete handoff receipt.
6. Attempt to signal a stale/reused PID and prove hostd identity validation rejects it.
7. Append review labels, quarantine without confirmation friction where site policy permits, undo the
   action, compact a new manifest, and retain the immutable decision history.
8. Publish automated scores and prove they neither overwrite human decisions nor alter membership
   until an explicit policy/materialization node consumes them.
9. Attempt a training/export use with unknown, incompatible, or expired source constraints and fail
   before dispatch.
10. Crash while publishing an eval gallery; the scrubber shows either the prior immutable revision or
    the complete new revision, never half-written pairs.
11. Re-evaluate one checkpoint/step and retain both evaluator revisions with generated, target, and
    held-out hashes.
12. Export a checkpoint/adapters and a representation compressor into fresh bundles, validate them in
    isolated readers, and reject overwrite, external symlink, or missing-parent attempts.
13. Promote and roll back through alias revisions while candidate/export bytes and historical run
    evidence remain unchanged.
14. Plan retention with shared parents and CAS objects, quarantine a leaf, block a referenced parent,
    restore the leaf, and later purge only zero-reference payloads with a complete receipt.
15. Run a benchmark matrix with one faster but parity-failing bucket and prove the candidate is not
    globally promoted; repeat with a declared narrowed scope and retain both decisions.
16. Rebuild every artifact, gallery, alias, retention, and benchmark projection from the immutable
    journal and obtain the same visible state.

## Migration order

1. Land schemas, canonical hashing, staging/published roots, and fake artifact-authority fixtures.
2. Import legacy datasets, caches, galleries, and exports as `unqualified` observations; do not grant
   authority based on their current paths.
3. Wrap representative inventory/materialization, cache, gallery, and export operations and publish
   validation receipts without changing existing trainers.
4. Move cache handoffs and gallery scrubber discovery from polling/path conventions to typed events
   and manifests.
5. Add review/quarantine/compaction, policy, promotion/alias, and retention commands through TrainVM.
6. Add isolated benchmark map/reduce and require its decision receipt before optimization-profile
   promotion.
7. Disable legacy path discovery and direct dashboard filesystem mutation for active VM-owned runs.

No migration step grants launch, reuse, export, or deletion authority merely because a legacy
workflow was imported successfully.
