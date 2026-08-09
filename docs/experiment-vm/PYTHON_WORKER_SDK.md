# Python worker SDK

Tensor and model execution remains in Python, while TrainVM owns experiment
compilation, scheduling, state transitions, persistence, host resources, and
process authority. The worker SDK is the narrow typed bridge between them.

## Two-stage launch contract

An authority-launched worker receives a sealed `trainvm.worker-bootstrap/v1`
descriptor. It contains only:

- the TrainVM Unix-socket target;
- run, node, attempt, launch, lease, and fencing identity;
- the exact adapter version and code fingerprint;
- the sorted capability set independently bound to the sealed worker profile,
  plus a replay cursor.

It contains no model path, dataset path, optimizer setting, learning-rate
setting, shell fragment, environment override, or secret. After the worker
opens `WorkerControl`, TrainVM returns a content-addressed
`trainvm.worker-invocation/v2` document in `WorkerWelcome`. That immutable
invocation contains the resolved inputs, controls, artifact declarations, and
training-component composition selected by the authority.

Both documents use exact canonical JSON, a 16/48 KiB bound, and SHA-256. The
Python and C++ implementations share golden digests, reject duplicate members,
reject noncanonical encodings, and verify all Welcome/bootstrap identity joins.
The Python decoder recursively freezes the invocation before adapter code sees
it. A non-null training composition is also decoded into
`ResolvedTrainingComposition`: it independently verifies the composition and every descriptor
digest, enforces the exact slot/envelope shape and model-family binding, and exposes
`require(slot, category=...)`. The returned `ResolvedTrainingComponent.runtime_envelope()` is the
only object passed to the category-owned optimizer, LR-schedule, router, activation, or other
tensor-runtime factory. This keeps authority resolution, adapter topology integration, and tensor
implementation as separate layers.

### Adding a field to a shared envelope

The exact-field checks on these envelopes are deliberate: an unrecognised key
inside a digest-bound object is drift, and refusing it is the point. That makes
adding a field to one side a contract change, and there are two correct ways to
make it.

1. **Bump the envelope's `api_version`** and let the reader select its field set
   by version. The worker invocation does this for `resume`, which
   `adapter_invocation.cpp` attaches only for non-legacy documents and
   `invocation.py` expects only for the matching version.
2. **Teach the reader in the same change** — add the field to its known-optional
   set, carry it onto the decoded object rather than dropping it, and add a test
   that fails when the two sides disagree.
   `tests/test_trainvm_resolved_composition_parity.py` is the reference: it reads
   `composition_body()` out of the registry source and fails naming any key the
   Python side does not know.

Relaxing a check to ignore unknown keys is not one of them. It fixes the symptom
and re-arms the trap for the next field.

This is written down because it was got wrong. The native side added `topologies`
to the resolved-composition envelope and the Python worker, which compared the
key set exactly, refused every composition carrying it — so an experiment
selecting a research topology compiled, resolved, reached the worker and died
there. It stayed broken across several merges with CI green, then a second field
was added the same way before anyone noticed.

Nothing caught it because every test on each side used a fixture built by that
same side: the native tests asserted on emitted JSON, the Python tests on
hand-written envelopes. **A boundary is only tested by something that crosses
it.** A fixture the reader wrote agrees with the reader by construction.

## Session API

`rwkv_lab.trainvm_worker.WorkerSession` provides:

- one Hello-first bidirectional gRPC stream;
- replay-aware worker/controller sequence checks;
- typed heartbeats and scalar metrics;
- complete artifact-manifest publication;
- immutable compile/warmup requests and fenced execution-phase receipts;
- typed pause, resume, checkpoint, cancel, and control commands;
- explicit control acknowledgement at adapter-selected safe points;
- one canonical terminal result and durable receipt.

`rwkv_lab.trainvm_worker.execution_phases` owns the worker side of compile and warmup. It verifies
that every protobuf request exactly matches the sealed invocation and request digest. The shared
runtime captures a caller-supplied complete training-state identity before and after the phase,
counts warmup steps through an explicit completion callback, publishes partial progress and a
diagnostic on failure, and refuses a successful receipt unless disposable execution restored the
same state. The identity must cover model, optimizer, RNG, scaler, data cursor, schedule segment,
and effective controls; adapters cannot substitute a model-weights-only checksum. Repeating a
phase after a lost connection creates another independently fenced receipt, so recovery never has
to guess whether an unacknowledged phase ran.

A phase is cancellable while it runs. `WorkerSession` records a `CancelCommand` as a sticky
`execution_phase_cancellation` reason at the moment the receive thread decodes it, *and* still
queues the command for the adapter — the phase must be able to see a cancellation while the adapter
is inside a blocking phase and has not reached its next poll point, without stealing the command
the adapter owes a lifecycle acknowledgement for. The runtime consults that reason before starting
and after each bounded step, so cancellation always lands on a completed step boundary. It then
publishes a `DISPOSITION_CANCELLED` receipt naming how far the phase got, and raises
`WorkerExecutionPhaseCancelled`, which is deliberately not a `WorkerExecutionPhaseError`: nothing
was violated, so an adapter catching phase errors to report a defect must not swallow it. The
authority holds a cancelled receipt to the same bounds as a completed one — never more steps than
the request declared, always a restored trajectory — and additionally requires diagnostics, because
a partial step count in the journal is otherwise unattributable. A disabled phase can only ever be
skipped; there is nothing running to cancel.

Heartbeats carry two phase fields for two different jobs. `phase` stays a free-text operator label
(`"train"`, `"checkpointing"`, `"verifying_inputs"`). `execution_phase` is the typed one: pass it
while inside an authority-requested compile or warmup and the journal binds the heartbeat to that
request. The authority refuses a typed phase this attempt did not request or that is declared but
disabled, and refuses a `phase` label of `"compile"` or `"warmup"` that the typed field does not
agree with — without that second rule a worker could report progress on a compile the authority
never authorized, and every downstream reader of the journal would show it. `WorkerObservability`
threads `execution_phase` through `optimizer_step` and `keepalive`.

`WorkerExecutionPhases` is the one-shot adapter coordinator. It fails the operation if any declared
phase is omitted or receipted twice. The scratch-RWKV adapter is the first production consumer: its
compile phase triggers Torch's lazy compiler with a disposable forward/backward, and its warmup
phase executes exactly the requested number of disposable workloads. Before and after identities
come from `torch_trajectory_state`, which hashes every model, gradient, buffer, and optimizer-state
tensor in bounded chunks plus optimizer groups, all Torch/CUDA/Python/NumPy RNG state, and the data
cursor. It deliberately hashes tensor contents because `Tensor.data` writes can evade version
counters.

All three MageFlow training adapters use the same coordinator through the family bridge in
`trainvm_adapters.mageflow_phases`. The invocation's compile declaration owns both regional
transformer and VAE compilation, a deterministic real training batch triggers lazy compilation,
and warmup repeats that disposable forward/backward exactly as declared. The terminal/TREAD route
also exercises its configured REPA, immiscible-flow, loss-weighting, direction-loss, and loop
objectives and fingerprints auxiliary modules alongside the transformer. The bridge preserves every
Python/Torch/CUDA RNG stream, clears gradients, fingerprints the complete model/optimizer plus
schedule/control/data-cursor identity, and reports a changed trajectory as a failed phase rather
than relying on the controller to reject a false success. When a frozen-encoder cache is selected,
the chosen phase batch must already have complete text and VAE-moment coverage; compile/warmup never
hide cache population as a disposable side effect.

Before dispatch, `rwkv_lab.trainvm_worker.runtime_policy` independently parses the resource policy
from the sealed invocation. It owns only non-root runtime controls: narrowing/rechecking process CPU
affinity, tensor-library/OpenMP thread counts, the preprocessing-worker hint, and effective nice
verification. Family adapters do not read environment variables or reconstruct this policy.
CPU/I/O cgroup weights are hostd authority and are never reported effective by this worker layer.

`rwkv_lab.trainvm_worker.observability` independently owns the sealed observability declaration.
It strictly decodes heartbeat cadence and the selected metric catalog, freezes metric units and
step domains, bounds labels/scalars, and publishes only names selected by the experiment. The three
native training handlers pass one family-neutral observer beside the independent component and
profiler services. Their optimizer safe points emit cadence-limited heartbeats and offer common
loss, throughput, and GPU-memory measurements; undeclared measurements remain local rather than
silently changing run telemetry. The fixed entry point emits an immediate `initializing` heartbeat
before importing trainer work. No trainer reconstructs metric units or aggregation policy.

`EvalGalleryPublisher` is the shared qualitative-evaluation publication path. It requires an exact
`rwkv-lab.eval-gallery.v2` output declaration from the immutable invocation, confines all input
images to declared read/write roots, fully decodes supported raster formats with Pillow, and copies
them into a SHA-256 object store beneath the run directory. It then atomically seals a canonical
manifest, publishes that manifest through `WorkerSession.artifact()`, and returns the durable worker
sequence. Generated and target/source images, held-out membership, condition digest, evaluator,
checkpoint, policy, seed, sampling attributes, producer attempt, and optimizer step are all bound.
Family handlers do not receive the controller session: they return typed
`EvalGalleryPublicationRequest` values after evaluation has completed, and the fixed worker runner
publishes those requests before its terminal event. This keeps transport and artifact authority out
of evaluator code while allowing a handler to return multiple append-only gallery revisions. A
request may refer to a checkpoint publication from the same handler by its request index; after the
checkpoint is sealed, the runner substitutes its actual artifact ID and manifest SHA-256 before the
gallery can be published.

`CheckpointPublisher` is the independent state-artifact path. A family handler returns only a
typed publication request after its trainer has atomically completed a checkpoint; it never gets a
controller session or implements transport. The fixed runner then requires a declared immutable
checkpoint output, confines the source beneath the exact run directory, rejects symlinks and
nonregular entries, and reflinks every file out of rolling trainer retention when the filesystem
supports copy-on-write cloning, with descriptor-copy fallback. It detects identity or
tree changes during the copy, hashes each object, records the closed resume grade, optimizer step,
state-component inventory, producer attempt, and parent artifact lineage, then atomically promotes
a content-addressed `trainvm.checkpoint-snapshot.v1` revision. Replays verify both the canonical
manifest and every frozen payload object before repeating the artifact announcement. Only after
that durable publication succeeds does the runner emit its terminal worker event, which carries the
published checkpoint artifact ID rather than a mutable trainer path. MageFlow appearance,
MageFlow terminal/TREAD, and Qwen AO3 handlers all use this one publication service when their
invocation declares a checkpoint output; their current honest resume grade remains `compatible`.
The native controller independently matches every worker artifact's logical name, kind, schema,
and fingerprint algorithm against that sealed output declaration before it journals the event;
undeclared, ambiguous, or shape-changing publications fail without consuming a worker sequence.

GPU profiling is likewise a worker service rather than trainer-local CLI state. A native training
adapter receives one `WorkerStepProfiler` and calls `step()` exactly once after each optimizer
update. The service owns the bounded skip/warmup/capture state machine, Torch profiler lifetime,
immutable trace publication, invocation binding, and restricted-artifact metadata. Trainers do not
construct profiler schedules or publish raw trace paths themselves. A null implementation keeps
non-profiled invocations free of profiler setup overhead. Nsight backends require dedicated sealed
host launch profiles and fail closed in the in-process Torch adapter path. Torch summaries include
the captured optimizer-step wall interval, unioned GPU-active time and ratio, accelerator launch
count, and allocator baseline/peak allocated and reserved bytes; the dashboard exposes these
without reading the restricted raw trace. The same protocol supplies `input_wait()` for direct
batch acquisition and `track_input()` for prefetch iterators. A trainer marks its real blocking
boundary; complete captured windows publish input-stall time/ratio, while uninstrumented adapters
omit those fields instead of estimating them from accelerator inactivity.

The helper is deliberately trainer-family-neutral; MageFlow, RWKV vision, and transformer vision
adapters must not implement their own gallery-directory conventions.

The receive thread never performs tensor work and never opens dashboard or
journal storage. `WorkerControlRuntime` drains that transport only at a trainer-owned
microbatch, optimizer-step, eval, or checkpoint boundary. It preserves revision order across
different safe points, applies each patch atomically through a family callback, acknowledges the
exact requested values and effective step, rejects stale/duplicate/adapter-invalid patches without
mutating effective state, and reports pause/restart controls as replacement-worker work. The fixed
runner constructs this service from the sealed invocation rather than giving a trainer the session.
Scratch RWKV consumes all four safe-point hooks, persists the effective control snapshot in its
terminal checkpoint, verifies it on resume, and honestly rejects live mutation because its v1
catalog is restart-only. All MageFlow training adapters additionally lower their authority-selected
initial controls before construction and atomically apply learning rate, evaluation cadence, and
caption dropout at their declared boundaries. Learning-rate rebasing is owned by the schedule
component and preserve schedule phase plus full-backbone or appearance/terminal expert, backbone, and REPA group
ratios; checkpoints bind and verify the
effective revision and values, while the static training-contract identity excludes only the named
mutable fields. Cached conditioning rejects a zero-to-positive caption-dropout transition when no
null embedding was built. Qwen retains its frozen configuration object and uses a separate mutable
overlay for learning rate and evaluation cadence; startup LR overrides proportionally preserve the
PowerCool minimum ratio, and live rebasing preserves the current schedule phase. Its checkpoints
carry the same verified effective-control state. The lifecycle service remains separate from
mutable controls and only exposes the checkpoint/pause/cancel surface declared by each exact
adapter profile.

## Worker-originated runtime evidence

A cache namespace is only reusable if the runtime, device and ABI that produced
the bytes are the runtime, device and ABI about to consume them. Those facts
exist inside the worker process — it holds the interpreter, the loaded ELF
graph and the compute context — and nowhere else. The worker therefore measures
them, and `rwkv_lab.trainvm_worker.runtime_evidence` is where it does so, with
the standard library only, because it sits on the same side of the trust
boundary as the pre-import guard.

Measuring is not deciding. The report the worker emits, a canonical
`trainvm.worker-runtime-evidence/v1` object, is split so that admission cannot
be confused with transport:

- **measurements** — compute vendor and architecture, device UUID and PCI
  address where the launch is placement specific, the driver identity taken
  from the guard's own reading rather than a second parse of the same file, the
  installed versions of the distributions a compiled artifact binds to, a host
  ABI digest, and a compute compatibility digest carrying what the architecture
  string cannot (the host's instruction set, or the exact attributes behind an
  architecture name several parts share);
- **identity claims** — run, node, attempt, launch nonce, concurrency key,
  lease and fencing token, echoed from the sealed bootstrap. The authority
  already knows all of them, and compares rather than believes: a report from a
  superseded attempt, a released lease, or a second launch of the same attempt
  is refused;
- **nothing else.** There is no host id, no boot id, no launch spec digest, no
  inventory receipt digest, no resource binding digest and no receipt name. The
  runtime closure fingerprint is not measured here either — it is whatever the
  pre-import guard verified, passed in by the caller, because re-deriving it
  would be a second answer to a question the guard already answered under
  stricter conditions.

That last bullet is the design, not a simplification of it. Those absent fields
are exactly the ones that name the immutable receipt a cache lookup reads, so a
worker cannot author a document that authorizes reuse — not because it would be
rejected, but because it cannot express one. `admit_worker_runtime_evidence`
derives the probe context from host identity, the sealed launch and the
inventory receipt, copies those fields into the snapshot itself, and only then
compares the worker's measurements against the devices the launch was actually
fenced to. `publish_worker_runtime_evidence` is the only path to the
authority-owned immutable publisher, so no caller reaches it with an unadmitted
snapshot.

Two tests carry this rather than prose. `worker_runtime_evidence_tests` feeds a
complete sealed runtime receipt back in as a report and requires it to be
refused, grafts an authority-derived digest onto a valid report and requires
that to be refused too, and then walks each forged binding — superseded fence,
foreign lease, other attempt, other launch, other run, unsealed closure, a
device the launch holds no fence for — asserting after each that the receipt
directory is still empty and that derivation still has nothing to read.
`worker_runtime_evidence_python_parity` then runs that same suite over a report
the real probe measured on the host, through the real bootstrap decoder, so the
two sides are not each testing their own idea of the other. It uses the
portable CPU path deliberately: no accelerator, no driver and no CUDA stack, so
it asserts the same thing on a hosted runner as on the training host.

### Sending it

`WorkerSession.publish_runtime_evidence(report)` puts the measured report on
the `WorkerToController` stream the worker already holds, as the
`runtime_evidence` arm. The authority decodes it, resolves the active attempt's
`ResolvedLaunchSpec` and `HostInventoryReceipt`, and publishes one immutable
receipt through `publish_worker_runtime_evidence`.

Two things about this arm are unlike every other worker publication, and both
follow from the message being exactly the report and nothing else:

- **It carries no `worker_sequence` and is never acknowledged.** A sequence
  would be the first field on a message whose whole property is that it carries
  nothing the authority derives for itself, and once one bookkeeping field is
  admissible the argument for the next one is already made. Acceptance is the
  receipt the authority published; refusal is the stream's terminal status. The
  call therefore returns nothing and does not wait.
- **A field the report does not have is refused, not dropped.** Passing an
  authority-derived key raises rather than silently sending a document without
  it, so a caller cannot believe it sent something the authority never saw.

Three literals now describe this one document — the proto arm, the C++ struct,
and the session's field lists — and nothing keeps them in step on its own. Two
tests do: `worker_runtime_evidence_wire_hop` in the native suite compares the
proto descriptor against the reflected struct, and
`test_runtime_evidence_transport_matches_the_protocol_message` compares it
against the session's. Both also assert, by name, that the arm has grown none
of the authority-derived fields.

The authority refuses a report whose lease has already moved **on the wire**,
before the publisher is asked anything: it re-reads the durable launch binding
rather than trusting the identity its connection was opened with, because a
connection's identity was true when the stream opened and says nothing about
now. Refusing at admission instead would reach the same verdict and is not the
same property — the native test proves the distinction by counting publisher
invocations, not by reading the status.

A deployment that configures no immutable receipt root holds no evidence
authority at all, and the message is then refused with `FAILED_PRECONDITION`.
Silently accepting a report that cannot be published would be indistinguishable
to a worker from a published one.

## Fixed adapter runner

`scripts/build_trainvm_worker_artifact.py` deterministically builds the sole sealed project-code
artifact for the currently migrated training adapters. The stored zipapp contains the complete
`rwkv_lab` Python source surface, the checked-in TrainVM protobuf package, one fixed `__main__`, and
an embedded canonical per-member SHA-256 manifest. It also embeds a
`trainvm.python-bootstrap-runtime-closure/v4` manifest produced by the exact copied interpreter.
That manifest binds the interpreter identity, Python standard-library tree, declared root
distributions, and their recursively complete installed-file closure. Since v3 it also closes over
what the dynamic loader would then load underneath that file closure, which the file closure by
itself cannot express: the ELF graph reachable from the closure and the import path, each
`DT_NEEDED` name resolved through the loader's own search order and pinned as a manifest entry, the
NVIDIA driver identity — which lives in the kernel and so is carried by no file digest — and a
kernel registry inventorying every loadable object on the import path, scoped to the directories
rather than to the closure's distributions so an extension no distribution claims is still seen.

v4 splits that driver identity in two. `driver_identity` is what is compared: the version token from
`/proc/driver/nvidia/version` plus the GNU build ID of the loaded `nvidia` module, read from
`/sys/module/nvidia/notes/.note.gnu.build-id`. `driver_report` is the whole first line of
`/proc/driver/nvidia/version`, recorded and quoted in a rejection but never compared. v3 compared
that whole line, which ends in the user and host that compiled the module, so a DKMS rebuild — or
the same driver version built on another machine — read as a driver change: the closure refused the
host and every cache namespace went cold for a bit-for-bit compatible driver. The whole line was
never a legal identity downstream either, since it contains spaces, parentheses and `@` and
`fixed_public_identity` in `trainvm/src/cache_namespace.cpp` accepts none of them. The build ID is
kept because the version token alone is too weak on its own: it is the linker's hash of the module's
own contents, so a rebuild producing identical bytes reads as identical and one producing different
bytes reads as different. A v3 manifest is refused by its `api_version`, so closures sealed against
the old whole-line form must be resealed — a one-time cold cache, taken deliberately in preference
to accepting an old-format identity as equal to a new-format one.

The system search path is not re-derived at verification time; the `ld.so.conf` files that produce
it are pinned instead. The reflected native
`trainvm.rwkv-lab-worker-runtime-requirements/v1` contract is the sole adapter-to-root-distribution
authority: the materializer queries it with `inspect-rwkv-lab-runtime-requirements` instead of
maintaining a second Python dependency map. Shared dispatch and evaluation roots include `grpcio`,
`pillow`, `protobuf`, and `torch`; adapter profiles add their exact MageFlow, Qwen, or RWKV trainer
roots.
The stdlib-only guard verifies its canonical digest, permissions, non-worker-writable ancestors,
symlinks, sizes, and every regular-file SHA-256 before importing the worker entrypoint or any
third-party module. It then re-runs two decisions rather than re-reading their recorded answers,
because a digest cannot carry either: it resolves each object's `DT_NEEDED` names again over that
object's recorded search order, so a library planted in an earlier directory is rejected while every
pinned byte still matches, and it re-walks the import path and requires the discovered object set to
equal the recorded one, so an extension that has merely appeared is rejected too. Member order, timestamps, modes, and bytes are
fixed; the normal native test target builds it twice and requires byte equality. Hostd executes the
sealed interpreter with `-I`, installs the zipapp at fd 3 through its separately bound code-argument
slot, clears the environment, and appends exactly `--trainvm-bootstrap-fd=4`; the runner rejects
every other argument. Its dispatch table contains exact `(adapter, version, operation, contract)`
tuples for MageFlow full-backbone plus appearance and terminal experts, Qwen AO3 continuation, eight canonical
MLA-family variants, scratch RWKV pretraining, and restart-only RWKV post-training. The latter
publishes adapter, reward-head, terminal result, and metric files through the generic immutable-tree
artifact boundary rather than presenting them as a resumable checkpoint. An experiment cannot
select an import string, script, argv, environment
override, or entry-point path.

`trainvm inspect-rwkv-lab-deployment` lowers each adapter's exact zipapp,
bootstrap-runtime-closure, and interpreter fingerprints, paths, working directory, and trusted
roots into matching reflected adapter and `trainvm.host-launches/v4` documents. Every one of the
four profiles advertises the
capability set implemented by those exact
worker bytes; requested capabilities may only narrow it. The artifact execution gate loads the
zipapp under `python -I` with an empty environment, verifies that the generated host registry binds
the archive and closure digests and isolation-before-code argv ABI, rejects a self-consistent
manifest whose claimed file digest differs from the host, passes a real sealed bootstrap at fd 4,
and completes a gRPC Hello/Welcome already-completed replay without importing trainer tensor work.

Trainer configuration is an inline object in invocation `inputs.config`. Because the invocation is
canonical and content-addressed, these values cannot change between submission and execution. A
pathname to a mutable JSON configuration is rejected. The adapter also requires the trainer's run
directory to equal `workspace.run_directory`. `WorkspacePathAuthority` canonicalizes every
top-level model, dataset/pack, manifest, checkpoint/resume, and encoder-cache pathname used by the
four fixed adapters. Existing reads must resolve beneath a declared read or write root; prospective
writes must have a resolved directory ancestor beneath a declared write root; symlink escapes and
relative or non-normalized paths fail before trainer code runs. Larger configurations must become
immutable artifact manifests with authority-verified fingerprints rather than path references.

The same workspace can declare `input_content_roots`, a strictly sorted, non-overlapping list of
reflected `trainvm.input-content-root/v1` identities. Generate each identity with:

```sh
trainvm inspect-input-content-root /absolute/dataset-or-model-root
```

Each compact identity binds the selected absolute path, file/directory kind, bounded file count,
total bytes, and a deterministic SHA-256 Merkle root. Directory nodes commit sorted UTF-8 child
names, node kinds, and child digests; file nodes commit size and the full file hash. Static input
roots reject symlinks, special files, mutation during measurement, noncanonical paths, excessive
depth, and count/byte limit violations. The native declaration is frozen into the plan and worker
invocation. Before importing MageFlow, Qwen, or RWKV trainer modules, the stdlib worker verifier
walks the same descriptor-relative tree and requires an exact identity match. Every ordinary path
read must fall within one verified content root, so pre-launch drift in a JSONL, packed corpus,
model/tokenizer tree, expert bank, encoder cache, or schedule is rejected even when its path is
unchanged. Production content roots must remain authority-owned and non-worker-writable for the
duration of the invocation: measurement detects drift before dispatch and races during its walk,
but does not turn a mutable filesystem path into immutable storage. Immutable
controller-published resume checkpoints use their existing per-object manifest verifier and are not
double-classified as static workspace inputs.

For experiment templates with several roots, keep the unhashed path list in a closed
`trainvm.input-content-root-set/v1` document and produce the runnable snapshot without hand-editing:

```sh
trainvm lock-input-content experiment.json input-roots.json > experiment.locked.json
trainvm validate experiment.locked.json
```

The native command decodes the root-set through the reflected schema, rejects unknown fields,
duplicates and overlaps, measures every root with the same Merkle implementation, sorts the
identities, replaces `workspace.input_content_roots`, and recompiles the complete experiment before
emitting JSON. It never launches a worker or writes either source document. A later worker still
remeasures every identity, so drift between authoring and dispatch fails closed.

The long-lived controller may reuse file digests between a dry preview and its fenced launch. That
cache is controller-owned, bounded LRU state: entries are keyed by a non-reusable mount identity
plus exact inode metadata, directory membership is always re-enumerated, unknown filesystem
semantics bypass reuse, and newly measured records publish only after the locked plan recompiles.
Each transaction is capped at the cache capacity as well; excess measurements are rehashed safely
but not staged, and both root telemetry and the receipt report those staging saturations.
The resulting typed measurement receipt is excluded from the plan hash but a successful launch
persists it in `run.created`. This optimization deliberately does **not** weaken the worker boundary:
the Python worker still rereads and hashes every nested byte before trainer import. For very large
models that remaining verification is substantial startup work and currently happens after resource
acquisition. A future sealed-manifest/load-stream design may fuse verification with model loading or
move it before accelerator acquisition, but it must preserve exact nested-object byte verification.

The controller warm-commit microbenchmark is reproducible without adding it to the normal build or
test workload:

```bash
cmake --build trainvm/build --target input_content_cache_benchmark
trainvm/build/input_content_cache_benchmark 20000
```

It creates the requested number of temporary zero-byte files, cold-populates the authority cache,
then reports warm measurement and transaction-commit microseconds separately.

The fixed runner returns an already-completed replay without executing tensor work, publishes a
durably receipted terminal result on success, freezes any declared checkpoint before that terminal
result, and converts trainer exceptions to a bounded
`operation.failed` event containing only an error class. MageFlow and Qwen profiles expose their
implemented compatible-resume checkpoint and safe-point lifecycle controls; scratch RWKV exposes
only terminal-checkpoint semantics. Mutable-control and lifecycle capability remain exact
per-adapter claims rather than inferences from the shared runtime. Top-level paths are confined and
declared content-root trees are recursively bound. A manifest that references payloads outside its
declared root set remains invalid; remote/object-store references still require a typed immutable
artifact provider rather than pathname authority.

Downstream Python operations consume controller-selected non-checkpoint artifacts through
`resolve_input_artifact()`, never by opening the descriptor URI directly. The resolver requires the
exact artifact descriptor shape, local canonical manifest URI, declared kind and schema, workspace
confinement, manifest fingerprint, producer and parent lineage, canonical tree digest, and every
payload object hash. `read_input_artifact_file()` rechecks a selected object's stable file identity
and digest at the point of use, while `load_input_artifact_json()` additionally requires canonical,
finite JSON. The scalar-metric decision adapter is the first production consumer of this boundary:
it compares two independently verified result artifacts and publishes a new immutable receipt whose
parents are exactly those candidate artifact IDs.

Install the optional runtime dependencies with:

```sh
uv sync --extra trainvm-worker
```

Materialize an immutable deployment directory without hand-editing either registry:

```sh
scripts/materialize_trainvm_worker_deployment.py \
  --trainvm /opt/trainvm/bin/trainvm \
  --adapter-python rwkv-lab.mageflow-appearance-expert=/opt/trainvm/venvs/mageflow/bin/python3 \
  --adapter-python rwkv-lab.mageflow-terminal-expert=/opt/trainvm/venvs/mageflow/bin/python3 \
  --adapter-python rwkv-lab.qwen-ao3=/opt/trainvm/venvs/qwen/bin/python3 \
  --adapter-python rwkv-lab.rwkv-scratch=/opt/trainvm/venvs/rwkv/bin/python3 \
  --source-root /src/rwkv-lab/src \
  --output-directory /opt/trainvm/workers/rwkv-lab-v1 \
  --working-directory /srv/trainvm/work \
  --runtime-digest-cache /opt/trainvm/cache/runtime-digests.json \
  --trusted-root /opt/trainvm \
  --trusted-root /srv/trainvm
```

The command groups adapters only when both interpreter and native-declared root set match, then
publishes contract-grouped worker zipapps and closure manifests plus `adapters.json`,
`host-launches.json`, and a digest-bound `deployment-receipt.json`. The two MageFlow routes can
share one runtime group; Qwen and RWKV remain separate. `--python` remains a shared-root fixture and
compatibility mode. Materialization is byte-idempotent and refuses to replace changed outputs
unless the operator explicitly supplies `--replace`.

`--runtime-digest-cache` stops the builder re-reading an unchanged multi-gigabyte package closure
on every deployment; it is passed straight through to the closure builder's own `--digest-cache`.
A cached digest is bound to the open descriptor's device, inode, mode, ownership, size, mtime **and
ctime**, and the descriptor is reattested after the digest is settled whether it came from the
cache or from a read. `st_ctime_ns` is the field that makes this safe rather than merely fast:
`st_mtime_ns` is forgeable with no privilege by the file's own owner via `utimensat`, so an
mtime-keyed cache can be made to serve a stale digest for changed bytes, while `st_ctime_ns` is
kernel-maintained and moves on every write regardless. It is also why no boot fence is needed
against `st_ino` reuse — a recycled inode carries its own creation ctime, which would have to
collide to the nanosecond.

The cache file itself must be a bounded, owner-only, non-group-or-world-writable regular file
carrying the `trainvm.runtime-closure-digest-cache/v1` schema; anything else is rejected rather
than read leniently. Without the flag the cache still exists for the duration of one
materialization, so runtime groups sharing a stdlib hit it. Measured on this host's live
torch/grpcio/pillow/protobuf closure — 19,969 files, 8.14 GB pinned — a cold build hashes 10.18 GB
across 20,129 opened descriptors and a warm one hashes 4.05 MB across 234, for a byte-identical
closure manifest. The cache affects build latency only and is never part of worker launch
authority: `rwkv_lab.trainvm_runtime_guard` re-reads every pinned byte at worker start and has no
cache, deliberately.

The interpreter path must be a regular executable, not the usual symlink created by many virtual
environment tools. Hostd refuses symlink traversal for launch artifacts, and silently resolving a
venv symlink to the base interpreter would select the wrong package closure. A deployment can use a
venv created with a copied interpreter. The deployment directory and every closure ancestor must
be non-writable by the configured worker UID; materialization by that same UID is therefore not a
production deployment boundary.

Regenerate the checked-in protobuf bindings after changing the protocol. One script
covers all four generated files — the Python pair under `src/trainvm/v1/` and the Go
pair under `dashboard/gen/trainvm/v1/` — because regenerating only one language is how
the Go client ended up without a `GetReconciliationStatus` method while CI stayed green:

```sh
scripts/generate_trainvm_proto.sh          # rewrite the bindings
scripts/generate_trainvm_proto.sh --check  # fail if they no longer match the proto
```

The `--check` form is what the `proto-bindings` CI job runs, so forgetting to
regenerate is a red pull request rather than a missing method discovered months later.

Hostd descriptor delegation and the stopped launcher now attest the bootstrap, install sealed
Python code at fd 3 and the bootstrap at fd 4, and bind their combined identity into durable launch
evidence. Host-launch v4 additionally binds the bootstrap closure independently of the zipapp and
requires an authority-owned cache probe to report that exact digest. The fixed runner closes the
project-code entry-point/argv boundary and verifies the shared pre-dispatch Python closure.
Production process launch remains deployment-gated: TrainVM drives the guarded hostd transaction,
while per-adapter lazy trainer dependencies, system ELF/native-library and CUDA/driver closure,
nested path authority, privileged crash-window evidence, and per-adapter end-to-end qualification
remain outstanding. `-I` removes user-site and environment injection; those remaining layers are
represented separately rather than being falsely treated as zipapp contents.
