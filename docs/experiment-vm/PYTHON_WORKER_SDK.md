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
Family handlers do not receive the controller session. Terminal evaluation returns typed
`EvalGalleryPublicationRequest` values to the fixed runner. Long MageFlow operations additionally
receive only a same-step callback backed by `WorkerPublicationRuntime`: the trainer supplies one
completed checkpoint directory and completed gallery result, then the runtime freezes and announces
the checkpoint before substituting its actual artifact ID and manifest SHA-256 into the gallery.
The runtime records each durable publication for terminal result assembly and requires strictly
increasing optimizer steps. This keeps transport and artifact authority out of evaluator code while
making intermediate append-only revisions visible during training rather than only after exit.

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
full-backbone, and terminal/TREAD handlers use it for periodic checkpoint-bound galleries as well as
terminal publication; Qwen AO3 uses the terminal path. Their current honest resume grade remains
`compatible`.
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

## Fixed adapter runner

`scripts/build_trainvm_worker_artifact.py` deterministically builds the sole sealed project-code
artifact for the currently migrated training adapters. The stored zipapp contains the complete
`rwkv_lab` Python source surface, the checked-in TrainVM protobuf package, one fixed `__main__`, and
an embedded canonical per-member SHA-256 manifest. It also embeds a
`trainvm.python-bootstrap-runtime-closure/v2` manifest produced by the exact copied interpreter.
That manifest binds the interpreter identity, Python standard-library tree, declared root
distributions, and their recursively complete installed-file closure. The reflected native
`trainvm.rwkv-lab-worker-runtime-requirements/v1` contract is the sole adapter-to-root-distribution
authority: the materializer queries it with `inspect-rwkv-lab-runtime-requirements` instead of
maintaining a second Python dependency map. Shared dispatch and evaluation roots include `grpcio`,
`pillow`, `protobuf`, and `torch`; adapter profiles add their exact MageFlow, Qwen, or RWKV trainer
roots.
The stdlib-only guard verifies its canonical digest, permissions, non-worker-writable ancestors,
symlinks, sizes, and every regular-file SHA-256 before importing the worker entrypoint or any
third-party module. Member order, timestamps, modes, and bytes are
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
  --trusted-root /opt/trainvm \
  --trusted-root /srv/trainvm
```

The command groups adapters only when both interpreter and native-declared root set match, then
publishes contract-grouped worker zipapps and closure manifests plus `adapters.json`,
`host-launches.json`, and a digest-bound `deployment-receipt.json`. The two MageFlow routes can
share one runtime group; Qwen and RWKV remain separate. `--python` remains a shared-root fixture and
compatibility mode. Materialization is byte-idempotent and refuses to replace changed outputs
unless the operator explicitly supplies `--replace`.

The interpreter path must be a regular executable, not the usual symlink created by many virtual
environment tools. Hostd refuses symlink traversal for launch artifacts, and silently resolving a
venv symlink to the base interpreter would select the wrong package closure. A deployment can use a
venv created with a copied interpreter. The deployment directory and every closure ancestor must
be non-writable by the configured worker UID; materialization by that same UID is therefore not a
production deployment boundary.

Regenerate checked-in Python protobuf bindings after changing the protocol:

```sh
scripts/generate_trainvm_python_proto.sh
```

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
