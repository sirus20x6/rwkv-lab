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
- a sorted capability set and replay cursor.

It contains no model path, dataset path, optimizer setting, learning-rate
setting, shell fragment, environment override, or secret. After the worker
opens `WorkerControl`, TrainVM returns a content-addressed
`trainvm.worker-invocation/v1` document in `WorkerWelcome`. That immutable
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
- typed pause, resume, checkpoint, cancel, and control commands;
- explicit control acknowledgement at adapter-selected safe points;
- one canonical terminal result and durable receipt.

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
journal storage. A trainer polls `poll_commands()` at its own microbatch,
optimizer-step, eval, or checkpoint boundary, applies an eligible control, and
then calls `acknowledge_controls()` with the effective values and step.

## Fixed adapter runner

`scripts/trainvm_worker_entrypoint.py` is the sole sealed Python code artifact for the currently
migrated training adapters. Hostd executes it from fd 3 and appends exactly
`--trainvm-bootstrap-fd=4`; the runner rejects every other argument. Its dispatch table contains
exact `(adapter, version, operation, contract)` tuples and fixed imports for the MageFlow appearance
expert, MageFlow terminal expert, and Qwen AO3 continuation. An experiment cannot select an import
string, script, argv, environment override, or entry-point path.

Trainer configuration is an inline object in invocation `inputs.config`. Because the invocation is
canonical and content-addressed, these values cannot change between submission and execution. A
pathname to a mutable JSON configuration is rejected. The adapter also requires the trainer's run
directory to equal `workspace.run_directory`. `WorkspacePathAuthority` canonicalizes every
top-level model, dataset/pack, manifest, checkpoint/resume, and encoder-cache pathname used by the
three fixed adapters. Existing reads must resolve beneath a declared read or write root; prospective
writes must have a resolved directory ancestor beneath a declared write root; symlink escapes and
relative or non-normalized paths fail before trainer code runs. Larger configurations must become
immutable artifact manifests with authority-verified fingerprints rather than path references.

The fixed runner returns an already-completed replay without executing tensor work, publishes a
durably receipted terminal result on success, freezes any declared checkpoint before that terminal
result, and converts trainer exceptions to a bounded
`operation.failed` event containing only an error class. It deliberately does not claim live
pause/checkpoint/control support yet: trainers must first expose safe-point hooks through this
session before their registry capability can advertise those controls. Top-level paths are now
confined, but nested references inside image JSONL and packed-corpus/model manifests remain
non-production-qualified until the authority binds their immutable artifact identities and the
adapter recursively validates every referenced object.

Install the optional runtime dependencies with:

```sh
uv sync --extra trainvm-worker
```

Regenerate checked-in Python protobuf bindings after changing the protocol:

```sh
scripts/generate_trainvm_python_proto.sh
```

Hostd descriptor delegation and the stopped launcher now attest the bootstrap, install sealed
Python code at fd 3 and the bootstrap at fd 4, and bind their combined identity into durable launch
evidence. The fixed runner closes the entry-point/argv boundary. Production process launch remains
deployment-gated: TrainVM now drives the guarded hostd transaction, while complete runtime closure,
nested path authority, privileged crash-window evidence, and per-adapter end-to-end qualification
remain outstanding.
