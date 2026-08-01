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

`EvalGalleryPublisher` is the shared qualitative-evaluation publication path. It requires an exact
`rwkv-lab.eval-gallery.v2` output declaration from the immutable invocation, confines all input
images to declared read/write roots, fully decodes supported raster formats with Pillow, and copies
them into a SHA-256 object store beneath the run directory. It then atomically seals a canonical
manifest, publishes that manifest through `WorkerSession.artifact()`, and returns the durable worker
sequence. Generated and target/source images, held-out membership, condition digest, evaluator,
checkpoint, policy, seed, sampling attributes, producer attempt, and optimizer step are all bound.
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
durably receipted terminal result on success, and converts trainer exceptions to a bounded
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
disabled until TrainVM drives that guarded hostd transaction, runtime closure and nested path
authority are enforced, and each Python adapter is qualified end to end.
