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

Install the optional runtime dependencies with:

```sh
uv sync --extra trainvm-worker
```

Regenerate checked-in Python protobuf bindings after changing the protocol:

```sh
scripts/generate_trainvm_python_proto.sh
```

Hostd descriptor delegation and the stopped launcher now attest the bootstrap,
install sealed Python code at fd 3 and the bootstrap at fd 4, and bind their
combined identity into durable launch evidence. Production process launch
remains disabled until TrainVM drives that guarded hostd transaction, runtime
closure is enforced, and the Python adapter entry points are qualified end to
end.
