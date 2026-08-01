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
it.

## Session API

`rwkv_lab.trainvm_worker.WorkerSession` provides:

- one Hello-first bidirectional gRPC stream;
- replay-aware worker/controller sequence checks;
- typed heartbeats and scalar metrics;
- complete artifact-manifest publication;
- typed pause, resume, checkpoint, cancel, and control commands;
- explicit control acknowledgement at adapter-selected safe points;
- one canonical terminal result and durable receipt.

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

Production process launch remains disabled until the bootstrap memfd is wired
through hostd's descriptor delegation and the Python adapter entry points are
qualified end to end.
