# Workflow coverage and migration checks

This matrix keeps TrainVM from being designed only around one trainer. Each existing workflow was
reduced to the proposed node/event/artifact model before implementation begins.

| Existing workflow | Current mechanism | Declarative representation | Required adapter capability |
|---|---|---|---|
| MageFlow cache boundary and resume | shell polls `status.json`, `pgrep`, `jq`; rewrites configs; launches cache and trainer | train node transitions on typed boundary event; plan/build/validate cache nodes; exact resume node | checkpoint manifest, cache-plan and cache-build operations, structured terminal reason |
| RADIO/V4H eval restarts | shell loop treats exit 42 as planned and guards rapid failures | cyclic train/eval node with `worker.restart_requested`, visit bound, monotonic step guard | clean-process eval safe point and exact checkpoint ack |
| vision representation A/B | Python supervisor runs two arms sequentially and scans JSONL for completion | two arm nodes or a parameterized map template; result artifacts feed a compare node | run-arm operation and metric/result contract |
| GPU launch queue | Go table plus PID liveness and optional start-next | queued desired state plus exclusive resource lease | no trainer-specific capability |
| dashboard live tuning | Go and Python share SQLite numeric rows | versioned typed control patch, safe-point application, effective-value ack | operation control descriptor and worker safe-point SDK |
| checkpoints and pause | Unix signals and inferred files | typed command and `CheckpointCommitted` artifact event | atomic checkpoint publication and graceful stop |
| RLVR/post-training campaigns | family-specific Go form validation and Python argv construction | registered campaign operation plus declarative arm graph and promotion policy | campaign input/result contracts |
| qualification runs | custom Go handler and receipt discovery | qualification node publishes versioned report artifact | qualification operation and report schema |
| eval galleries | trainer filesystem conventions and dashboard path discovery | append-only image-gallery artifact with generated/target pairs and step index | gallery manifest publisher |
| legacy runs | JSONL ingestion and PID discovery | compatibility observer emits canonical events in shadow mode | legacy translator only; cannot claim exact ownership |

## Acceptance scenarios

The first runtime is not ready to own training until these scenarios pass without manual file edits:

1. Validate and plan the example MageFlow document without starting a worker.
2. Kill TrainVM before dispatch, after dispatch, before receipt, and after receipt at every node; replay
   must choose the same next action without duplicating a published artifact.
3. Kill a MageFlow worker during training and recover only from a complete fingerprinted checkpoint.
4. Pause at an optimizer safe point, release the GPU, restart the daemon, and resume with identical
   data cursor, RNG, optimizer, and control revision.
5. Apply a three-value control patch concurrently with an eval boundary; all values become effective
   at one declared safe point or none do.
6. Request a stale control revision and confirm it is rejected without changing effective values.
7. Run a clean-process eval loop past three iterations, proving the visit guard observes increasing
   optimizer steps; reject a synthetic restart storm with no progress.
8. Execute sequential A/B arms from one template and compare their versioned result artifacts.
9. Rebuild every dashboard projection from the immutable journal and obtain the same visible state.
10. Render the experiment editor and control panel from schemas without checking the experiment name.

## Decisions that require measured prototypes

- SQLite event throughput and metric compaction thresholds.
- gRPC Python safe-point polling overhead at the current images/second rate.
- process adoption reliability using pidfd plus worker launch identity on Linux.
- cost and diagnostic quality of GCC reflection compared with generated registration code.
- whether artifact payloads need a local content-addressed store or only content-addressed manifests
  referencing existing run directories.

None of these measurements changes the persisted schema-first boundary.
