# TrainVM cross-family compiler coverage fixtures

These documents exercise the v1 experiment compiler across representative non-MageFlow training
families. They are declarative shape and semantic-validation fixtures only. Validation and planning
must remain GPU-free and process-free.

The family-worker selectors are intentionally unique, exact
adapter/version/operation/contract tuples. They do not correspond to entries in an authority-owned
adapter registry, contain no executable or command paths, and therefore cannot authorize a worker or
host launch. Every graph does use the compiled `trainvm.core` resource-admission and release
operations so it exercises the common lease topology. Registering and qualifying each real family
adapter remains a separate prerequisite.

Every fixture declares bounded resources and `exact_resume: false`. This is deliberate: the current
legacy implementations range from restart-only or terminal-checkpoint behavior to compatible resume,
but none of these coverage documents upgrades that evidence into an exact-recovery claim.
They also omit a pause resource policy: retaining or releasing accelerators while paused is an
operation capability that only an authority-owned adapter profile may qualify.

The covered families are:

- `scratch-rwkv-pretrain.json`
- `transformer-mla-continuation.json`
- `vision-distillation-training.json`
- `rwkv-posttraining.json`
- `rwkv-rlvr.json`
- `external-revision-pinned-trainer.json`

Validate them with:

```sh
for fixture in docs/experiment-vm/examples/coverage/*.json; do
  trainvm/build/trainvm validate "$fixture"
done
```
