# Native algorithm migration

Status: CPU-only analysis and learning-rate schedule tranches implemented.

The goal is not to rewrite PyTorch in C++. It is to move deterministic control,
analysis, validation, and measured hot paths into strongly typed native code while
keeping model construction, autograd, tensor-rich dataset work, and checkpoint
serialization in the Python worker.

## Implemented tranche

`trainvm_experiment_analysis` now owns:

- paired deltas, sample effect size, bootstrap confidence intervals, exact or
  sampled sign-flip permutation tests, and sample-count guidance;
- Holm family-wise correction and O'Brien-Fleming, Pocock, or linear sequential
  alpha spending;
- mixed maximize/minimize Pareto-front decisions with explicit missing-data
  behavior;
- a bounded, typed, strictly read-only SQLite snapshot of campaign summaries and
  latest paired comparisons;
- the legacy `results` fallback: deterministic latest-row-per-config collapse,
  metric extraction, ranking, and baseline deltas;
- `trainvm inspect-registry`, a JSON diagnostic suitable for the dashboard or
  operators without importing Python.

The reader uses SQLite's read-only open mode, `query_only`, a deny-by-default SQL
authorizer, one snapshot transaction, bounded text/row/query work, strict dynamic
type checks, and no schema installation. Tests make the fixture filesystem
read-only and verify unchanged schema, size, modification time, and absence of
SQLite side files.

The v1 native bootstrap and Monte Carlo permutation path uses a specified stable
SplitMix64 stream. Exact-enumeration p-values and deterministic corrections match
the Python helper. Seeded Monte Carlo results are reproducible within the native
contract but intentionally do not claim byte-for-byte equality with NumPy's PCG64
stream; migration receipts must record the analysis implementation/version.

`trainvm_training_schedules` now owns the reflected configuration validation and
pure multiplier math for `rwkv_lab.schedule.linear_warmup_cosine.v1` and
`rwkv_lab.schedule.powercool.v1`. The bounded `trainvm inspect-training-schedule`
diagnostic exposes those native values over an optimizer-step range. The binding
evidence is `training_schedule_python_parity`, which evaluates a configuration and
step grid against `rwkv_lab.training_runtime.schedules` and requires exact results
for exactly representable cases or at most `1e-12` relative error elsewhere.

Python continues to own the PyTorch `LambdaLR` construction,
`rebase_learning_rate_schedule`, and every operation that reads or mutates
optimizer or scheduler state. This tranche does not move optimizer mechanics,
parameter routing, or trainer lifecycle state into C++.

## Next low-risk tranches

1. Move experiment result writes behind typed TrainVM events and an authority-owned
   projection, then retire Python's direct writable `experiments.db` connection.
2. Port grokking metric aggregation, comparison/report formatting, control-patch
   normalization, and deterministic manifest joins or cache validation.
3. Add cross-language golden fixtures that run both implementations during shadow
   mode and bind decision-version identities into result artifacts.
4. Profile before moving tensor work. Only measured bottlenecks graduate to
   C++/CUDA extensions: fused softcapped cross entropy, RWKV recurrence/channel
   mix, optimizer kernels such as Muon, and qualified FP4/FP8 paths.
5. Add a reflected native registry for composable optimizer, parameter-routing,
   activation, normalization, loss, precision, clipping, accumulation, and
   curriculum contracts. Keep tensor kernels in their qualified runtime; native
   code owns configuration/state validation and composition where useful.

## Boundaries

- Native analysis is read-only diagnostic authority, not run or host authority.
- Native schedules own pure configuration validation and multiplier math, not
  optimizer/scheduler construction or state.
- No arbitrary SQL, Python, shell, command, or expression is admitted through its
  API.
- Database and JSON bounds are part of the fail-closed contract.
- Reflection is used where it removes schema/descriptor boilerplate. Statistical
  kernels stay ordinary typed C++ because reflection adds no value to their inner
  loops.
- GPU work remains disabled during this migration and requires separate parity,
  determinism, gradient, memory, and throughput qualification before use.
