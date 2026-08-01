# Native algorithm migration

Status: first CPU-only tranche implemented on `dashboard/declarative-vm-fsm`.

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

## Next low-risk tranches

1. Move experiment result writes behind typed TrainVM events and an authority-owned
   projection, then retire Python's direct writable `experiments.db` connection.
2. Port grokking metric aggregation, comparison/report formatting, control-patch
   normalization, power/cooling policy calculations, and deterministic manifest
   joins or cache validation.
3. Add cross-language golden fixtures that run both implementations during shadow
   mode and bind decision-version identities into result artifacts.
4. Profile before moving tensor work. Only measured bottlenecks graduate to
   C++/CUDA extensions: fused softcapped cross entropy, RWKV recurrence/channel
   mix, optimizer kernels such as Muon, and qualified FP4/FP8 paths.
5. Add a reflected native registry for composable optimizer, parameter-routing,
   schedule, activation, normalization, loss, precision, clipping, accumulation,
   and curriculum contracts. Keep tensor kernels in their qualified runtime;
   native code owns configuration/state validation and composition where useful.

## Boundaries

- Native analysis is read-only diagnostic authority, not run or host authority.
- No arbitrary SQL, Python, shell, command, or expression is admitted through its
  API.
- Database and JSON bounds are part of the fail-closed contract.
- Reflection is used where it removes schema/descriptor boilerplate. Statistical
  kernels stay ordinary typed C++ because reflection adds no value to their inner
  loops.
- GPU work remains disabled during this migration and requires separate parity,
  determinism, gradient, memory, and throughput qualification before use.
