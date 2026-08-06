# AutoMegaKernel static-validator compatibility audit

Status: source-pinned CPU audit; no GPU execution or performance qualification

## Source boundary

- Repository: `https://github.com/RightNow-AI/AutoMegaKernel`
- Audited commit: `a514bbc20a03bbf698a17443f8f14a27a617fc10`
- Commit date and subject: `2026-06-29`, `Throughput M4: amortize the GEMM weight read over B rows - batched decode beats B-separate`
- Git-archive SHA-256: `ed7e83c906be8a1f4940db451d824a8a0931f64541e0e331c620a63440cafa81`
- License: MIT; audited `LICENSE` SHA-256
  `468e1c94fd1df17861656d2d03d3f01e2f93cd39aac50b6c89208f94d8db5033`
- Static validator: `schedule/ir.py`; SHA-256
  `d393b0a8c226bb4ab6a0b13410b0bb3b224fe63899444d4f2b59c1a7f5276fd6`
- Adversarial regressions: `tests/test_validator_races.py`; SHA-256
  `ccfe9b850985241a1aa4e5739d78268b0782e58c0e478a6223598ffa93fde6a1`

The audit treats the commit, not the repository's moving README, as authority. No upstream code is
vendored into TrainVM by this change.

## Reproduction

On a clean checkout of the pinned commit, with Python 3.12 and `uv` available:

```bash
git clone https://github.com/RightNow-AI/AutoMegaKernel.git
cd AutoMegaKernel
git checkout --detach a514bbc20a03bbf698a17443f8f14a27a617fc10
uv run --no-sync pytest -q \
  tests/test_ir_smoke.py \
  tests/test_validator_races.py \
  tests/test_robustness.py \
  tests/test_abi_sync.py
```

Observed on the TrainVM host on 2026-08-06: `27 passed` in `0.19s`; environment creation plus test
startup completed in `5.48s`. This command is CPU-only and does not import or build the CUDA VM.

A broader CPU reference/lowering set was also run:

```bash
uv run --no-sync pytest -q \
  tests/test_lower.py \
  tests/test_search.py \
  tests/test_fused.py \
  tests/test_hf_import.py \
  tests/test_ir_smoke.py \
  tests/test_validator_races.py \
  tests/test_abi_sync.py
```

Observed: `43 passed, 1 failed`. The reproducible failure is
`tests/test_search.py::test_search_rejects_invalid_configs_without_crashing`: the test deliberately
invalidates schedules with `N_tile <= 64`, while the pinned `default_config()` now selects
`N_tile=64`. The search correctly rejects that default, but the test still requires a non-null
`default_score_us`. This does not falsify the individual static-validator regressions, but it means
the pinned upstream CPU suite is not green and its end-to-end search-safety assertion is stale.

The validator's stronger claim that malformed programs always return `REJECTED` is also false at
this pin. Starting from `tests.test_ir_smoke.build_valid_chain()`, changing buffer 0's ID and the
matching task input to 100 makes `validate()` raise `IndexError`; changing task 0's ID to 100 does
the same, while task ID -1 raises `ValueError: negative shift count`. More seriously, changing the
first counter and its producer/wait references to either -1 or 100 returns `ok=True`: sparse counter
IDs pass validation even though the ABI/runtime uses counter IDs as array indices. The implementation
builds ID sets but later assumes externally stored IDs are dense canonical list positions. A TrainVM
port must validate or canonicalize IDs before any indexed access, and its error-return contract needs
malformed sparse, duplicate, and negative ID tests for every entity kind.

The paper result claiming 7,160 schedules and zero false accepts is **not reproducible from the OSS
tree**. `EXPERIMENTS.md` names `paper/exp_validator.py`, then explicitly says that the driver is a
paper artifact and is not shipped. Only the result JSON is present, at SHA-256
`0717a7db46b59ef84057408b0b3836a28c9592393194a451eec24ea9221f0e57`. TrainVM therefore records
the 7,160-schedule result as an upstream claim, not qualification evidence.

## Typed IR and validation inventory

The useful boundary is a dependency-light, versioned task-DAG IR:

- `MegakernelProgram` owns typed buffers, monotonic counters, tasks, physical page allocation,
  schedule configuration, target data, and IR/ABI versions.
- A task names one instruction opcode, bounded input/output buffer IDs, exactly one completion
  counter, bounded static waits, scalar parameters, and an optional SM assignment.
- `ScheduleConfig` separates tiling, fusion hints, SM placement, pipeline depth, page policy, block
  size, and dynamic shared memory from the generated task graph.
- `validate()` returns a structured accepted/rejected result rather than launching or raising for
  ordinary invalid programs.

Invariant-to-test coverage at the pinned commit:

| Validator invariant | Upstream executable coverage | Audit disposition |
|---|---|---|
| Buffer/counter references, opcode arity, ABI input/output/wait/rank caps, required parameter keys | `test_ir_smoke`, `test_validator_races::test_capacity_overflow_is_rejected`, `test_abi_sync` | Adapt the bounded typed records and ABI drift test. Add malformed-ID and every-cap boundary cases in TrainVM. |
| Positive/satisfiable wait thresholds and self-wait refusal | `test_ir_smoke`; generated mutants are claimed but their driver is absent | Adapt, with an independent TrainVM generator rather than accepting the result JSON. |
| Producer/consumer DAG acyclicity and stack-safe cycle witness | `test_ir_smoke::test_cycle_is_rejected`, `test_validator_races::test_huge_cycle_does_not_crash_validator` | Adopt the invariant; reimplement in typed C++ and keep a bounded witness. |
| Shared-counter waits are all-of-N joins | `test_partial_wait_on_shared_counter_is_rejected`, `test_two_producers_one_consumer_partial_is_rejected` | Adopt the invariant and counterexample tests. |
| Transitive read-after-write/KV happens-before | `test_missing_happens_before_edge_is_rejected`, `test_kv_read_before_append_is_rejected` | Adopt the invariant, but remove the upstream task-count escape hatch before use. |
| Overlapping multi-writer WAW refusal; disjoint tiled writers accepted | `test_two_unordered_writers_one_buffer_is_rejected`, overlapping/disjoint GEMV tests | Adapt. TrainVM needs an opcode-independent region algebra; AMK only proves regions for selected tile forms. |
| Per-SM serial queues are a topological extension and SM IDs fit the target | Search tests exercise placement, but no direct adversarial validator regression was found | Adapt only after direct queue-order, absent-target, and live-device-binding tests exist. |
| On-chip buffers never cross SMs | Validator code exists; no direct CPU regression was found | Do not adopt yet. Add positive and negative placement cases first. |
| Physical-page reuse has fully ordered live ranges | Validator code exists; no direct CPU regression was found | Do not adopt yet. Require read/write live-range and alias counterexamples. |
| Every output is produced and target labels are coherent | Output reachability is enforced; target-name mismatch is only a warning | Adapt output reachability. Replace target labels with TrainVM's host-inventory and fenced-resource identity. |

## Fail-closed gaps

The following prevent direct adoption as a TrainVM safety authority:

1. Sparse/noncanonical buffer and task IDs can raise `IndexError`/`ValueError` from `validate()`
   instead of returning `REJECTED`, and sparse or negative counter IDs can be accepted. This
   contradicts both the clean-signal and ABI-safety contracts. Untrusted serialized IR must not
   reach graph validation until identity structure is proven.
2. For programs above `_PROVENANCE_MAX_TASKS`, transitive RAW and WAW checks are skipped with a
   warning and the program can still be accepted. The upstream hardening document identifies this
   as a hang/miscompute risk for large models. TrainVM may not turn a safety proof into a warning
   based on graph size; it must use a scalable proof or reject.
3. Per-op semantic bounds such as positive GEMV dimensions, attention windows, RoPE dimensions,
   token/position limits, loader arena bounds, and a device-side wait watchdog remain in the
   upstream backlog. Type-correct int32 values are not necessarily semantically safe values.
4. `ScheduleConfig`, its JSON schema, search space, CPU evaluation, and CUDA loader disagree about
   `threads_per_block=512`: the schema permits it, the search no longer proposes it, CPU evaluation
   can label it valid, and the CUDA loader rejects it because it previously deadlocked. A single
   reflected descriptor must drive all four surfaces in TrainVM.
5. Unknown configuration fields are silently dropped for additive compatibility. TrainVM experiment
   authoring rejects unused or unknown configuration; silent drops would make two author intents
   execute the same program.
6. The target table is static data and the CUDA build derives code generation from the live device,
   but the audited path does not prove that the selected target record is the live fenced device.
   TrainVM must bind architecture, occupancy, limits, driver, and measured evidence to hostd's
   resource authority.
7. The advertised 7,160-case independent validator driver is absent, and its documented dynamic
   oracle partly shares AMK implementation code. The eager CPU reference is useful evidence, but it
   is not a substitute for a shipped independent adversarial generator.

## Model, workload, and architecture boundary

- This is an **inference-forward** persistent megakernel system. There is no backward graph,
  gradient synchronization, optimizer update, checkpoint trajectory, or training qualification.
  It cannot be inserted into TrainVM training merely because its forward validator is useful.
- The importer is a bias-free, dense, RMSNorm, full/default-RoPE, SiLU/SwiGLU Llama-style decoder
  template with GQA. It rejects MoE, MLA, projection bias, non-SiLU activations, and scaled RoPE.
  Sliding-window attention, fused QKV, partial rotary, and nonuniform/per-layer shapes are outside
  the proven contract. RWKV, MageFlow, multimodal encoders, and arbitrary transformers are not
  supported.
- Dynamic shapes, continuous batching, and MoE routing remain a placeholder package. Current
  decode is one forward/token per cooperative launch; a separate persistent multi-token path has
  different watchdog exposure.
- The CUDA source derives `sm_XX` from the live device. The `cp.async` path is gated to SM80+;
  older SM75 uses the synchronous path. The registry contains SM75, SM80, SM86, SM89, SM90, SM100,
  and SM120 targets. The repository reports measured runs on SM80, SM90, and SM120, but this audit
  performed no device execution and makes no performance or cross-architecture correctness claim.
- Cooperative-launch occupancy is checked before launch and block size is currently capped at 256
  in the loader due a measured 512-thread deadlock. A device-side spin watchdog is not implemented.

## Adoption decisions

| Subsystem | Decision | TrainVM action |
|---|---|---|
| Versioned task-DAG, monotonic counter/wait model, and structured rejection receipt | **adapt** | Define a TrainVM-owned reflected C++ IR with closed op/state domains and exact serialization. Do not import Python dataclasses as authority. |
| DAG, wait-satisfiability, all-join, RAW/KV, and WAW invariants | **adapt** | Port the algorithms and adversarial counterexamples, then add scalable/no-skip proofs, semantic bounds, alias/SM tests, and independent fuzz generation. |
| Upstream paper soundness result and result JSON | **reject as qualification** | Retain as a source note only. Require the generator, seeds, oracle implementation, and immutable output before making a TrainVM claim. |
| Static GPU target registry and performance numbers | **reject as authority** | Use hostd's fenced live inventory and TrainVM qualification receipts. Upstream numbers may guide hypotheses only. |
| Llama importer and forward instruction set | **defer** | Consider only for a future sealed Llama serving/inference adapter; it is neither generic nor a training backend. |
| CUDA persistent VM/megakernel | **defer** | No training adoption until backward/state/interrupt/watchdog semantics exist and pass device qualification per architecture. |
| Agent schedule-search loop | **adapt conceptually** | Keep declarative bounded proposals and correctness-before-timing, but execute them through TrainVM experiment/qualification state machines with no raw code or silent config fields. |
| MIT source reuse | **allowed with attribution** | If code is later copied, preserve the MIT notice and bind the exact upstream commit in source dispositions. No code was copied in this audit. |

The near-term value is the validator's counterexample catalog and the separation between a typed
schedule and its executor. It is not a shortcut to a generic training megakernel. A follow-on may
prototype a TrainVM-owned validator only after the graph-size fail-open, semantic bounds, alias/SM
coverage, descriptor drift, and missing independent corpus are closed.
