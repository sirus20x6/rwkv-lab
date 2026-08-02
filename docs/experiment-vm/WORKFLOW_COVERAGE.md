# Workflow coverage and migration checks

This matrix keeps TrainVM from being designed only around one trainer. The original
`/thearray/git/moe-mla` worktree is the compatibility baseline: its runnable entrypoints,
experiment documents, supervisors, checkpoints, evaluation tools, and dashboard launch paths must
all be classified here before ownership migrates. Presence in that tree proves scope, not safety;
only an audited, registered operation receives launch authority. Each existing workflow is reduced
to the proposed node/event/artifact model before implementation begins.

The reviewed executable baseline is
[`compatibility-workflows.v1.json`](compatibility-workflows.v1.json), validated under the
non-authoritative contract in [`COMPATIBILITY_CATALOG.md`](COMPATIBILITY_CATALOG.md). Its exact ID
set and referenced source bytes are native test gates; it cannot supply an adapter, executable, or
host launch profile.

The original package exposes only six console commands; many legitimate workflows are module- or
script-only. All six console surfaces, their effectfully distinct subcommands, the reviewed
supported module/script entrypoints and essential graph libraries, concrete trainer-specific vision
profiles, and legacy dashboard launch/control paths are now in the evidence catalog. The dashboard
side of that surface is closed rather than sampled: every `POST` route the legacy router registers
has exactly one record, and the router registration bytes are bound so a new or retired mutation
path cannot pass review silently. The inclusion
rule is intentionally narrower than “has `__main__`”: a synthetic smoke block is not executable
workflow evidence, while an essential component used by a supported graph is recorded only as
`library_only`. Executability is still an explicit authority-registry property with a sealed
entrypoint and effect contract, not something inferred from this inventory, packaging,
importability, filenames, or the old dashboard process list.

## Lifecycle authority and current resume grades

The authority registry uses a required, reflected `lifecycle` object on every
operation profile. Its fixed fields are `stateful`, `graceful_stop`,
`checkpoint_now`, `pause_keep_resources`, `pause_release_resources`, `compile`,
`warmup`, `qualify`, `profile`, and `resume_grade`. The closed grades mean:

| Grade | Authority meaning |
|---|---|
| `none` | no operation-state restart or checkpoint-resume claim; mandatory for stateless operations |
| `restart_only` | interruption restarts the operation from its immutable inputs or parent |
| `terminal_checkpoint` | a successful terminal invocation publishes reusable state, but there is no mid-run safe-point claim |
| `compatible` | a checkpoint can seed continuation, but omitted state or identity prevents trajectory-equivalence claims |
| `exact` | the operation implements the complete, identity-bound checkpoint and resume contract and supports checkpoint-now |

The following classifications document the audited original implementation;
they are test data and migration requirements, not registration or launch
authority for the legacy entrypoints:

| Original operation family | Current grade | Reason |
|---|---|---|
| `rwkv_pretrain` | `terminal_checkpoint` | the baseline non-distributed AdamW/PowerCool path has a closed TrainVM v1 config, recursive static-input content authority, terminal immutable publication, live metrics/heartbeats, and bounded profiling; research levers remain legacy-only, and there is still no periodic/signal checkpoint |
| `rwkv_finetune`, consolidation and current RWKV post-training arms | `restart_only` | interruption restarts a bounded operation from immutable inputs |
| `gpu_engram_prefill` | `compatible` | an Engram checkpoint can warm-start weights, but optimizer, RNG, corpus cursor, and step restart |
| `attn_L3_poc` | `terminal_checkpoint` | the standalone alignment/distillation trainer emits only a terminal `core_final.pt` |
| `convert_train` | `compatible` | signal checkpoints and optimizer warm-start exist, but the loop, RNG, and data cursor restart |
| canonical `train_mla` / `train_mla_engram` | `compatible` | checkpoints omit RNG and data cursor state |
| `qwen_ao3_cpt` | `compatible` | atomic cursor/config/RNG checkpoint and signal safe point exist, and packed-token/base-model trees are now content-bound; exact trajectory equivalence remains unproven |
| MageFlow full-backbone pretrain and routed expert trainer | `compatible` | static manifests, payload trees, source checkpoints, and caches are recursively content-bound; complete exact-resume trajectory evidence remains outstanding |
| MageFlow terminal/TREAD trainer | `compatible` | static image/caption roots, source checkpoints, schedules, caches, and expert banks are recursively bound in addition to strong cursor/RNG state; exact trajectory evidence remains outstanding |
| canonical `vision_train` and vision teacher compressor | `exact` candidates | atomic sampler/RNG and manifest-bound state; require the same golden gate |
| vision native head and raw-pixel student | `compatible` | missing cursor or external-input identity prevents exactness |
| direct RLVR and external LTX wrapper | `compatible` | resume state or upstream resume exists, but verifier/config/revision/input closure is incomplete |
| deterministic materializers, validation, generation, evaluation, and diagnostics | `none` | stateless/replay semantics are represented by effect and idempotency, not a fictitious training-resume grade |
| resumable remote acquisition and inventory/dedup scanners | `compatible` candidates | SQLite cursors, partial transfers, ETags, API pagination, and scan state permit useful continuation, but mutable remote identity and complete replay state must be sealed before stronger claims |

An exact-recovery plan may mix an exact stateful trainer with stateless process
nodes such as cache builders. It fails registry validation if any reachable
stateful process operation is graded `compatible`, `terminal_checkpoint`,
`restart_only`, or `none`.

| Existing workflow | Current mechanism | Declarative representation | Required adapter capability |
|---|---|---|---|
| Synthetic and architecture/objective A/B campaigns | distinct `rwkv_lab.experiment` and `rwkv_lab.config` campaigns, plus checked-in loop, factored-loop, latent, fixed-step-latent, and G1G sweeps | parameterized arm graph with successive-halving rungs, paired tapes, confirm nodes, and statistical decision artifact | deterministic arm, rung checkpoint, paired evaluation, campaign aggregation |
| Scratch recurrent LM pretraining | `rwkv_lab.rwkv_pretrain`, `config run-lm`, loop/latent sweep supervisors | the closed baseline TrainVM adapter binds its scalar config and recursively verified corpus bytes, publishes model/optimizer/RNG/component state only at terminal completion, and exposes live scalar metrics and GPU trace hooks; it remains terminal-checkpoint grade because mid-run control state and the broader research switches are not closed | terminal checkpoint grade today; future exact model/optimizer/RNG/data/control resume, distributed checkpoint, token-weighted metrics, and versioned research-topology adapters |
| Distributed recurrent LM handoff | `rwkv_pretrain` FSDP2/DCP supports topology changes but deliberately gives new ranks fresh RNG | explicit reshard/topology handoff node, never labeled exact resume when world size changes | source DCP manifest, old/new topology, reshard receipt, new-rank RNG policy |
| Pretrained recurrent continuation | G1G continuation through `rwkv_pretrain`/`config run-lm` and native-load qualification | immutable parent checkpoint feeds a resumable continuation and native-parity graph | parent/checkpoint identity, exact state resume when configured, PPL/parity report |
| Recurrent optimizer A/B harness | `rwkv_finetune` loads and trains a model but only emits `train.jsonl` and a synthetic terminal checkpoint event; its default budget is wall-clock based | explicitly non-resumable seeded arm; interruption restarts it. Deterministic comparison additionally requires a fixed step budget and deterministic backend | input model identity, optimizer config, budget kind/value, terminal metric log; no checkpoint claim |
| Transformer MLA-family finetuning | canonical `train_mla` now backs eight exact TrainVM v1 profiles for MLA, MTP-only, MuToR auxiliary-only, FSP auxiliary-only, parallel-head auxiliary-only, RWKV8 replacement, Engram, and full-backbone unfreeze; `train_mla_engram` and the divergent `dashboard/instrumented/train_mla.py` remain compatibility evidence only and the legacy dashboard cannot launch either MLA script | each authority key closes its topology/freeze policy, executes the resolved optimizer/objective/accumulation/clipping/decay components, emits live train/eval metrics and heartbeats, services checkpoint/pause only at optimizer boundaries, and publishes a compatible-grade checkpoint because RNG and the exact random-window cursor remain absent | implemented compatible-grade model/optimizer/topology/group/schedule/control/component state; exact RNG and data-cursor resume remains a future profile version |
| Lexical-memory allocation preprocessing | `engram_lmb_build freq|alloc` builds frequency and allocation artifacts but does not train | replay-safe stateless preprocessing nodes publish content-bound allocation manifests | corpus/tool identity, frequency artifact, allocation manifest |
| Standalone lexical-memory pretraining | `gpu_engram_prefill` trains over sequential corpora and accepts Engram-weight warm starts | ordered compatibility-grade corpus-stage nodes; a warm-started operation resets optimizer, RNG, corpus cursor, and step until complete state is implemented | patch publication, allocation manifest, parent Engram weights, N-gram CE; no exact-resume claim |
| MoE-transformer continued pretraining | AO3 prepare/tokenize/pack plus `qwen_ao3_cpt` hybrid QLoRA | immutable data-preparation artifacts feed auto-resuming adapter train/eval nodes | corpus/config fingerprints, packed cursor, atomic adapter checkpoint, PPL |
| Cross-architecture conversion campaign | `convert_gdn_lossless` monkey-patches and self-tests a loaded model; `attn_L3_poc` performs terminal-only block-MSE/top-k-logit-KL alignment; `convert_train` is the compatibility-grade per-layer GDN/full-attention-to-RWKV trainer | exact conversion publication needs a new typed wrapper and receipt; the existing helper cannot claim an artifact. Terminal and compatibility-grade trained candidates need target caches, honest checkpoint semantics, qualification, and selection | teacher/content fingerprint, exact-map publication receipt or trained-candidate checkpoint, paired structural/quality metrics |
| Per-layer conversion supervision and selection | `drive_isolation` sequences 24 layers with opportunistic warm starts; `gdn_sweep.sh`, `rel_sweep.sh`, and `gate_ab.sh` launch, monitor, terminate, and select candidates using different metrics | typed per-layer map/fan-out nodes feed an explicit metric-specific selector; early stop, stall, failure, retry, and ordered assembly are durable decisions instead of parsed logs or an in-memory done list | layer/map identity, warm-start parent, budget, early-stop/stall receipt, selection metric/policy, ordered candidate hashes |
| Converted-stack assembly and consolidation | `assemble_looped` currently drops input provenance; `distill_consolidate` is non-resumable; `build_memory_targets` does not hash teacher/data content | accepted candidates feed provenance-closing assembly and bounded joint distillation; legacy artifacts are untrusted until re-verified and interrupted consolidation restarts | complete teacher/data/input lineage, best/final converted weights and PPL telemetry |
| In-memory expert merge oracle | `distillation_merge` merges experts and performs logit-loss tolerance acceptance but has no entrypoint or published artifact contract | library-only/nonlaunchable until a typed merge node seals ordered expert identities, weights, output checkpoint, and bound evaluation | ordered expert hashes and weights, merge implementation identity, checkpoint publication, acceptance receipt |
| Frozen-vision-to-LM adapter training | canonical `vision_train` has strong atomic model/optimizer/sampler/RNG resume plus input/cache fingerprints | exact-resume train/eval loop with cache artifacts, clean-process eval boundary, best/last checkpoints | sampler/RNG resume, feature-cache identity, caption/grounding metrics, eval gallery |
| Native vision head and raw-pixel student training | `vision_native_train` lacks batch/sampler cursor; `vision_rwkv_student_train` restores sampler/RNG but does not bind all manifest/cache/baseline/compressor/native-head inputs | separate operations with non-exact and conditional-compatible grades; identical-length changed inputs must be rejected before resume | explicit input closure, sampler/cursor state where present, honest resume grade, transfer metrics |
| MageFlow cache boundary and resume | shell polls `status.json`, `pgrep`, `jq`; rewrites configs; launches cache and trainer | train node transitions on typed boundary event; plan/build/validate cache nodes; exact resume node | checkpoint manifest, cache-plan and cache-build operations, structured terminal reason |
| MageFlow full-backbone continued pretraining | separately reviewed `mage_flow_pretrain prepare-data|prepare-reddit`, `plan`, and `train` effects; Accelerate state and epoch/pack position resume usefully, but checkpoint loading does not enforce config/data/source fingerprints | data preparation and immutable plan feed a compatibility-resume train/eval graph until a hardened adapter seals every input identity | optimizer/RNG/data cursor, enforced preparation/config/source receipts, generated-image evaluation |
| MageFlow routed/terminal/TREAD expert training | separately reviewed expert plan/cache/train and terminal prepare/cache/cache-span/domain-window/train effects, plus the terminal-expert migration, TREAD conversion, and cache-resume supervisors | route/domain-window graph with cache-span handoffs, component freeze map, migration and joint-eval nodes | route state, domain cursor, cache coverage, expert/base checkpoints, unified gallery |
| MageFlow high-resolution joint refinement | architecture/config document is `design_only_not_launchable` | validation-only document remains rejected by launch authority until an implemented, registered operation exists | design artifact only; never inferred launch authority |
| RADIO/V4H eval restarts | shell loops treat exit 42 inconsistently; some bound rapid restarts and others accept it indefinitely, with no optimizer-step progress check | cyclic train/eval node with `worker.restart_requested`, visit/time bounds, and monotonic optimizer-step guard | clean-process eval safe point, checkpoint ack, no-progress rejection |
| vision representation A/B | Python supervisor launches two arms sequentially and scans JSONL for completion, but publishes no versioned comparison or decision | two arm nodes or a parameterized map template; result artifacts feed a deterministic compare/decision node | run-arm operation, versioned metric/result contract, decision receipt |
| Multi-teacher vision representation distillation | teacher caches, canonical compressor, native projection and raw-pixel student supervisors | staged cache, compressor, native-head, student, and transfer graph with manifest-bound handoffs | multi-teacher cache lineage, representation checkpoint, reconstruction/relational/caption eval |
| External video/audio LoRA orchestration | separately reviewed `ltx23_lora plan`, `prepare`, `train`, and composite `run` effects record an upstream revision but may accept unknown/dirty HEAD, overwrite receipts/config, and resume from path existence alone | new wrapper must require an expected clean revision and immutable config/input/cache/checkpoint receipts before preprocess, train, validation, or resume | exact clean external revision, structured config/input hashes, upstream resume contract, cache/checkpoint/video receipts |
| GPU launch queue | Go table plus PID liveness and optional start-next | queued desired state plus exclusive resource lease | no trainer-specific capability |
| Multi-GPU and campaign device-slot scheduling | FSDP2/DCP workers and campaign-local device allocators | topology-aware resource leases, per-device child attempts, bounded replica/slot allocation | distributed rendezvous identity, reshardable checkpoint, device-slot release receipt |
| dashboard live tuning | Go and Python share SQLite numeric rows | versioned typed control patch, safe-point application, effective-value ack | operation control descriptor and worker safe-point SDK |
| checkpoints, stop, pause, and resume | signal handling is inconsistent; only specific operations are known safe and many are stop-only or non-resumable | distinct adapter capabilities for checkpoint-now, graceful stop, pause-with-retained-state, resource-releasing pause, and exact resume | operation-declared safe point and semantic state manifest; never inferred from logs/files |
| Single-run RWKV RLVR | `rlvr_train` requires a self-describing `rwkv_pretrain` checkpoint and supports GSPO, Dr.GRPO, or DAPO with typed verifiers; current `--resume` does not enforce saved config/task/verifier/reference identity | optional cold-start SFT feeds rollout/train/eval nodes under token/time budgets; resume remains compatibility-grade until all identities are sealed | rollout state, verifier boundary, reference policy, config/task/verifier hashes, reward/pass@k and regression report |
| RLVR campaign and recursive RLVR | `rlvr_campaign`, dashboard launcher, `recursive_improve`; completed arms are skipped and incomplete arms restart | algorithm-by-seed arm graph plus bounded recursive proposal/verify/accept loop; external verifier/proposal argv are untrusted process boundaries until registered and sandboxed | equal-budget arms, sealed verifier/proposal, immutable parent pointer, rollback and lineage |
| Typed RWKV adapter post-training | `posttrain_train` currently applies SFT/DPO/KTO/ORPO/SimPO/RM/PRM to RWKV checkpoints with LoRA or NF4 QLoRA | deterministic but non-resumable objective arm publishes a final adapter or reward head; interruption restarts the arm. Other model families require their own registered objective adapters | role/reset masks, equal scored-token accounting, final adapter/head, calibration/family metrics |
| Paired post-training and adapter recursion | `posttrain_campaign`, dashboard campaign, `adapter_recursive`; its command hash does not detect same-path dataset/checkpoint content changes | completed-arm resume, explore/confirm graph with isolated adapters and immutable recursive parents; incomplete arms restart from their original seed | equal-token accounting, content-addressed inputs plus command hash, promotion receipt, rejected-artifact retention |
| Guarded adapter consolidation | `adapter_consolidation.GuardedAdapterConsolidator` snapshots state/replay but invokes injected train/evaluate callbacks and mutates a candidate receipt during promotion | library-only/nonlaunchable until decomposed into `snapshot -> bounded train -> held-out eval -> human promote` with registered callbacks and immutable receipts | baseline/state/replay hashes, callback identities, candidate hash, held-out eval identity, promotion-time recheck and append-only receipt |
| Reasoning-cache supervision synthesis | `reasoning_cache.run_reasoning_cache` invokes injected generate/summarize/verify callbacks under iteration/token/time limits | library-only/nonlaunchable bounded recursive graph until every callback/model/verifier and output pair is sealed | callback/model/verifier identities, iteration/token/time budget, supervision-pair content receipts |
| Guarded test-time training | `guarded_test_time_train` mutates a live module, snapshots only module state, and invokes arbitrary loss/eval callbacks | library-only/nonlaunchable; future execution uses an isolated disposable model transaction and can never mutate serving state in place | model/input/optimizer/RNG and callback identities, acceptance/rollback receipt, serving-state isolation proof |
| Training-kernel and production-serving qualification | `posttrain_kernels`, `production_kernels`, dashboard qualification | parity/determinism/gradient/state gates precede speed and optional checkpoint-bound serving node | portable/native backend evidence, compile identity, launch/memory/speed report, optional `.pt2` |
| Runtime and operator profiling | vision Chrome/operator traces and timing/memory profiles, production/megakernel summaries, MageFlow CUDA-event telemetry; LTX wrapper has no native profiling contract | bounded generic profile node binds exact code/runtime/checkpoint/cache identities and declared step range, separates instrumented timing from baseline qualification, and publishes restricted raw trace plus summary | backend capability, step-range receipt, trace sensitivity, deterministic comparison artifact |
| Isolated MageFlow optimization benchmarking | `benchmark_mage_flow_runtime.py` mutates modules and process-global compiler, allocator, and quantization state; it reports throughput/memory/OOM/last loss without enforcing parity | one fresh process per variant feeds a compare/decision node; a speedup is rejected unless declared numerical and training-quality thresholds pass | variant/runtime/code identity, pristine-process proof, baseline parity thresholds, speed/memory report and decision receipt |
| Dataset validation/versioning and preference capture | post-training data tools and dashboard inspect/compare; current preference action directly appends JSONL | immutable dataset-version and checkpoint-compare nodes; TrainVM must add a hashed append receipt before owning preference capture | data hash/schema report and comparison artifact; legacy preference append is observed but not trusted authority |
| Safe model/export bundle | legacy `export_bundle` can recursively replace an arbitrary destination and only shallowly parses some dataset/promotion receipts; frozen vision-compressor export writes an overwriteable pickle without a digest manifest | distinct verification and publication nodes write a safe format to a new immutable destination; no legacy path replacement receives publish authority | complete lineage and file hashes, source digest, safe-format receipt, immutable destination, independently verified dataset/promotion evidence |
| Remote dataset acquisition | archive and Civitai downloaders persist cursor, ETag/pagination, partial-transfer, lock, and storage-pressure state against mutable sources | stateful bounded network operations with explicit `waiting_for_space`, retry, source-version, partial-byte, and resume identities | request/source identity, cursor/ETag/page token, partial-byte digest, budget/retry/storage receipt, license/use policy |
| Inventory, dedup, materialization, and cache construction | image inventory persists SQLite scan state; exact dedup may replace source files with hardlinks; deterministic tranche/cache builders publish manifests | separate scanner, mutation, and replay-safe materialization profiles rather than one fictitious stateless builder | source snapshot and scan cursor, duplicate policy, hardlink/filesystem effects, input/output hashes and deterministic ordering |
| Cache qualification and publication | V4H's default verifier proves presence/geometry only; overlay/reuse/finalize/handoff may hardlink, move directories/entries, publish receipts, signal workers, and alter active/frozen/overflow caches | qualification explicitly grades presence/geometry, producer identity, and content integrity; overlay, reuse, finalization, and live handoff are separate receipted mutation operations | qualification grade, producer/source/payload identities, hardlink/move/signal effects, atomic publication and rollback policy |
| Human image review and live dataset revision | quality review can trash files, fsync audit actions, and rewrite live metadata/manifests; aesthetic scoring resumes external inference; cutoff review publishes human policy | versioned review session with immediate audited actions, recoverable trash, asynchronous manifest compaction, registered scoring inference, and explicit policy artifact | reviewer/session/action identity, trash recovery receipt, before/after dataset versions, model revision, score/policy provenance |
| Text generation and model-bound evaluation | `generate` is executable; `decoding_eval` is a callable harness and `recurrent_serving` provides scheduler-neutral state primitives rather than a network service | generation is a registered read-only operation; decoding and recurrent state paging remain library-only until sealed evaluator/scheduler/service wrappers exist | checkpoint/tokenizer/template identity, decoding policy, state/parity/latency evidence |
| External budgeted teacher and captioning jobs | `kimi_teacher` plus image captioning supervisors | explicit network-budgeted external operation with frozen request manifest and append-only result receipts | provider/model revision, request identity, retry/budget policy, redacted provenance, result hash |
| Read-only diagnostics and analysis | eval baselines, circuit/loop probes, grokking metrics, calibration, quality and parity audits | checkpoint-bound read-only nodes that may publish metrics/reports but can never acquire training or mutation authority by importability alone | input identities, bounded resources, deterministic/report contract |
| eval galleries | vision eval atomically rewrites an incomplete per-step document while generating; MageFlow target selection does not bind target-image bytes | mutable attempt-local staging records per-item progress, then exactly one immutable gallery publication binds generated and target digests, held-out dataset version, prompt/seed, and step | staging owner/fence, item completion, generated/target content hashes, held-out dataset identity, immutable gallery manifest |
| Run evidence, promotion, aliases, and retention | `vision_run_evidence` classifies startup/committed/exact recovery and best publication; MageFlow promotion replaces aliases and removes prior promoted directories | run-evidence classification, candidate selection, immutable promotion, alias update, retention, and deletion are distinct receipted operations | checkpoint completeness, evidence policy, candidate hash/metrics, alias CAS, retention/deletion target and recovery policy |
| legacy runs | JSONL ingestion, PID discovery, and an incomplete dashboard process-name allowlist that misses legitimate RWKV, MageFlow, vision, and experiment processes | compatibility observer emits canonical events in shadow mode; new ownership comes only from the audited operation registry, never the legacy allowlist | legacy translator only; cannot claim exact ownership, GPU idleness, or launch safety |
| CLI supervisors versus library-only research oracles | shell/Python watchdogs may rewrite configs, call host services, inspect/kill processes, or use local locks; capability-panel functions have ambiguous launch status | effect-audit and decompose safe supervisor behavior into typed operations; library/design-only entries require a registered operation wrapper | explicit launchability/effect class and host authority; no shell wrapping or inferred executable |

`diffusion_rwkv`, `native_g1g`, Engram LMB, the MLA/layer-swap/SVD components, the ROSA/SMT modules,
and `verify_engram` remain library or diagnostic surfaces in the compatibility baseline.
Importability or a self-test does not turn them into launchable trainers. Module-only trainers such
as `attn_L3_poc`, `convert_train`, and `gpu_engram_prefill` must be registered with sealed
`python -m` entrypoints; they cannot be inferred from the package's six installed console scripts.

## Golden fixture set

Tiny deterministic fake adapters cover distinct semantics without building one fixture per trainer.
They use the real event and artifact schemas but never require a GPU:

| Fixture | Required semantics | Representative workflows |
|---|---|---|
| Exact single-file trainer | optimizer safe points; atomic checkpoint; model/optimizer/RNG/scaler/data/control state; stop/resume equivalence; periodic eval | scratch RWKV, resumable vision/native and AO3-style workers |
| Distributed checkpoint trainer | fake ranks; rendezvous/lease identity; DCP completeness; per-rank RNG; crash during shard publication; reshard plan | FSDP2/DCP training |
| Staged cache plus clean-process eval | plan/build/validate cache; boundary checkpoint; manifest mismatch; bounded restart cycle; paired gallery | MageFlow cache handoff, RADIO/V4H eval, vision teacher caches |
| Segmented-cursor trainer | corpus/shard/rung and alternating-domain cursors; compatible handoff versus exact resume; freeze map | future resumable Engram stages, vision transfer, MageFlow domains, curricula |
| Fan-out campaign | paired tapes, rungs, equal budgets, device slots, retry/timeout, fresh confirmation, rejected artifacts, statistics | LM A/B, conversion, RLVR and post-training campaigns |
| Heterogeneous conversion DAG | one exact remap and one trained candidate; target cache; acceptance; assembly; non-resumable consolidation; verified export | GDN remap plus attention distillation and bundle export |
| Recursive immutable-parent loop | registered proposal/verifier, bounded rounds/bytes/time/tokens, accept/reject/rollback, promotion and resume | recursive RLVR and adapter improvement |
| External revision-pinned runner | plan/preprocess/train/validate, isolated host profile, revision/config receipts, upstream resume and mismatch refusal | LTX-2.3 and official external trainers |
| Terminal-only trainer | checkpoint-now, pause, and exact resume are refused; interruption restarts from its immutable parent | current `rwkv_pretrain` and other bounded compatibility workers |
| Topology-changing checkpoint handoff | same-topology resume can be exact; changed rank count is an explicit reshard with fresh-rank RNG | FSDP2/DCP topology changes |
| Dynamic transformer topology | freeze map, parameter topology, auxiliary heads, and optimizer groups change only through receipted handoffs | MLA/MTP/MuToR/FSP/parallel-head/RWKV8/Engram variants |
| Vision resume-grade suite | exact sampler worker, non-exact native head, and conditional student; same-length changed manifests/caches are rejected | vision trainer, native projection, raw-pixel student |
| External repository mutation | expected commit, dirty tree, overwritten config, changed input, and stale checkpoint are checked before any subprocess | hardened LTX and future upstream trainers |
| Progress-bound restart loop | repeated exit-42 at one optimizer step terminates; increasing steps remain bounded and legal | clean-process RADIO/V4H evaluation |
| Profiling receipt | bounded CPU trace for declared steps binds code/runtime/checkpoint/cache and compares with an unprofiled baseline | generic torch/nsys/ncu lifecycle without requiring GPU in protocol tests |
| Content and conversion lineage | same-path byte mutation invalidates cached completion; in-memory conversion cannot publish; assembly preserves ordered input hashes | AO3/posttrain/RLVR caches and conversion/distillation |
| Resumable image preprocessing | exact and perceptual duplicates, phase interruption, and deterministic SQLite/manifest recovery | large image inventory/dedup/quality pipelines |
| Budgeted external queue | append-only item receipts, request/spend caps, idempotent retry, and credential-reference redaction | Kimi teacher and captioning jobs |
| Nested resource owner | a campaign child inherits the parent lease and blocks a second launch even when its basename is absent from legacy inventory | experiment/config supervisors and all child workers |
| Evaluation and serving | immutable checkpoint produces deterministic metrics/samples and may consume a separately qualified AOT/cache receipt | generation, decoding eval, recurrent serving, megakernel AOT |

Protocol-only fixtures additionally cover a shadow legacy observer that can never issue control,
authority rejection of library-only/design-only/unregistered shell or verifier entries, and stale
fork/code-hash rejection. Artifact schemas are shared variants, not bespoke dashboard handlers.

## Exhaustive source disposition evidence

The compatibility workflow catalog is complemented by two versioned,
evidence-only source inventories. `source-dispositions.scripts.v1.json` binds all
128 top-level `scripts/*.{py,sh}` files; `source-dispositions.rwkv-lab.v1.json`
binds all 165 top-level `src/rwkv_lab/*.py` modules. Each row records exact source
SHA-256, a closed role/effect/resume classification, a canonical entry point,
and compatibility linkage. Optional live-root validation rejects missing, stale,
or byte-drifted sources. Neither catalog grants execution authority.

RWKV compatibility coverage is 68 direct, 60 transitive, 36 uncovered library
or research surfaces, and one runnable direct gap: `rwkv_lab.registry`. Native
TrainVM now covers that module's read-only campaign listing, latest-result
collapse, comparison view, and shared statistical decision helpers. Writable
campaign/result registration and reproducibility-capsule capture remain the
explicit gap; this partial migration grants no permission to launch the Python
module or let a worker share a writable database.

## Acceptance scenarios

The first runtime is not ready to own training until these scenarios pass without manual file edits:

The `trainvm_dashboard_live_e2e` CTest now covers the real non-GPU boundary chain: C++ daemon startup
on its permission-restricted Unix socket, Go HTTP compile and immutable submission, authority-backed
plan readback, revision-fenced semantic diff, lineage-recorded fork, clean shutdown, and native
journal verification. It intentionally runs without hostd and therefore proves that launch-disabled
authoring/replay remains usable; it does not claim worker launch, privileged host recovery, or GPU
training acceptance.

The separate `rwkv_lab_worker_artifact` CTest now closes the non-GPU project-code deployment
boundary: it builds the real worker zipapp twice, verifies exact bytes and every embedded source
digest, loads it under isolated Python with an empty environment, completes a real sealed-bootstrap
gRPC already-completed replay, lowers all twelve real adapter profiles into the v3 host registry, and
materializes the same deployment twice without drift. It
does not replace the outstanding privileged hostd or real trainer qualification scenarios below.

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
   optimizer steps; reject a synthetic restart storm with no progress regardless of exit code.
8. Execute sequential A/B arms from one template and compare their versioned result artifacts.
9. Rebuild every dashboard projection from the immutable journal and obtain the same visible state.
10. Render the experiment editor and control panel from schemas without checking the experiment name.
11. Compile representative scratch-RWKV, transformer-continuation, MageFlow, vision-distillation,
    post-training, and RLVR documents through the same plan/compiler path without family-specific UI
    validation.
12. For adapters declaring exact resume, resume a self-describing fake single-file checkpoint and an
    exact-topology FSDP2/DCP checkpoint directory, proving model, optimizer, RNG, scaler, data cursor,
    schedule segment, and control revision equivalence. The current `rwkv_pretrain` compatibility
    adapter must expose terminal-checkpoint grade, and a world-size-changing DCP operation must be a
    semantic handoff rather than exact resume.
13. Execute a conversion DAG from immutable teacher and target cache through isolated candidates,
    acceptance, provenance-preserving assembly, consolidation, and verified export without path
    discovery. Reject the legacy monkey-patch-only conversion helper as an artifact publisher.
14. Stop and resume an ordered multi-corpus, multi-shard, multi-rung, and alternating-domain workflow;
    each must preserve its distinct stage cursor and reject an incompatible handoff artifact.
15. Run representative SFT, preference, reward-model, RLVR, and recursive campaign graphs with equal
    budget accounting, rejected-artifact retention, immutable promotion/parent receipts, and
    same-path content-mutation rejection.
16. Drive an external revision-pinned trainer through plan, preprocess, train, validation, and resume;
    an unknown/dirty revision, changed input, stale checkpoint, or structured-config mismatch must
    fail before launch.
17. Publish and render adapter, reward-head, conversion-candidate, representation-compressor,
    generated/target gallery, compiled-plan, qualification, and export-bundle artifacts from their
    declared schemas rather than filename conventions.
18. Capture a bounded accelerator trace at a declared step range, attach the exact code/runtime/cache
    namespace, compare it with a baseline, and render it without editing adapter code.
19. Warm every declared curriculum shape on disposable state, restore the initial state, and prove the
    resulting timed run matches a non-warmed reference at its first checkpoint.
20. Attempt to launch an importable library-only research oracle and confirm it is rejected until an
    authority registry supplies an exact typed operation and effect contract.
21. Interrupt remote acquisition during a partial transfer and while storage is exhausted; resume
    only with the same source/cursor/ETag and partial-byte identity, surface `waiting_for_space`
    without lease loss, and reject an experiment whose declared use violates dataset policy.
22. Crash an eval attempt while its staging gallery is incomplete, then resume or discard staging
    without exposing it as published; the final immutable manifest must bind every generated and
    target byte plus held-out dataset, prompt, seed, and checkpoint identities.
23. Refuse to treat a presence/geometry cache check as producer-identity or content-integrity
    qualification, and replay overlay/finalize/handoff operations without duplicating moves,
    signals, hardlinks, or publication receipts.
24. Exercise review-to-trash recovery, manifest compaction, checkpoint promotion/alias CAS,
    retention, and export collision paths; every mutation has an immutable receipt, destructive
    targets are exact, and no legacy recursive replacement receives publication authority.

## Decisions that require measured prototypes

- SQLite event throughput and metric compaction thresholds.
- gRPC Python safe-point polling overhead at the current images/second rate.
- process adoption reliability using pidfd plus worker launch identity on Linux.
- GCC reflection/compiler upgrade compatibility against the golden plan and descriptor corpus.
- whether artifact payloads need a local content-addressed store or only content-addressed manifests
  referencing existing run directories.

None of these measurements changes the persisted schema-first boundary.
