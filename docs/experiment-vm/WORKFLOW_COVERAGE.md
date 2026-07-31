# Workflow coverage and migration checks

This matrix keeps TrainVM from being designed only around one trainer. The original
`/thearray/git/moe-mla` worktree is the compatibility baseline: its runnable entrypoints,
experiment documents, supervisors, checkpoints, evaluation tools, and dashboard launch paths must
all be classified here before ownership migrates. Presence in that tree proves scope, not safety;
only an audited, registered operation receives launch authority. Each existing workflow is reduced
to the proposed node/event/artifact model before implementation begins.

The original package exposes only six console commands; many legitimate workflows are module- or
script-only. Executability is therefore an explicit registry property with a sealed entrypoint and
effect contract, not something inferred from packaging, importability, filenames, or the old
dashboard process list.

| Existing workflow | Current mechanism | Declarative representation | Required adapter capability |
|---|---|---|---|
| Synthetic and architecture/objective A/B campaigns | `rwkv_lab.experiment`, `rwkv_lab.config`, experiment registry, paired/factorial arms and confirmation seeds | parameterized arm graph with successive-halving rungs, paired tapes, confirm nodes, and statistical decision artifact | deterministic arm, rung checkpoint, paired evaluation, campaign aggregation |
| Scratch recurrent LM pretraining | `rwkv_lab.rwkv_pretrain`, `config run-lm`, loop/latent sweep supervisors | current trainer is terminal-checkpoint resumable, not a full safe-point lifecycle: it restores model/optimizer/RNG but does not bind config, data, or parent identity and rewrites `train.jsonl`; a hardened adapter must close those gaps before claiming exact resume | terminal checkpoint grade today; future exact model/optimizer/RNG/data/control resume, distributed checkpoint, token-weighted metrics |
| Distributed recurrent LM handoff | `rwkv_pretrain` FSDP2/DCP supports topology changes but deliberately gives new ranks fresh RNG | explicit reshard/topology handoff node, never labeled exact resume when world size changes | source DCP manifest, old/new topology, reshard receipt, new-rank RNG policy |
| Pretrained recurrent continuation | G1G continuation through `rwkv_pretrain`/`config run-lm` and native-load qualification | immutable parent checkpoint feeds a resumable continuation and native-parity graph | parent/checkpoint identity, exact state resume when configured, PPL/parity report |
| Recurrent optimizer A/B harness | `rwkv_finetune` loads and trains a model but only emits `train.jsonl` and a synthetic terminal checkpoint event | explicitly non-resumable bounded arm; interruption restarts the deterministic arm | input model identity, optimizer config, terminal metric log; no checkpoint claim |
| Transformer MLA-family finetuning | canonical `train_mla`, `train_mla_engram`, staged watchdog; the divergent `dashboard/instrumented/train_mla.py` fork is stale and rejected | weight/optimizer checkpoint and staged handoff; current periodic checkpoints omit RNG/data cursor. MLA, MTP/aux-only, MuToR/FSP/parallel heads, RWKV8 replacement, Engram, and full-backbone unfreeze require distinct topology/freeze/optimizer-group identities | module and optimizer state, complete topology/freeze/group descriptor, baseline/periodic/final PPL, honest partial-resume grade |
| Standalone lexical-memory pretraining | `gpu_engram_prefill`, `engram_lmb_build` over sequential corpora | ordered non-resumable corpus-stage nodes; interruption restarts a stage until optimizer/cursor/RNG state is implemented | patch publication, allocation manifest, N-gram CE; no exact-resume claim |
| MoE-transformer continued pretraining | AO3 prepare/tokenize/pack plus `qwen_ao3_cpt` hybrid QLoRA | immutable data-preparation artifacts feed auto-resuming adapter train/eval nodes | corpus/config fingerprints, packed cursor, atomic adapter checkpoint, PPL |
| Cross-architecture conversion campaign | `convert_gdn_lossless` monkey-patches and self-tests a loaded model; learned full-attention alignment/distillation has trainable candidates | exact conversion publication needs a new typed wrapper and receipt; the existing helper cannot claim an artifact. Trained candidates need target caches, checkpoints, qualification and selection | teacher/content fingerprint, exact-map publication receipt or trained-candidate checkpoint, paired structural/quality metrics |
| Converted-stack assembly and consolidation | `assemble_looped` currently drops input provenance; `distill_consolidate` is non-resumable; `build_memory_targets` does not hash teacher/data content | accepted candidates feed provenance-closing assembly and bounded joint distillation; legacy artifacts are untrusted until re-verified and interrupted consolidation restarts | complete teacher/data/input lineage, best/final converted weights and PPL telemetry |
| Frozen-vision-to-LM adapter training | canonical `vision_train` has strong atomic model/optimizer/sampler/RNG resume plus input/cache fingerprints | exact-resume train/eval loop with cache artifacts, clean-process eval boundary, best/last checkpoints | sampler/RNG resume, feature-cache identity, caption/grounding metrics, eval gallery |
| Native vision head and raw-pixel student training | `vision_native_train` lacks batch/sampler cursor; `vision_rwkv_student_train` restores sampler/RNG but does not bind all manifest/cache/baseline/compressor/native-head inputs | separate operations with non-exact and conditional-compatible grades; identical-length changed inputs must be rejected before resume | explicit input closure, sampler/cursor state where present, honest resume grade, transfer metrics |
| MageFlow cache boundary and resume | shell polls `status.json`, `pgrep`, `jq`; rewrites configs; launches cache and trainer | train node transitions on typed boundary event; plan/build/validate cache nodes; exact resume node | checkpoint manifest, cache-plan and cache-build operations, structured terminal reason |
| MageFlow full-backbone continued pretraining | `mage_flow_pretrain prepare-data|prepare-reddit|plan|train`; Accelerate state and epoch/pack position resume usefully, but checkpoint loading does not enforce config/data/source fingerprints | data preparation and immutable plan feed a compatibility-resume train/eval graph until a hardened adapter seals every input identity | optimizer/RNG/data cursor, enforced preparation/config/source receipts, generated-image evaluation |
| MageFlow routed/terminal/TREAD expert training | `mage_flow_expert_train`, `mage_flow_terminal_train`, migrations and TREAD conversion | route/domain-window graph with cache-span handoffs, component freeze map, migration and joint-eval nodes | route state, domain cursor, cache coverage, expert/base checkpoints, unified gallery |
| MageFlow high-resolution joint refinement | architecture/config document is `design_only_not_launchable` | validation-only document remains rejected by launch authority until an implemented, registered operation exists | design artifact only; never inferred launch authority |
| RADIO/V4H eval restarts | shell loops treat exit 42 inconsistently; some bound rapid restarts and others accept it indefinitely, with no optimizer-step progress check | cyclic train/eval node with `worker.restart_requested`, visit/time bounds, and monotonic optimizer-step guard | clean-process eval safe point, checkpoint ack, no-progress rejection |
| vision representation A/B | Python supervisor launches two arms sequentially and scans JSONL for completion, but publishes no versioned comparison or decision | two arm nodes or a parameterized map template; result artifacts feed a deterministic compare/decision node | run-arm operation, versioned metric/result contract, decision receipt |
| Multi-teacher vision representation distillation | teacher caches, canonical compressor, native projection and raw-pixel student supervisors | staged cache, compressor, native-head, student, and transfer graph with manifest-bound handoffs | multi-teacher cache lineage, representation checkpoint, reconstruction/relational/caption eval |
| External video/audio LoRA orchestration | `ltx23_lora plan|prepare|train|run` records an upstream revision but may accept unknown/dirty HEAD, overwrite receipts/config, and resume from path existence alone | new wrapper must require an expected clean revision and immutable config/input/cache/checkpoint receipts before preprocess, train, validation, or resume | exact clean external revision, structured config/input hashes, upstream resume contract, cache/checkpoint/video receipts |
| GPU launch queue | Go table plus PID liveness and optional start-next | queued desired state plus exclusive resource lease | no trainer-specific capability |
| Multi-GPU and campaign device-slot scheduling | FSDP2/DCP workers and campaign-local device allocators | topology-aware resource leases, per-device child attempts, bounded replica/slot allocation | distributed rendezvous identity, reshardable checkpoint, device-slot release receipt |
| dashboard live tuning | Go and Python share SQLite numeric rows | versioned typed control patch, safe-point application, effective-value ack | operation control descriptor and worker safe-point SDK |
| checkpoints, stop, pause, and resume | signal handling is inconsistent; only specific operations are known safe and many are stop-only or non-resumable | distinct adapter capabilities for checkpoint-now, graceful stop, pause-with-retained-state, resource-releasing pause, and exact resume | operation-declared safe point and semantic state manifest; never inferred from logs/files |
| Single-run RLVR | `rlvr_train` with GSPO, Dr.GRPO, or DAPO and typed verifiers; current `--resume` does not enforce saved config/task/verifier/reference identity | optional cold-start SFT feeds rollout/train/eval nodes under token/time budgets; resume remains compatibility-grade until all identities are sealed | rollout state, verifier boundary, reference policy, config/task/verifier hashes, reward/pass@k and regression report |
| RLVR campaign and recursive RLVR | `rlvr_campaign`, dashboard launcher, `recursive_improve`; completed arms are skipped and incomplete arms restart | algorithm-by-seed arm graph plus bounded recursive proposal/verify/accept loop; external verifier/proposal argv are untrusted process boundaries until registered and sandboxed | equal-budget arms, sealed verifier/proposal, immutable parent pointer, rollback and lineage |
| Typed adapter post-training | validated data feeds SFT/DPO/KTO/ORPO/SimPO/RM/PRM with LoRA or NF4 QLoRA | deterministic but non-resumable objective arm publishes a final adapter or reward head; interruption restarts the arm | role/reset masks, equal scored-token accounting, final adapter/head, calibration/family metrics |
| Paired post-training and adapter recursion | `posttrain_campaign`, dashboard campaign, `adapter_recursive`; its command hash does not detect same-path dataset/checkpoint content changes | completed-arm resume, explore/confirm graph with isolated adapters and immutable recursive parents; incomplete arms restart from their original seed | equal-token accounting, content-addressed inputs plus command hash, promotion receipt, rejected-artifact retention |
| Training-kernel and production-serving qualification | `posttrain_kernels`, `production_kernels`, dashboard qualification | parity/determinism/gradient/state gates precede speed and optional checkpoint-bound serving node | portable/native backend evidence, compile identity, launch/memory/speed report, optional `.pt2` |
| Runtime and operator profiling | vision Chrome/operator traces and timing/memory profiles, production/megakernel summaries, MageFlow CUDA-event telemetry; LTX wrapper has no native profiling contract | bounded generic profile node binds exact code/runtime/checkpoint/cache identities and declared step range, separates instrumented timing from baseline qualification, and publishes restricted raw trace plus summary | backend capability, step-range receipt, trace sensitivity, deterministic comparison artifact |
| Dataset validation/versioning and preference capture | post-training data tools and dashboard inspect/compare; current preference action directly appends JSONL | immutable dataset-version and checkpoint-compare nodes; TrainVM must add a hashed append receipt before owning preference capture | data hash/schema report and comparison artifact; legacy preference append is observed but not trusted authority |
| Safe model/export bundle | `export_bundle` verifies checkpoint, tokenizer, template, adapter, dataset and promotion inputs | explicit verification node publishes a new content-addressed bundle; never overwrites parent | complete lineage manifest and file hashes; no publish authority |
| Corpus acquisition, tokenization, packing, and feature-cache construction | AO3/Qwen preparation, image/vision manifest builders, teacher/representation caches, exact-dedup and quality tools | immutable preprocess/cache DAG whose membership, ordering, source content, tool identity, and output manifest are receipts rather than inferred paths | bounded network/host effects, input/output content hashes, deterministic cursor, cache validation |
| Text generation, decoding evaluation, and recurrent serving | `generate`, `decoding_eval`, `recurrent_serving`, checkpoint-bound megakernel/AOT paths | read-only or serving-profile operations bound to an exact checkpoint and backend qualification receipt | checkpoint/tokenizer/template identity, decoding policy, state/parity/latency evidence |
| External budgeted teacher and captioning jobs | `kimi_teacher` plus image captioning supervisors | explicit network-budgeted external operation with frozen request manifest and append-only result receipts | provider/model revision, request identity, retry/budget policy, redacted provenance, result hash |
| Read-only diagnostics and analysis | eval baselines, circuit/loop probes, grokking metrics, calibration, quality and parity audits | checkpoint-bound read-only nodes that may publish metrics/reports but can never acquire training or mutation authority by importability alone | input identities, bounded resources, deterministic/report contract |
| eval galleries | trainer filesystem conventions and dashboard path discovery | append-only image-gallery artifact with generated/target pairs and step index | gallery manifest publisher |
| legacy runs | JSONL ingestion, PID discovery, and an incomplete dashboard process-name allowlist that misses legitimate RWKV, MageFlow, vision, and experiment processes | compatibility observer emits canonical events in shadow mode; new ownership comes only from the audited operation registry, never the legacy allowlist | legacy translator only; cannot claim exact ownership, GPU idleness, or launch safety |
| CLI supervisors versus library-only research oracles | shell/Python watchdogs may rewrite configs, call host services, inspect/kill processes, or use local locks; capability-panel functions have ambiguous launch status | effect-audit and decompose safe supervisor behavior into typed operations; library/design-only entries require a registered operation wrapper | explicit launchability/effect class and host authority; no shell wrapping or inferred executable |

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

## Decisions that require measured prototypes

- SQLite event throughput and metric compaction thresholds.
- gRPC Python safe-point polling overhead at the current images/second rate.
- process adoption reliability using pidfd plus worker launch identity on Linux.
- GCC reflection/compiler upgrade compatibility against the golden plan and descriptor corpus.
- whether artifact payloads need a local content-addressed store or only content-addressed manifests
  referencing existing run directories.

None of these measurements changes the persisted schema-first boundary.
