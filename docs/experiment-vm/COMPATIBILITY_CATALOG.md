# Compatibility workflow catalog

[`compatibility-workflows.v1.json`](compatibility-workflows.v1.json) is the machine-readable,
reviewed inventory of training and adjacent workflows discovered in the original repository. It
keeps TrainVM broad enough for RWKV, transformer/MLA, vision, MageFlow/diffusion, conversion,
post-training, RLVR, external trainers, data/cache, evaluation, profiling, and export work.

The catalog is evidence only. Its root `authority` is fixed to
`compatibility_evidence_only`; the C++ types contain no adapter key, executable identity, worker
capability, credential, or execution method. Only the sealed adapter and host-execution registries
can authorize work.

The reviewed v1 inventory contains 156 effect-specific records over 145 unique source files.
Its closed families are `rwkv`, `transformer`, `vision_multimodal`, `mageflow_diffusion`,
`conversion_distillation`, `rwkv_posttraining`, `rwkv_rlvr`, `external_trainer`, `data_cache`,
`evaluation_profile_export`, and `control_plane`. The broad `transformer` name is intentional:
the baseline includes MLA/Engram experiments and Qwen MoE continued pretraining, and future
transformer adapters must not be mislabeled as MLA merely to fit the inventory.

Each entry contains a stable ID, closed family, reviewed source paths, `observed_invocation`,
statefulness, catalog-local `resume_evidence`, an operation role, and notes. The optional
`legacy_invocation_display` is untrusted display text: it is never parsed or executed. The closed
observed values are `python_module`, `console_script`, `host_script`,
`http_control_handler`, `retired_legacy`, `library_only`, and `design_only`. Go dashboard request
handlers use `http_control_handler`; removed dashboard mutations use `retired_legacy` so their
parity obligations remain visible without claiming that a route or process capability still exists.

This is the reviewed supported workflow surface, not an inventory of every file containing
`__main__`. A source is included when it is an installed console command or distinct subcommand, a
checked-in production module/script entrypoint or launch profile, a legacy HTTP/control handler, or
an essential graph component/library oracle used by an included workflow. A synthetic smoke-only
`__main__` block never qualifies as `python_module`. Essential smoke-bearing components such as
Engram LMB, MLA/layer-swap/SVD initialization, and ROSA variants are explicitly `library_only` and
remain nonlaunchable. Tests pin that classification so a self-test cannot silently become launch
evidence.

Catalog resume evidence is deliberately a separate type from adapter resume authority. Its values
are `none`, `restart_only`, `terminal_checkpoint`, `compatible`, and `exact_candidate`.
`exact_candidate` means that a legacy implementation appeared to preserve exact state during this
review; it is not, and cannot compare equal to, an AdapterRegistry `exact` grade. Library-only and
design-only records are stateless and use `none`.

The v1 inventory is an exact compiled mapping, not merely a set of reviewed IDs. Removing or adding
an entry fails, and changing any field of an existing entry fails until the canonical catalog digest
is deliberately updated in compiled code. The catalog also carries `source_tree_digest`, a
deterministic SHA-256 tree digest over every unique referenced path and its exact bytes. Validation
hashes each source through the same descriptor used to inspect it.

The mutable export records are intentionally candid. `export.legacy-mutable-bundle` recursively
overwrites its destination and performs only shallow listed-file/tensor-count verification.
`export.frozen-vision-compressor` is also overwriteable and has no self-bound artifact hash. Neither
record implies immutable artifact identity.

Effectfully different commands are separate records even when they share a Python module. Qwen
audit/plan/train, AO3 tokenize/pack/in-place rewrite, Engram frequency/allocation, MageFlow
plan/cache/train/adaptation preparation/audit, LTX plan/prepare/train/run, production qualification,
production-qualified AOT publication, post-training qualification, and standalone megakernel AOT
export therefore have distinct stable IDs. The importable lossless GDN mapper is likewise separate
from its executable PPL self-test. Vision launch profiles are split by invoked trainer and handoff
or resume contract: RADIO1D, V4H, MoonViT continuation, native head, raw-pixel student, and teacher
compressor are independent evidence records. Every source byte is bound and no profile carries
adapter or host authority. Legacy dashboard launch and queue handlers are recorded under
`control_plane` so migration cannot silently forget an old process-control path.

The dashboard mutation surface is closed rather than sampled. Every route the legacy Go router
registers with a `POST` method has exactly one `http_control_handler` record. Removed mutations
remain as `retired_legacy` records, and both kinds bind their handler file plus
`dashboard/internal/server/server.go`, the registration table itself. Two independent gates keep
that closed:

1. adding, removing, or renaming any route changes the router bytes and therefore fails the
   `source_tree_digest` check before anything else is examined;
2. `compatibility_catalog_tests` re-reads the registration table and fails when a served legacy
   route has no active record, an active record names an unserved route, or a retired record becomes
   served again. TrainVM's own `/api/trainvm/` namespace is excluded because it is declarative
   authority, not legacy evidence.

The retired records are candid about what the legacy control plane could not do. Stop and
checkpoint were bare signal deliveries with no typed acknowledgement or receipt. Live control
patches were whitelisted numeric rows in shared SQLite that the trainer polled, with no application
point or revision fence. `POST /api/experiments/run` started a detached trainer with no script
allowlist, no process group, and no audit row. The auto-stop and queue-automation toggles armed
background goroutines that signaled or spawned processes while holding their armed state only in
dashboard memory, so neither survived a restart nor left an audit trail. Preference capture and
dataset versioning grew training inputs with no published artifact manifest. These records are
migration evidence, not callable compatibility shims.

Validate the checked-in inventory with:

```bash
trainvm/build/trainvm validate-catalog \
  "$PWD/docs/experiment-vm/compatibility-workflows.v1.json" "$PWD"
```

The catalog itself is opened with `O_NOFOLLOW`, bounded, verified as a regular file, and checked for
stable identity while reading. Sources are opened descriptor-relatively beneath the displayed
repository-root identity with Linux `openat2`, `RESOLVE_BENEATH`, `RESOLVE_NO_MAGICLINKS`, and
`RESOLVE_NO_SYMLINKS`. Unknown or duplicate JSON fields, unknown enums, within-entry source duplication,
nonregular files, source-byte drift, ID-set drift, and contradictory role/state claims fail closed.
The compatibility library is linked only into the CLI validator and its tests; the long-running
service does not contain it.

## The two digests

The catalog pins two, and they answer different questions.

`source_tree_digest` binds the **whole bytes** of every referenced source. It is the provenance
record: it says the entries were reviewed against exactly these files. It still fails closed, so the
record cannot go stale.

`classification_surface_digest` binds only the part of each source that could change how the entry
beside it is classified — its entrypoint declaration, its argument surface, and its checkpoint and
resume call sites. For Python this is extracted lexically; every other language falls back to its
full bytes, because a narrower surface has to be earned per language rather than assumed.

Only the second is covered by the compiled `kReviewedCatalogDigest`. That split is the point: adding
a comment or changing an internal computation re-pins the byte digest and nothing else, while
renaming an entrypoint or adding an argument moves the reviewed digest and forces a re-review.

Before this split, every byte demanded both. Three separate cards resolved as "re-reviewed, no
classification change, hashes updated" — which is what a gate looks like shortly before people stop
reading it, and is a slower failure than having no gate, because it still reads as protection.

The extraction is deliberately over-broad: `checkpoint` and `resume` match more lines than strictly
bear on `resume_evidence`. That is the safe direction. A surface that is too wide costs an occasional
unnecessary review; one that is too narrow lets a real classification change through unnoticed.

A vacuous extraction would be the dangerous failure — a file whose surface silently came back empty
would be permanently invisible to review. Any entry recorded as `python_module` or `console_script`
must therefore expose a nonempty surface, or validation refuses the catalog and names the entry.

## Re-pinning

Nothing used to compute these values, which is much of why re-pinning felt like a chore rather than
a review. Now:

```bash
# Build first: trainvm/build/ is gitignored, so an existing binary there is as
# old as your last build and a stale one fails misleadingly (usage dump if it
# predates this subcommand, schema rejection if it merely predates the catalog).
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j "$(nproc)" --target trainvm
trainvm/build/trainvm print-catalog-digests \
  "$PWD/docs/experiment-vm/compatibility-workflows.v1.json" "$PWD"
```

It computes both digests without checking them, so it still reports when they disagree, and lists
any Python source whose classification surface came out empty. Paste `source_tree_digest` back into
the JSON after an unrelated source edit. If `classification_surface_digest` also moved, that is the
signal to actually re-read the entry before pinning it and bumping `kReviewedCatalogDigest`.
