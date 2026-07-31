# Compatibility workflow catalog

[`compatibility-workflows.v1.json`](compatibility-workflows.v1.json) is the machine-readable,
reviewed inventory of training and adjacent workflows discovered in the original repository. It
keeps TrainVM broad enough for RWKV, transformer/MLA, vision, MageFlow/diffusion, conversion,
post-training, RLVR, external trainers, data/cache, evaluation, profiling, and export work.

The catalog is evidence only. Its root `authority` is fixed to
`compatibility_evidence_only`; the C++ types contain no adapter key, executable identity, worker
capability, credential, or execution method. Only the sealed adapter and host-execution registries
can authorize work.

Each entry contains a stable ID, closed family, reviewed source paths, `observed_invocation`,
statefulness, catalog-local `resume_evidence`, an operation role, and notes. The optional
`legacy_invocation_display` is untrusted display text: it is never parsed or executed. The closed
observed values are `python_module`, `console_script`, `host_script`, `library_only`, and
`design_only`.

Catalog resume evidence is deliberately a separate type from adapter resume authority. Its values
are `none`, `restart_only`, `terminal_checkpoint`, `compatible`, and `exact_candidate`.
`exact_candidate` means that a legacy implementation appeared to preserve exact state during this
review; it is not, and cannot compare equal to, an AdapterRegistry `exact` grade. Library-only and
design-only records are stateless and use `none`.

The v1 inventory is an exact compiled set of reviewed IDs. Removing an entry still fails even when
its family remains represented, and adding an entry requires a deliberate code review and compiled
set change. The catalog also carries `source_tree_digest`, a deterministic SHA-256 tree digest over
every unique referenced path and its exact bytes. Validation hashes each source through the same
descriptor used to inspect it.

The mutable export records are intentionally candid. `export.legacy-mutable-bundle` recursively
overwrites its destination and performs only shallow listed-file/tensor-count verification.
`export.frozen-vision-compressor` is also overwriteable and has no self-bound artifact hash. Neither
record implies immutable artifact identity.

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
