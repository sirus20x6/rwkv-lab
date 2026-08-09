# Source disposition catalogs

The versioned source-disposition catalogs are the machine-readable migration
inventory for legacy training code:

- `source-dispositions.scripts.v1.json` covers every top-level Python and shell
  file in `scripts/` exactly once.
- `source-dispositions.rwkv-lab.v1.json` covers every top-level Python module in
  `src/rwkv_lab/` exactly once.

These catalogs are evidence only. Their fixed `authority` value is
`compatibility_evidence_only`; a catalog row cannot grant an adapter key, worker
capability, executable identity, or host execution permission.

## Common v1 contract

Every sorted entry records its repository-relative source path, closed class,
canonical entry point, closed effect set, resume relevance, nullable canonical
compatibility-workflow link, and exact lowercase SHA-256 of the source bytes.
Optional fields retain domain-specific family, coverage, consumer, language, and
additional workflow-link evidence. Unknown fields and enum values fail closed.

Three source roots are reviewed: `src/rwkv_lab` and `scripts` non-recursively,
and `dashboard` recursively. The dashboard scope is recursive because its Python
lives in several subdirectories, so a non-recursive prefix would let a new tool
escape review simply by being placed one level down. It classifies the stale
`dashboard/instrumented/train_mla.py` fork as an explicit exclusion rather than
leaving a 2,000-line divergent trainer unclassified.

The root `source_scope` declares a repository-relative prefix, whether traversal
is recursive, and the included filename extensions. The tree digest is SHA-256
over the ASCII domain `trainvm.source-disposition-tree/v1`, followed in entry
order by NUL, `source_path`, NUL, and the full `sha256:...` leaf digest for each
entry. It binds ordering, membership, paths, and source content without
depending on a Git object format.

**No catalog stores it.** It is a pure function of `entries`, computed at load
time by `trainvm::source_tree_digest()` and exposed as
`document().source_tree_digest`. It used to be a stored field, and because it
changes whenever any file in scope changes, it was the single line every
concurrent change to the scope had to rewrite — four pull requests conflicted on
it in one evening while touching entirely disjoint work, each paying a rebase
and a ~10-minute CI cycle. The `entries` list around it merges cleanly, because
two changes touch two different entry objects. A document that still declares a
`source_tree_digest` is **rejected**, correct value or not, so a stale copy
cannot sit in a catalog looking authoritative.

Removing the stored value removed something load-bearing, and it was replaced
rather than dropped. `scripts/print_disposition_digests.py` carries a second,
hand-written implementation of the same fold, and the stored digest was what
kept the two in step — transitively, and only at the moment somebody
regenerated. `trainvm/tests/source_disposition_catalog_tests.cpp` now drives
both implementations over the same entries (`--tree-digest` on the Python side)
and requires the answers to be equal, with no stored value between them. That
is strictly stronger: it compares them on every native run, over the real
catalogs and over synthetic vectors including the `("ab","c")` / `("a","bc")`
pair that only the NUL framing separates.

`reviewed_classification_digest()` is SHA-256 over the document with every
per-entry `source_sha256` erased. It moves when a review decision moves —
paths, classes, entry points, effects, resume grades, workflow links — and
stands still when a classified source is merely edited. It replaced a digest
over the whole document, which moved on every source edit and so made the
constants pinning it in the native tests a second per-scope serialization point
in a second file. Source bytes remain pinned per entry and re-checked against
disk, so nothing is unpinned by the split.

The C++ loader rejects duplicate JSON keys, duplicate or out-of-scope paths,
malformed digests, unknown classes/effects/resume grades, a stored
`source_tree_digest`, and unresolved workflow links when the reviewed workflow
ID set is provided. With a repository root it also enumerates the declared scope
exactly, rejecting missing or stale files and source-byte drift.

Tests are hermetic by default. To additionally validate both checked-in catalogs
against a local original repository:

```bash
cmake -S trainvm -B trainvm/build \
  -DTRAINVM_LEGACY_SOURCE_ROOT=/path/to/moe-mla
cmake --build trainvm/build --target source_disposition_catalog_tests
ctest --test-dir trainvm/build -R source_disposition_catalog_tests --output-on-failure
```

The recorded `git-sha1:` revision is provenance only. Per-file SHA-256 and the
domain-separated tree digest are the content authority, including for a dirty
working tree.
