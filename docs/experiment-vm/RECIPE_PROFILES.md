# Versioned declarative training recipes

A recipe profile is an authority-owned, exact-versioned `Experiment` template
with a deliberately small scalar override surface. It makes an ordinary run a
data document instead of a new trainer, handler, argv builder, or deployment.
The expanded result is still the same canonical `CompiledPlan` consumed by the
controller, preflight, component registry, worker launch, and dashboard.

This layer is for composition, not implementation. Registering
`hf_multimodal_sft@1` does not dynamically import a trainer. Its template names
an exact registered adapter, operation, and training-component graph. Runtime
registration and qualification remain separate authority decisions.

## Documents

The authority registry has schema `trainvm.recipe-profiles/v1`:

```json
{
  "api_version": "trainvm.recipe-profiles/v1",
  "recipes": [{
    "key": {"name": "hf_multimodal_sft", "version": "1"},
    "template_document": {"api_version": "trainvm.rwkv-lab/v1alpha1"},
    "overrides": [{
      "name": "hyperparameters.learning_rate",
      "domain": "hyperparameters",
      "type": "number",
      "target": "/spec/workflow/nodes/train/invoke/training/components/optimizer/configuration/learning_rate",
      "required": false,
      "minimum": 1e-8,
      "maximum": 0.01
    }]
  }]
}
```

The omitted template body above is only for exposition. A registry entry must
contain one complete, compiler-valid `Experiment`. The checked-in example is
[hf-multimodal-sft.recipe-profiles.v1.json](examples/hf-multimodal-sft.recipe-profiles.v1.json).

A run instance has schema `trainvm.recipe-instance/v1`:

```json
{
  "api_version": "trainvm.recipe-instance/v1",
  "recipe": {"name": "hf_multimodal_sft", "version": "1"},
  "run_identity": "qwen-caption-lora-r256",
  "overrides": {
    "model.path": "/models/Qwen3.6-35B-A3B",
    "data.manifest": "/datasets/captions/train.jsonl",
    "trainability.lora_rank": 256,
    "hyperparameters.learning_rate": 0.00002
  }
}
```

The complete compact Qwen caption instance is
[qwen-caption-lora-r256.recipe-instance.v1.json](examples/qwen-caption-lora-r256.recipe-instance.v1.json).
Changing its model, dataset, rank, learning rate, evaluation cadence, or
checkpoint cadence does not change source code.

The example's graph selects the registered `hf_multimodal` model loader and
`lora` trainability policy as a required pair. Model path and fingerprint are
owned by the loader configuration; rank and alpha are owned by the LoRA policy.
There are no duplicate operation arguments that can drift away from those
component-owned values. The remaining objective, optimizer, and BF16 precision
slots also use exact keys present in the checked-in training-component
registry; the native test resolves the complete graph against that registry.
The all-zero fingerprint in the documentation fixture is visibly non-production
test data; a launch preflight/load receipt must match the real checkpoint bytes
and will reject it rather than silently load a different model.

`run_identity` is one bounded symbolic value, not a workspace override. During
expansion it atomically derives the experiment metadata name, a child run
directory beside the template's run directory, the exact corresponding write
root, a namespaced concurrency key, and both resource acquire/release literals.
All derived leaves receive `instance_run_identity` provenance. This prevents
two variants from sharing output directories and prevents the coupled resource
fields from drifting. The caller cannot provide a raw output path, widen write
authority, or update only one member of the coupled set.

## Closed override surface

Each override is a reflected scalar contract with one domain, type, exact JSON
pointer, required flag, optional numeric bounds, and (for an enumeration) a
finite allow-list. Domains cover model identity, data identity and mapping,
trainability, hyperparameters, evaluation, checkpointing, resources, and live
control defaults.

Targets are fail-closed. They may address only:

- a declared parameter's `value`;
- a declared training component's scalar `configuration` leaf;
- a declared control's `default`;
- a small explicit list of resource sizing leaves.

They cannot address API/kind, metadata, workspace roots or permissions,
adapter/runtime/operation/contract identity, workflow topology, bindings,
artifact contracts, recovery policy, selectors, executable material, shell,
Python/import names, Jinja/templates, arbitrary expressions, argv, or
environment maps. Targets must already exist in the canonical template and
must be scalar. Two fields cannot own the same target or overlapping targets.
Path values must also remain within the template's authority-owned
`allowed_read_roots`; an instance cannot widen those roots.

Unknown instance fields, missing required fields, type errors, out-of-bound
values, unused/nonexistent targets, and ambiguous ownership all reject before
launch. Component choices that need cross-field qualification use a finite
tuple allow-list. This expresses such rules as “bf16 with 80 GiB or fp16 with
96 GiB” without adding a predicate or expression language. The expanded
ordinary experiment is compiled again, so all existing compiler compatibility
and authority checks still apply.

## Identity and evidence

Expansion records four content identities:

- `registry_digest`: canonical set of all exact recipe versions;
- `profile_digest`: canonical template, override contracts, and compatibility;
- `instance_digest`: exact recipe key, run identity, and canonical supplied
  overrides;
- `expanded_plan_digest`: the ordinary canonical plan hash, namespaced as a
  SHA-256 digest.

The expanded plan is not a parallel execution format. Its `plan_hash` is
exactly the hash produced by compiling the same fully expanded Experiment
directly. Existing expanded documents remain accepted, and equivalent recipe
and full-document authoring produces the same plan identity.

Every scalar leaf in the canonical plan carries provenance. An unchanged leaf
points to `recipe_template`; an overridden leaf points to the exact named
`instance_override`. Recipe diffs compare canonical leaves and include the
source on both sides, so a reviewer sees both what changed and why it has that
value.

## CLI

```bash
trainvm recipe inspect <recipe-profiles.json>
trainvm recipe expand <recipe-profiles.json> <instance.json>
trainvm recipe diff <recipe-profiles.json> <left-instance.json> <right-instance.json>
```

`inspect` validates and prints the canonical registry and registry digest.
`expand` prints the canonical ordinary plan, all identities, effective values,
and leaf provenance. `diff` prints changed canonical paths with values and
sources on each side. The expand output can be handed to passive preflight by
using its `canonical_plan`; preflight does not need to trust or interpret recipe
internals.

Profile versions are immutable in use. Changing a template, override bound,
allow-list, or compatibility tuple changes its digest; a semantic change should
also receive a new recipe version so authors opt into it explicitly.
