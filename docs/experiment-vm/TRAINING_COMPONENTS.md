# Training-component boundaries

TrainVM separates algorithm selection, tensor construction, trainer integration, and presentation.
No layer is allowed to recreate the responsibilities of another layer.

## Control-plane authority (C++)

`TrainingComponentRegistry` owns exact `(category, name, version)` keys, reflected scalar fields,
model-family compatibility, backend and symbolic implementation identity, required worker
capabilities, schedule step domains, checkpoint-state schemas, and state grades. It resolves every
node composition, applies defaults, rejects unknown fields, and freezes the result into the
submission lock and immutable worker invocation.

The experiment can select an authority entry, but it cannot provide Python imports, argv,
environment variables, source paths, or implementation code. Selected component capabilities are
unioned with the adapter's requirements before a launch intent exists. Exact-resume plans reject a
compatibility-grade component.

## Tensor-runtime implementation (Python/CUDA)

Tensor code stays in the runtime that implements it well. `rwkv_lab.training_components` is a
stable compatibility facade; implementations are physically separated under
`rwkv_lab.training_runtime` into `optimizers.py`, `schedules.py`, and `routers.py`, with only the
shared resolved-envelope decoder at the package root. Each category owns its closed enum, typed
immutable configuration, validation, construction, and resolved dispatch. Category modules do not
import one another or any family trainer. The facade has no implementation logic and trainers do
not depend on the category internals.

Future categories follow the same rule: add `activations.py`, `normalizations.py`, `objectives.py`,
`precision.py`, `gradient_policy.py`, `weight_decay_schedules.py`, or `curricula.py` only when a
real adapter consumes the descriptor. Do not create a placeholder module or advertise decorative
configuration. The resolved-worker dispatch functions accept only the canonical envelope from
TrainVM and fail on extra keys, unknown implementation IDs, wrong categories, missing defaults,
nonfinite numbers, or incorrect scalar types.

Every category keeps its configuration, validation, construction, checkpoint codec, and tests
together. Shared files contain protocols and resolved-envelope plumbing only; they are not a home
for optimizer math, learning-rate policy, activation kernels, or family exceptions. CI checks the
one-way dependency graph and rejects direct component construction in supported trainer loops. This
keeps new choices local: adding an optimizer, schedule, activation, or precision policy changes its
category, descriptor, and contract fixtures without requiring unrelated trainer or dashboard code.

The initial concrete cross-family catalog contains:

- Torch AdamW;
- FP32-master AdamW for lower-precision live model weights;
- linear warmup followed by a cosine tail over optimizer steps;
- warmup/plateau/PowerCool over optimizer steps for RWKV and transformers;
- appearance-expert versus shared-backbone exclusive parameter routing;
- terminal-expert versus shared-backbone versus VAE-REPA exclusive parameter routing.

The three MageFlow training paths, RWKV AdamW path, and Qwen transformer AdamW/PowerCool path use
the common tensor boundary. The MageFlow appearance/terminal expert trainers and Qwen AO3
continuation additionally consume authority-resolved worker compositions; their composition digest
is resume identity, and Qwen persists the registered scheduler cursor alongside optimizer state.
Scratch RWKV consumes the same optimizer slot and the typed PowerCool configuration directly; its
optimizer-step cursor remains the schedule state because tied-head and sparse-routing transitions
can change parameter-group topology during the run.
Routing aggregates aliased names,
deduplicates tensor identities, rejects overlap and unclaimed trainable tensors, and records an
exact group/count/rate audit in the run contract. The catalog does not claim another
activation, objective, optimizer, router, or kernel until a real adapter path consumes the symbolic
implementation and has CPU parity plus representative accelerator qualification.

## Model-family integration

Adapters own topology-specific work only: parameter discovery and exhaustive routing, where an
activation or normalization is installed, loss inputs, and checkpoint tensor placement. They do
not own generic optimizer hyperparameter validation or schedule math. A parameter router must
produce an exhaustive ownership report; overlapping or unclaimed trainable parameters fail before
the optimizer is constructed.

Python worker adapters use `rwkv_lab.trainvm_adapters.WorkerTrainingComponents` as the one
composition bridge. It binds the already-verified worker composition to an expected model family,
requires a category for every named slot, delegates construction to the category runtime, and emits
descriptor/implementation evidence. It contains no model topology or training loop. MageFlow,
RWKV, transformer, vision, and post-training adapters may share this bridge but may not add family
conditionals to it.

## Dashboard

`GetDescriptor(trainvm.training-components@1.0.0)` returns the canonical registry bytes and digest.
The Go bridge recomputes the hash. The browser composer renders fields, defaults, ranges, and enums
from those descriptors, and writes only a declarative node/family/slot selection. Adding a component
does not add a Go endpoint or component-specific JavaScript.

## Adding a component

1. Add the exact authority descriptor and state schema.
2. Add a closed implementation enum and typed configuration/factory in the category-owned tensor
   runtime module, preserving the one-way dependency into family adapters.
3. Add cross-runtime tests proving descriptor fields/defaults match the factory.
4. Add reference output, gradient/update, state round-trip, and resumed-trajectory parity evidence.
5. Integrate it into an adapter and advertise the exact worker capability.
6. Qualify representative throughput, memory, and model quality before enabling it in a production
   registry.

Passing schema tests makes a component selectable; it does not prove mathematical interchangeability
or production fitness.
