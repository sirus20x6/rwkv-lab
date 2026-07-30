# Mage-Flow terminal expert architecture

This document supersedes the residual-expert topology described in the earlier
Mage-Flow adaptation notes. The residual implementation is preserved at Git
branch `beta/mageflow-residual-experts-20260730`.

## Required computation graph

Every generation executes the complete released Mage-Flow transformer before
executing exactly one terminal specialist:

```text
image/text/timestep embeddings
          |
released Mage-Flow blocks 0-11
          |
CPU prompt router / explicit dataset route
          |
          +-- photo terminal expert: 3 complete Mage-Flow blocks
          |
          +-- animation/illustration terminal expert: 3 complete Mage-Flow blocks
          |
released adaptive output normalization and projection
```

The two expert branches are mutually exclusive. There is no neutral route,
expert interpolation, residual addition to a shared FFN, or replacement of an
original backbone block.

## Capacity and residency

The pinned released transformer contains 4,115,745,408 parameters. Every
released double-stream block contains 339,831,296 parameters. A three-block
terminal expert therefore contains 1,019,493,888 parameters.

| Quantity | Parameters |
|---|---:|
| Released shared transformer | 4,115,745,408 |
| One terminal expert | 1,019,493,888 |
| Active routed model | 5,135,239,296 |
| Both expert files plus shared transformer | 6,154,733,184 |

Both expert files exist on disk, but only the selected expert is resident in
accelerator memory. At BF16, raw active transformer weights occupy about
10.27 GB before the VAE, text encoder, activations, gradients, optimizer state,
and runtime workspaces.

For inference, prompt routing happens before diffusion begins. Switching the
route replaces the one resident terminal module from the corresponding expert
file. Training manifests already contain explicit domains and never use prompt
heuristics.

## Initialization from the residual beta

The residual beta and terminal architecture are not functionally or
shape-equivalent, so a direct checkpoint conversion is impossible. The
migration uses the following bounded transfer:

1. Construct each terminal expert from released Mage-Flow blocks 9-11. This
   gives the new attention, text MLP, modulation, normalization, and image MLP
   tensors meaningful pretrained values rather than identity or random values.
2. For each domain, read trained residual image MLPs from residual blocks 9-11.
3. Score each residual hidden neuron by the product of its incoming and
   outgoing weight norms.
4. Retain the 12,288 strongest neurons from the 12,544-neuron residual MLP
   (97.959%).
5. Install them into the corresponding terminal image MLP and fold the learned
   residual scalar into its output projection.

This transfers late domain-specific image-FFN behavior. It does not claim to
preserve the old function: terminal attention and text processing are new
executed operations, and the old expert operated additively inside eight
backbone blocks.

The migration command is:

```bash
PYTHONPATH="$PWD/src" .venv-mage-flow/bin/python \
  scripts/migrate_mage_flow_terminal_experts.py \
  --residual-checkpoint-dir \
    runs/mage_flow_joint_expert_shared_tail_tranche_b_v1/checkpoint-00032060 \
  --output-dir runs/mage_flow_terminal_experts_init_v1 \
  --cache-dir .hf_cache \
  --dtype bf16
```

## Training contract

Training runs are domain-resident: a photo window loads the photo tail, and an
animation window loads the animation tail. Domain changes occur only at a
checkpoint/window boundary, never between microbatches. The inactive expert
and its optimizer state remain in host memory or on disk.

When both experts are trainable:

- selected terminal expert: `1.0x` learning rate;
- approved original Mage-Flow backbone tail: `0.5x` expert learning rate;
- all other original backbone parameters: frozen;
- VAE and Qwen text encoder: frozen.

Backbone checkpoints and expert checkpoints are independent. Alternating
domain windows must carry the newest shared-backbone checkpoint forward so
both experts continue against the same shared model.

## Required preflight checks

Before training or inference:

- there are exactly 12 original backbone blocks;
- all 12 execute before the expert;
- the resident terminal expert contains exactly 3 full blocks;
- no original image FFN is wrapped with residual experts;
- exactly one domain is resident;
- the resident expert contains 1,019,493,888 parameters;
- active transformer parameters total 5,135,239,296;
- prompt or manifest routing returns only `photo` or `animation`.
