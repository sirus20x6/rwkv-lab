"""
Wire Engram modules into the Qwen3.6-35B-A3B decoder stack.

Places an `EngramModule` from the attested ``engram-ext`` runtime distribution
at specified decoder layers, threads the batch's input_ids down to them, and has
each one residual-add its contribution at the TOP of the layer forward — matching
the paper's placement: Engram → Attention → MoE within each selected block.

Public surface:
    install_engram(model, layer_indices, engram_cfg, hidden_size)
        -> list[EngramModule] (trainable)
    uninstall_engram(model)        # undo patches (not usually needed)

Implementation note: we stash input_ids on the base model (model.model) at the
start of each top-level forward, and each Engram-wrapped decoder layer reads it
from there. This works across gradient checkpointing because the attribute is
set on the outer forward and layers are called (and re-called during backward)
from inside that scope.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - types only, never imported at runtime
    from engram_ext.engram_module import EngramConfig, EngramModule

# ---------------------------------------------------------------------------
# The engram-ext runtime distribution is imported at the point of use, not here.
#
# It is a separately attested distribution built outside this repository, and
# nothing in this repository or in CI installs it. Importing it at module scope
# meant `import rwkv_lab.load_mla_engram` and `import rwkv_lab.train_mla_engram`
# raised on every machine that is not the training host, so neither module could
# be tested anywhere — and nothing reported that. train_mla.py already defers
# the same import to its `if cfg.engram_enabled:` branch; this follows it.
#
# Deferring moves the failure from "importing an unrelated module" to "asking
# for Engram without the runtime", which is the only place the absence matters.
# ---------------------------------------------------------------------------

ENGRAM_EXT_DISTRIBUTION = "engram_ext"

ENGRAM_EXT_MISSING_MESSAGE = (
    "engram_ext is not importable, so Engram cannot be installed. It is the "
    "separately attested engram-ext runtime distribution, built outside this "
    "repository and deliberately not a dependency of rwkv-lab: only the Engram "
    "entry points need it. To make it importable, build the extension from the "
    "engram checkout and install its Python distribution into this "
    "environment, e.g.\n"
    "    pip install /path/to/engram/python\n"
    "Do not prepend a machine-local checkout to sys.path instead -- that "
    "bypasses the dependency attestation the sealed runtime relies on. See the "
    "README section \"The engram-ext runtime dependency\"."
)


class EngramExtensionUnavailable(ModuleNotFoundError):
    """Raised when Engram is requested but engram_ext cannot be imported.

    A ModuleNotFoundError subclass so callers with an existing
    ``except ImportError`` still catch it, while the message names the
    distribution and how to get it rather than pointing at an import line in a
    module the caller never asked about.
    """


def require_engram_ext():
    """Import and return ``engram_ext.engram_module``, or explain its absence.

    Every use of an engram-ext symbol in this package goes through here, so a
    missing runtime produces one message naming ``engram_ext`` instead of an
    ImportError traceback whose top frame is an unrelated import.
    """
    try:
        from engram_ext import engram_module
    except ImportError as error:
        raise EngramExtensionUnavailable(
            ENGRAM_EXT_MISSING_MESSAGE, name=ENGRAM_EXT_DISTRIBUTION
        ) from error
    return engram_module


# Once-per-run flag: warn loudly (once) when a forward carries no token ids
# and Engram is therefore silently inactive for that forward.
_ENGRAM_NO_IDS_WARNED = False


# ---------------------------------------------------------------------------
# Host-memory embedding (paper's "deterministic addressing enables offload")
#
# Weight lives in pinned CPU RAM; indices are moved CPU-side per lookup, the
# gather happens on CPU, and the resulting dense rows are transferred back to
# the GPU. At training time gradients use the embedding op's sparse path so
# only the touched rows need scatter-add on CPU.
#
# Bandwidth cost per layer at (batch=1, seq=2048, 16 hashes, 64 dim, bf16):
#   * indices: 32K * 8 B = 256 KB (G->C)
#   * gathered output: 2048 * 16 * 64 * 2 B = 4 MB (C->G)
#   Total ~4.3 MB per layer per forward over PCIe 4.0 ~ 130 us. Negligible
#   vs the 1-2 sec fwd through a 35B MoE.
# ---------------------------------------------------------------------------

class CpuHostEmbedding(nn.Module):
    """nn.Embedding variant whose weight tensor is kept on pinned host RAM.

    Designed as a drop-in replacement for the nn.Embedding inside
    engram_ext.MultiHeadEmbedding. Uses `sparse=True` so only the rows touched
    during a forward participate in gradient accumulation; pair with
    torch.optim.SparseAdam on the CPU side.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 dtype: torch.dtype = torch.float32, init_std: float = 0.02) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Allocate on CPU, initialize, then pin for fast DMA.
        w = torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        nn.init.normal_(w, std=init_std)
        self.weight = nn.Parameter(w.pin_memory(), requires_grad=True)
        # Hint for optimizer routing
        self.weight._is_host_embedding = True  # type: ignore[attr-defined]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        orig_device = indices.device
        indices_cpu = indices.to("cpu", non_blocking=True)
        N = self.weight.shape[0]
        # OOR-safe path: instead of .item() syncs + clamp-to-N-1 (which would
        # corrupt row N-1's gradient by routing every bad index to the same row),
        # mask OOR positions to row 0 and zero out their gathered contribution.
        # Valid indices behave normally; invalid contribute zero to output AND
        # zero gradient to row 0 (chain rule: mask=0 → upstream grad=0).
        valid = (indices_cpu >= 0) & (indices_cpu < N)
        safe = torch.where(valid, indices_cpu, torch.zeros_like(indices_cpu))
        gathered = torch.nn.functional.embedding(safe, self.weight, sparse=True)
        gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        return gathered.to(orig_device, non_blocking=True)

    def load_rows(self, new_weight: torch.Tensor) -> None:
        """Replace the weight with a provided tensor (must match shape). Re-pins."""
        assert new_weight.shape == self.weight.shape
        self.weight = nn.Parameter(
            new_weight.to(dtype=self.weight.dtype).cpu().pin_memory(),
            requires_grad=True,
        )
        self.weight._is_host_embedding = True  # type: ignore[attr-defined]


def offload_engram_embedding(em: EngramModule) -> None:
    """In-place: relocate em.embedding.embedding (the big table) to pinned CPU
    memory using CpuHostEmbedding. Leaves all other Engram parameters on their
    current device. Must be called BEFORE moving `em` to GPU if you want tables
    on CPU (or AFTER — we'll override the embedding either way)."""
    orig = em.embedding.embedding  # nn.Embedding
    N, D = orig.num_embeddings, orig.embedding_dim
    dtype = orig.weight.dtype
    new_emb = CpuHostEmbedding(N, D, dtype=dtype)
    # If the original was already initialized, keep its values for determinism.
    with torch.no_grad():
        new_emb.weight.data.copy_(orig.weight.data.detach().cpu())
    em.embedding.embedding = new_emb  # type: ignore[assignment]


def collect_host_embedding_params(engram_modules: list[EngramModule]) -> list[nn.Parameter]:
    """Return the list of CPU-resident embedding weights across the given Engram
    modules. Use this to build a separate SparseAdam optimizer."""
    out: list[nn.Parameter] = []
    for m in engram_modules:
        inner = m.embedding.embedding
        if isinstance(inner, CpuHostEmbedding):
            out.append(inner.weight)
    return out


def collect_gpu_engram_params(engram_modules: list[EngramModule]) -> list[nn.Parameter]:
    """Return Engram parameters that live on GPU (everything except the big
    embedding tables). These go into the regular 8-bit AdamW param group."""
    host_ids = {id(p) for p in collect_host_embedding_params(engram_modules)}
    return [
        p for m in engram_modules
        for p in m.parameters()
        if id(p) not in host_ids and p.requires_grad
    ]


def _resolve_layers(model: nn.Module) -> tuple[nn.Module, nn.ModuleList]:
    """Return (backbone_module_with_forward, layers_module_list).

    Handles two loading variants of Qwen3.5/3.6 MoE:
        ForCausalLM -> model.model.layers  (text-only load)
        ForConditionalGeneration -> model.model.language_model.layers
    The first returned object is the one whose .forward we patch to stash
    input_ids so decoder layers can read them.
    """
    inner = model.model
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return inner, layers
    inner = inner.language_model
    return inner, inner.layers


def install_engram(
    model: nn.Module,
    layer_indices: list[int],
    engram_cfg: EngramConfig,
    hidden_size: int,
    host_offload_embedding: bool = False,
) -> list[EngramModule]:
    """Attach Engram modules at the given decoder-layer indices, patch the model
    so input_ids flows to them, and return the list of new modules.

    If host_offload_embedding=True, the big per-layer embedding table is kept
    on pinned host RAM (as a CpuHostEmbedding) and ONLY the small projections /
    norms / conv are moved to the decoder layer's GPU. This avoids any transient
    allocation of the billion-row table on GPU.
    """
    if getattr(model, "_engram_installed", False):
        raise RuntimeError("Engram already installed; call uninstall_engram first.")

    # The point of use for the deferred engram-ext import: fail here, naming the
    # distribution, rather than at `import rwkv_lab.load_mla_engram`.
    engram_module = require_engram_ext()

    # ---- Stash input_ids at the backbone module whose forward iterates layers ----
    base, layers = _resolve_layers(model)
    orig_base_forward: Callable[..., Any] = base.forward

    def base_forward_with_stash(*args, **kwargs):
        # HF Qwen3.5 model forward uses `input_ids` as a named arg, but also
        # accept a positional first argument that looks like token ids.
        ids = kwargs.get("input_ids", None)
        if ids is None and args:
            cand = args[0]
            if (torch.is_tensor(cand) and cand.dim() == 2
                    and not cand.is_floating_point() and not cand.is_complex()):
                ids = cand
        if ids is None:
            global _ENGRAM_NO_IDS_WARNED
            if not _ENGRAM_NO_IDS_WARNED:
                _ENGRAM_NO_IDS_WARNED = True
                warnings.warn(
                    "Engram: forward received no token ids (inputs_embeds-only "
                    "call?) — Engram modules are INACTIVE for such forwards. "
                    "This warning is emitted once per run.",
                    RuntimeWarning, stacklevel=2)
        # We intentionally do NOT clear this in a finally block. Gradient
        # checkpointing re-executes layer forwards during backward, and the
        # Engram-wrapped layer reads this attribute — if we nulled it out
        # after the initial forward, the recomputed forward would skip Engram
        # and save a different number of tensors than the original forward,
        # triggering a CheckpointError. It's fine to let the stash persist;
        # the next forward overwrites it.
        base._engram_input_ids = ids
        return orig_base_forward(*args, **kwargs)

    base.forward = base_forward_with_stash  # type: ignore[assignment]
    base._engram_orig_forward = orig_base_forward

    # ---- Attach Engram modules at each target layer and wrap their forward ----
    engram_modules: list[EngramModule] = []
    for li in layer_indices:
        layer = layers[li]
        em = engram_module.EngramModule(
            layer_id=li, cfg=engram_cfg, hidden_size=hidden_size)
        target_dtype = next(layer.parameters()).dtype
        target_device = next(layer.parameters()).device

        if host_offload_embedding:
            # Move ONLY the small components to GPU. The big embedding stays
            # on CPU and will be wrapped as CpuHostEmbedding right after.
            em.value_proj.to(device=target_device, dtype=target_dtype)
            em.key_proj.to(device=target_device, dtype=target_dtype)
            em.norm_key.to(device=target_device, dtype=target_dtype)
            em.norm_query.to(device=target_device, dtype=target_dtype)
            em.short_conv.to(device=target_device, dtype=target_dtype)
            # Offsets buffer on GPU (indices need to be on the same device as
            # the outer MultiHeadEmbedding's adds)
            em.embedding.offsets = em.embedding.offsets.to(device=target_device)
            # Cast the embedding table's dtype but keep it on CPU; pin it.
            em.embedding.embedding.to(dtype=target_dtype)
            # Wrap in CpuHostEmbedding (pinned host memory, sparse-grad path)
            offload_engram_embedding(em)
        else:
            # Full GPU placement (legacy path)
            em = em.to(device=target_device, dtype=target_dtype)

        layer.engram_module = em
        engram_modules.append(em)

        orig_layer_forward = layer.forward

        def make_wrapped(_layer: nn.Module, _orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(hidden_states: torch.Tensor, *args, **kwargs):
                ids = getattr(base, "_engram_input_ids", None)
                if ids is not None and hidden_states.shape[1] == ids.shape[1]:
                    # Only apply when seq_len matches — during incremental
                    # generation the decoder may process a 1-token chunk while
                    # ids still points at the prompt; skip in that edge case.
                    delta = _layer.engram_module(hidden_states, ids)
                    hidden_states = hidden_states + delta
                return _orig(hidden_states, *args, **kwargs)
            return wrapped

        layer.forward = make_wrapped(layer, orig_layer_forward)  # type: ignore[assignment]
        layer._engram_orig_forward = orig_layer_forward

    model._engram_installed = True
    model._engram_layer_indices = list(layer_indices)
    return engram_modules


def uninstall_engram(model: nn.Module) -> None:
    """Restore original forwards and remove engram modules."""
    if not getattr(model, "_engram_installed", False):
        return
    base, layers = _resolve_layers(model)
    base.forward = base._engram_orig_forward  # type: ignore[assignment]
    del base._engram_orig_forward
    for li in model._engram_layer_indices:
        layer = layers[li]
        layer.forward = layer._engram_orig_forward  # type: ignore[assignment]
        del layer._engram_orig_forward
        del layer.engram_module
    del model._engram_installed
    del model._engram_layer_indices
