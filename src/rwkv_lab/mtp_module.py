"""
Multi-Token Prediction (MTP) module for Qwen3.6-35B-A3B, adapted for HF-
transformers training. Port of vLLM's Qwen3_5MultiTokenPredictor using HF's
own Qwen3_5MoeDecoderLayer for the single MTP decoder block.

Architecture:
    input_ids_shifted  -> embed_tokens (shared w/ backbone) -> embeds
    backbone_hidden    -> (pre_fc_norm_hidden)
    embeds             -> (pre_fc_norm_embedding)
    cat(embeds, hidden) -> fc (2H -> H) -> decoder_layer(s) -> norm -> out

At training time, at position p the MTP uses input_ids_shifted[p] = token[p+1]
and backbone_hidden[p] to predict token[p+2] via the shared lm_head.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local transformers build
    Qwen3_5MoeDecoderLayer = None
    Qwen3_5MoeRMSNorm = None

# Dense Qwen3.5 (e.g. 9B) uses a different decoder layer + RMSNorm pair. Try
# importing both so the same MTP module can target either backbone.
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
    )
except ModuleNotFoundError:  # pragma: no cover
    Qwen3_5DecoderLayer = None
    Qwen3_5RMSNorm = None


class Qwen3_5MoeMTPModule(nn.Module):
    """HF-friendly port of vLLM's Qwen3_5MultiTokenPredictor."""

    def __init__(self, config, full_attn_layer_idx: Optional[int] = None) -> None:
        super().__init__()
        # Pick the right backbone decoder/RMSNorm pair: MoE config (35B) uses
        # the qwen3_5_moe variants; dense config (9B) uses the qwen3_5 variants.
        # Detection: only the MoE config has num_experts_per_tok.
        is_moe = hasattr(config, "num_experts_per_tok")
        if is_moe:
            DecoderLayer, RMSNorm = Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm
            variant = "qwen3_5_moe"
        else:
            DecoderLayer, RMSNorm = Qwen3_5DecoderLayer, Qwen3_5RMSNorm
            variant = "qwen3_5 (dense)"
        if DecoderLayer is None or RMSNorm is None:
            raise ImportError(
                f"Qwen3_5MoeMTPModule requires a transformers build with the "
                f"{variant} module. Install/upgrade transformers before using "
                "--install-mtp=1."
            )
        self.config = config
        # Pick a "full_attention" layer_idx from backbone's layer_types, which
        # the DecoderLayer constructor uses to decide which attn class to build.
        if full_attn_layer_idx is None or config.layer_types[full_attn_layer_idx] != "full_attention":
            full_attn_layer_idx = next(
                i for i, t in enumerate(config.layer_types) if t == "full_attention"
            )
        self._full_attn_layer_idx = full_attn_layer_idx

        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        H = config.hidden_size

        # Shared with backbone — set externally via set_shared_embed_tokens().
        # Qwen3.6 config has mtp_use_dedicated_embeddings=False, so this sharing
        # is correct. If a future model has dedicated MTP embeddings, we'd own
        # a separate nn.Embedding here.
        self.embed_tokens: Optional[nn.Embedding] = None

        self.fc = nn.Linear(H * 2, H, bias=False)
        self.pre_fc_norm_hidden = RMSNorm(H, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(H, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=full_attn_layer_idx)
            for _ in range(self.num_mtp_layers)
        ])

        self.norm = RMSNorm(H, eps=config.rms_norm_eps)

    def set_shared_embed_tokens(self, embed_tokens: nn.Embedding) -> None:
        # Assign without adding as a submodule so its params aren't double-counted.
        object.__setattr__(self, "embed_tokens", embed_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "Qwen3_5MoeMTPModule.set_shared_embed_tokens() must be called first"
        )
        embeds = self.embed_tokens(input_ids)
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        x = self.fc(torch.cat([e, h], dim=-1))
        for layer in self.layers:
            layer_out = layer(
                hidden_states=x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        return self.norm(x)


# ---------------------------------------------------------------------------
# Helper: read mtp.* weights directly from the base checkpoint's safetensors
# shards (HF silently drops them per `_keys_to_ignore_on_load_unexpected`).
# ---------------------------------------------------------------------------

def load_mtp_weights_from_checkpoint(model_dir, device, dtype) -> dict:
    """Return a dict of {local_mtp_key -> tensor} where local_mtp_key has the
    'mtp.' prefix stripped (matches keys expected by Qwen3_5MoeMTPModule's
    state_dict — minus the layer_idx placement and minus embed_tokens which
    is shared)."""
    import json
    from pathlib import Path
    from safetensors import safe_open

    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    out = {}
    mtp_keys_by_shard: dict[str, list[str]] = {}
    for full_key, shard in wmap.items():
        if full_key.startswith("mtp."):
            mtp_keys_by_shard.setdefault(shard, []).append(full_key)
    for shard, keys in mtp_keys_by_shard.items():
        with safe_open(model_dir / shard, framework="pt") as f:
            for full_key in keys:
                t = f.get_tensor(full_key)
                # Strip the leading "mtp."
                local = full_key[len("mtp."):]
                out[local] = t.to(dtype=dtype)  # keep on CPU; caller moves to device
    return out
