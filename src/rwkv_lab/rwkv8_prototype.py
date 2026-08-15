"""RWKV-8 Prototype Implementation.

This module implements a clean, modular, and fully self-contained RWKV-8 prototype model
inheriting features from the RWKV-8 architecture, specifically:
- RWKV8TimeMixDeltaNet (time-mix block with cross-layer value-residual tracking)
- RWKV8ChannelMixDeltaNet (channel-mix block)
- DeepEmbed (per-layer, per-token multiplicative FFN gate: 1 + de_emb(ids))

It supports both:
1. Parallel training/evaluation (sequence-level forward pass).
2. Autoregressive recurrent decoding with full state carrying.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet, RWKV8ChannelMixDeltaNet


class RWKV8Block(nn.Module):
    """An RWKV-8 Block containing time-mix, channel-mix, and DeepEmbed gating.

    Each block uses standard layer normalization, cross-layer value-residual tracking,
    and a zero-init DeepEmbed multiplicative gate for FFN.
    """

    def __init__(
        self,
        d_model: int,
        i: int,
        num_layers: int,
        head_size: int = 16,
        deepembed: bool = True,
        vocab_size: int = 256,
        de_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.i = i
        self.d_model = d_model
        self.deepembed = deepembed

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Standard TimeMix at the token/byte level
        self.att = RWKV8TimeMixDeltaNet(
            hidden_size=d_model,
            num_heads=d_model // head_size,
            head_size=head_size,
            layer_idx=i,
            depth_layer_id=i,
            depth_n_layer=num_layers,
            is_first_rwkv_layer=(i == 0),
            out_correct=False,
        )

        # Standard ChannelMix at the token/byte level
        self.ffn = RWKV8ChannelMixDeltaNet(
            hidden_size=d_model,
            ffn_hidden_size=d_model * 4,
            layer_idx=i,
        )

        if self.deepembed:
            # DeepEmbed (BlinkDL, RWKV-8): per-layer per-token multiplicative FFN gate.
            # Applied on FFN hidden, so target output dimension is ffn_hidden_size (4 * d_model).
            # We initialize the embedding weights to 0 so the gate is exactly 1.0 (identity) at step 0.
            de_out_dim = self.ffn.ffn_hidden_size
            if de_dim is not None and de_dim < de_out_dim:
                self.de_emb = nn.Embedding(vocab_size, de_dim)
                self.de_proj = nn.Linear(de_dim, de_out_dim, bias=False)
                nn.init.zeros_(self.de_emb.weight)
                nn.init.zeros_(self.de_proj.weight)
            else:
                self.de_emb = nn.Embedding(vocab_size, de_out_dim)
                self.de_proj = None
                nn.init.zeros_(self.de_emb.weight)

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        initial_state_att: Optional[torch.Tensor] = None,
        shift_state_att: Optional[torch.Tensor] = None,
        shift_state_ffn: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ]:
        # 1. TimeMix with LayerNorm
        ln_x = self.ln1(x)
        if return_state:
            att_out, next_wkv_state, next_shift_att, v_first_out = self.att(
                ln_x,
                v_first=v_first,
                return_v_first=True,
                initial_state=initial_state_att,
                shift_state=shift_state_att,
                return_state=True
            )
        else:
            att_out, v_first_out = self.att(
                ln_x,
                v_first=v_first,
                return_v_first=True,
                initial_state=initial_state_att,
                shift_state=shift_state_att
            )
        x = x + att_out

        # 2. ChannelMix with DeepEmbed and LayerNorm
        ln_x2 = self.ln2(x)

        # Compute DeepEmbed gate
        gate = None
        if self.deepembed:
            # gate = 1.0 + de_emb(ids)
            g = self.de_emb(ids)
            if self.de_proj is not None:
                g = self.de_proj(g)
            gate = 1.0 + g

        if return_state:
            ffn_out, next_shift_ffn = self.ffn(
                ln_x2,
                hidden_gate=gate,
                shift_state=shift_state_ffn,
                return_state=True
            )
        else:
            ffn_out = self.ffn(
                ln_x2,
                hidden_gate=gate,
                shift_state=shift_state_ffn
            )
        x = x + ffn_out

        if return_state:
            return x, v_first_out, (next_wkv_state, next_shift_att), next_shift_ffn

        return x, v_first_out


class RWKV8LanguageModel(nn.Module):
    """An RWKV-8 language model prototype.

    Consists of token embeddings, multiple RWKV-8 blocks, post layer-normalization,
    and a language model projection head. Supports both parallel sequence training
    and recurrent causal decoding.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_layers: int = 2,
        head_size: int = 16,
        deepembed: bool = True,
        de_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            RWKV8Block(
                d_model=d_model,
                i=i,
                num_layers=n_layers,
                head_size=head_size,
                deepembed=deepembed,
                vocab_size=vocab_size,
                de_dim=de_dim,
            )
            for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Default initialization standard for RWKV
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        states: Optional[list] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Forward pass supporting both parallel sequence training and recurrent step execution.

        Args:
            ids: Input token ids of shape [B, T].
            states: List of states per layer, each being a tuple of:
                    ((wkv_matrix_state, time_shift_state_att), shift_state_ffn)
            return_state: If True, returns the updated recurrent states.
        """
        x = self.emb(ids)

        v_first = None
        next_states = []

        for i, block in enumerate(self.blocks):
            if return_state:
                # Extract block-specific states
                init_state_att = states[i][0][0] if states else None
                shift_state_att = states[i][0][1] if states else None
                shift_state_ffn = states[i][1] if states else None

                x, v_first, next_att, next_ffn = block(
                    x,
                    ids=ids,
                    v_first=v_first,
                    initial_state_att=init_state_att,
                    shift_state_att=shift_state_att,
                    shift_state_ffn=shift_state_ffn,
                    return_state=True
                )
                next_states.append((next_att, next_ffn))
            else:
                x, v_first = block(
                    x,
                    ids=ids,
                    v_first=v_first
                )

        x = self.ln_out(x)
        logits = self.head(x)

        if return_state:
            return logits, next_states
        return logits
