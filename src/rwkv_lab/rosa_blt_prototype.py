"""ROSA + BLT Language Model Prototype.

This module implements a clean, unified language model that combines:
1. ROSA (Rapid Online Suffix Automaton) for in-context exact suffix retrieval.
2. BLT (Byte Latent Transformer) for dynamic, context-aware byte-latent patching.

The model is byte-level (vocab_size=256), training the entropy predictor head on the
contextualized hidden states of the model to dynamically pool raw UTF-8 byte streams into
variable-length semantic patches.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet
from rwkv_lab.blt_channel_mix import RWKV7BLTChannelMix
from rwkv_lab.rosa import RosaLayer


class BLT_ROSA_Block(nn.Module):
    """An RWKV layer combining ROSA suffix-matching with BLT dynamic-patching ChannelMix."""

    def __init__(
        self,
        d_model: int,
        i: int,
        num_layers: int,
        head_size: int = 16,
        threshold: float = 3.0,
        max_patch: int = 8,
        rosa_M: int = 4,
    ) -> None:
        super().__init__()
        self.i = i
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 1. Standard TimeMix at the byte level
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

        # 2. ROSA suffix-automaton in-context retrieval layer
        # Installed additively alongside attention
        self.rosa = RosaLayer(hidden_size=d_model, M=rosa_M, max_match=32)

        # 3. BLT ChannelMix at the dynamic patch level
        self.ffn = RWKV7BLTChannelMix(
            hidden_size=d_model,
            ffn_hidden_size=d_model * 4,
            layer_idx=i,
            threshold=threshold,
            min_patch=1,
            max_patch=max_patch,
            streaming_mode="causal",
        )

    def forward(
        self,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        target_bytes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for parallel training.

        Args:
            x: Input sequence [B, T, C].
            v_first: Cross-layer value residual from Layer 0.
            target_bytes: Target byte IDs [B, T] for training the BLT entropy head.

        Returns:
            x_out: Output sequence [B, T, C].
            v_first_out: Updated value residual.
            entropy_loss: Next-byte entropy prediction loss (if target_bytes is provided).
        """
        # 1. TimeMix + ROSA in-context injection
        ln_x = self.ln1(x)

        # Standard attention
        att_out, v_first_out = self.att(ln_x, v_first=v_first, return_v_first=True)

        # Add ROSA suffix retrieval injection on the normalized states
        rosa_inj = self.rosa.injection(ln_x)

        # Combine additively into residual stream
        x = x + att_out + rosa_inj

        # 2. BLT ChannelMix
        ln_x2 = self.ln2(x)

        # Compute entropy head loss if targets are provided
        entropy_loss = None
        if target_bytes is not None:
            entropy_loss = self.ffn.compute_entropy_loss(ln_x2, target_bytes)

        ffn_out = self.ffn(ln_x2)
        x = x + ffn_out

        return x, v_first_out, entropy_loss


class BLT_ROSA_LanguageModel(nn.Module):
    """Unified ROSA + BLT byte-level language model."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_layers: int = 2,
        head_size: int = 16,
        threshold: float = 3.0,
        max_patch: int = 8,
        rosa_M: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BLT_ROSA_Block(
                d_model=d_model,
                i=i,
                num_layers=n_layers,
                head_size=head_size,
                threshold=threshold,
                max_patch=max_patch,
                rosa_M=rosa_M,
            )
            for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

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
        target_bytes: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list[torch.Tensor]]]:
        """Parallel forward pass.

        Args:
            ids: Input byte sequence [B, T].
            target_bytes: Target byte sequence [B, T] for training the entropy head.

        Returns:
            logits: Next-byte logits [B, T, vocab_size].
            entropy_losses: List of entropy prediction losses per layer (if target_bytes provided).
        """
        x = self.emb(ids)

        v_first = None
        entropy_losses = []

        for block in self.blocks:
            x, v_first, e_loss = block(x, v_first=v_first, target_bytes=target_bytes)
            if e_loss is not None:
                entropy_losses.append(e_loss)

        x = self.ln_out(x)
        logits = self.head(x)

        if target_bytes is not None:
            return logits, entropy_losses
        return logits
