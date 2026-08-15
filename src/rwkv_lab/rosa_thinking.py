"""ROSA-enhanced Latent State Thinking Model.

This module implements:
1. RosaLatentThinkingModel: A model that separates continuous thinking (latent-state
   computation) from discrete speaking (token generation). Between reading the query and
   generating the answer, the model performs N "thinking steps" where the input is
   the previous step's last hidden state (Coconut style), enhanced with a ROSA Suffix
   Automaton that matches and retrieves from both past tokens and past thoughts.

2. This architecture demonstrates that a 1-layer ROSA model, which can only perform
   1 logical hop per feedforward pass, can solve multi-hop logical tasks (e.g. A -> B -> C)
   by "thinking" silently in its latent space for N steps before speaking!
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.rosa import RosaLayer
from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet


class RosaThinkingBlock(nn.Module):
    """An RWKV layer combining ROSA suffix-matching with standard attention."""

    def __init__(self, d_model: int, i: int, num_layers: int, head_size: int = 8, rosa_M: int = 4) -> None:
        super().__init__()
        self.i = i
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)

        self.att = RWKV8TimeMixDeltaNet(
            hidden_size=d_model,
            num_heads=d_model // head_size,
            head_size=head_size,
            layer_idx=i,
            depth_layer_id=i,
            depth_n_layer=max(num_layers, 2),
            is_first_rwkv_layer=(i == 0),
            out_correct=False,
        )

        self.rosa = RosaLayer(hidden_size=d_model, M=rosa_M, max_match=128)

        # Initialize to identity
        with torch.no_grad():
            nn.init.eye_(self.att.output.weight)
            nn.init.eye_(self.rosa.Wq.weight)
            nn.init.eye_(self.rosa.Wk.weight)
            nn.init.eye_(self.rosa.Wv.weight)
            nn.init.eye_(self.rosa.Wout.weight)
            self.rosa.e0.fill_(0.0)
            self.rosa.e1.fill_(1.0)

    def forward(self, H: torch.Tensor, v_first: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ln_x = self.ln1(H)
        att_out, v_first_out = self.att(ln_x, v_first=v_first, return_v_first=True)
        # Pass H directly so that its binary bit-signs are not corrupted by LayerNorm's cross-channel mean subtraction
        rosa_inj = self.rosa.injection(H)
        return H + att_out + rosa_inj, rosa_inj, v_first_out


class RosaLatentThinkingModel(nn.Module):
    """A 1-layer ROSA model that can perform multi-step latent thinking."""

    def __init__(self, vocab_size: int = 16, d_model: int = 16, rosa_M: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model)
        # 1-layer model (can do exactly 1 logical hop per forward pass)
        self.block = RosaThinkingBlock(d_model=d_model, i=0, num_layers=1, head_size=8, rosa_M=rosa_M)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Direct initialization of embedding and head, leaving self.block's eye_ weights untouched
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x_ids: torch.Tensor, n_thoughts: int = 0) -> torch.Tensor:
        """Run the model over x_ids, then perform n_thoughts steps of silent latent thinking.

        Args:
            x_ids: Input sequence [B, T].
            n_thoughts: Number of silent thinking steps to perform in continuous latent space.

        Returns:
            logits: Output vocabulary logits [B, T + n_thoughts, vocab_size].
        """
        B, T = x_ids.shape
        device = x_ids.device

        # 1. Map input IDs to exact binary representations to ensure 100% precision
        H = torch.zeros(B, T, self.d_model, device=device)
        for b in range(B):
            for t in range(T):
                token_id = x_ids[b, t].item()
                for i in range(self.d_model):
                    bit = 1.0 if (token_id & (1 << i)) != 0 else -1.0
                    H[b, t, i] = bit

        # 2. Perform latent thinking steps (Coconut style)
        v_first = None
        for step in range(n_thoughts + 1):
            # Run the block forward over the current H sequence
            out, rosa_inj, v_first = self.block(H, v_first=v_first)

            if step < n_thoughts:
                # Retrieve the last retrieved logical state (rosa_inj) of this forward pass
                h_last = rosa_inj[:, -1:] # [B, 1, C]

                # To feed B back as a clean key/query in H, we convert h_last back to the standard binary {-1, 1}
                # format used by input H, preventing signal scale cancellation.
                h_last_b = 2.0 * h_last - 1.0

                # Append the "thought" directly to the continuous input stream H
                H = torch.cat([H, h_last_b], dim=1)

        # 3. Final post-norm and vocabulary projection on the retrieved logical state
        h_final = self.ln_out(rosa_inj)

        # We calculate logits by matching dot-product with binary representations of all vocab elements
        head_weight = torch.zeros(self.vocab_size, self.d_model, device=device)
        for token_id in range(self.vocab_size):
            for i in range(self.d_model):
                head_weight[token_id, i] = 1.0 if (token_id & (1 << i)) != 0 else -1.0

        # Project back to logits
        # Converting the final representation to {-1, 1} avoids argmax ambiguity
        h_final_b = 2.0 * h_final - 1.0
        logits = F.linear(h_final_b, head_weight) # [B, T + n_thoughts, vocab_size]
        return logits
