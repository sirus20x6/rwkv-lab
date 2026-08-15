"""RWKV-8 with Kolmogorov-Arnold Network (KAN) feedforward blocks.

This module implements:
1. KANLinear: A modular Kolmogorov-Arnold Network layer using B-splines.
2. RWKV8KANChannelMix: An RWKV-8 ChannelMix layer utilizing KANLinear.
3. RWKV8KANBlock: An RWKV-8 Block combining standard TimeMix and KAN ChannelMix.
4. KANRWKVLanguageModel: A full language model prototype based on KAN-RWKV blocks.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network (KAN) Linear Layer based on B-splines.

    Each connection between an input and output feature computes a learnable 1D spline,
    acting as a non-linear weight function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range: list = [-1, 1],
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Generate B-spline grid points
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1))

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            nn.init.normal_(
                self.spline_weight,
                mean=0.0,
                std=self.scale_noise / (self.grid_size + self.spline_order),
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-splines basis functions for input x.

        Args:
            x: Tensor of shape [..., in_features]
        Returns:
            bases: Tensor of shape [..., in_features, grid_size + spline_order]
        """
        orig_shape = x.shape
        x_reshaped = x.reshape(-1, self.in_features)

        # Bring grid to matching device/dtype
        grid = self.grid.to(device=x.device, dtype=x.dtype)
        grid_uns = grid.unsqueeze(0)  # [1, in_features, num_grid]
        x_uns = x_reshaped.unsqueeze(-1)  # [N, in_features, 1]

        # Order 0 bases
        bases = ((x_uns >= grid_uns[:, :, :-1]) & (x_uns < grid_uns[:, :, 1:])).to(x.dtype)

        # Recurrent calculation of higher-order B-splines
        for k in range(1, self.spline_order + 1):
            bases = (
                (x_uns - grid_uns[:, :, :-(k + 1)])
                / (grid_uns[:, :, k:-1] - grid_uns[:, :, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid_uns[:, :, k + 1:] - x_uns)
                / (grid_uns[:, :, k + 1:] - grid_uns[:, :, 1:-k])
                * bases[:, :, 1:]
            )

        return bases.reshape(*orig_shape, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KAN linear activation pass.

        Args:
            x: Input of shape [..., in_features]
        """
        # Linear base mapping
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Non-linear spline mapping
        splines = self.b_splines(x)  # [..., in_features, grid_size + spline_order]
        spline_output = torch.einsum("...ig,oig->...o", splines, self.spline_weight)

        return base_output + self.scale_spline * spline_output


class RWKV8KANChannelMix(nn.Module):
    """An RWKV-8 ChannelMix block with Kolmogorov-Arnold Network (KAN) feedforward layer."""

    def __init__(
        self,
        hidden_size: int,
        layer_idx: Optional[int] = None,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = hidden_size  # Gated DeepEmbed size compatibility
        self.layer_idx = layer_idx

        self.x_k = nn.Parameter(torch.zeros(hidden_size))

        self.kan = KANLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_gate: Optional[torch.Tensor] = None,
        shift_state: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Token shift logic
        prev = torch.zeros_like(hidden_states)
        if shift_state is not None:
            prev[:, :1] = shift_state.to(hidden_states.dtype).reshape(
                hidden_states.shape[0], 1, hidden_states.shape[-1]
            )
        if hidden_states.shape[1] > 1:
            prev[:, 1:] = hidden_states[:, :-1]
        if reset_mask is not None:
            prev = prev.masked_fill(reset_mask[..., None], 0.0)

        xk = self.x_k.to(dtype=hidden_states.dtype).view(1, 1, -1)
        mixed = hidden_states + (prev - hidden_states) * xk

        # Spline activation mapping
        out = self.kan(mixed)

        # Gated DeepEmbed
        if hidden_gate is not None:
            out = out * hidden_gate

        if return_state:
            return out, hidden_states[:, -1:]
        return out


class RWKV8KANBlock(nn.Module):
    """An RWKV-8 Block containing standard TimeMix and KAN ChannelMix."""

    def __init__(
        self,
        d_model: int,
        i: int,
        num_layers: int,
        head_size: int = 16,
        deepembed: bool = True,
        vocab_size: int = 256,
        de_dim: Optional[int] = None,
        grid_size: int = 5,
        spline_order: int = 3,
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

        # KAN-based ChannelMix
        self.ffn = RWKV8KANChannelMix(
            hidden_size=d_model,
            layer_idx=i,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        if self.deepembed:
            # DeepEmbed (BlinkDL, RWKV-8): per-layer per-token multiplicative FFN gate.
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
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
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
                return_state=True,
            )
        else:
            att_out, v_first_out = self.att(
                ln_x,
                v_first=v_first,
                return_v_first=True,
                initial_state=initial_state_att,
                shift_state=shift_state_att,
            )
        x = x + att_out

        # 2. KAN ChannelMix with DeepEmbed and LayerNorm
        ln_x2 = self.ln2(x)

        # Compute DeepEmbed gate
        gate = None
        if self.deepembed:
            g = self.de_emb(ids)
            if self.de_proj is not None:
                g = self.de_proj(g)
            gate = 1.0 + g

        if return_state:
            ffn_out, next_shift_ffn = self.ffn(
                ln_x2,
                hidden_gate=gate,
                shift_state=shift_state_ffn,
                return_state=True,
            )
        else:
            ffn_out = self.ffn(
                ln_x2,
                hidden_gate=gate,
                shift_state=shift_state_ffn,
            )
        x = x + ffn_out

        if return_state:
            return x, v_first_out, (next_wkv_state, next_shift_att), next_shift_ffn

        return x, v_first_out


class KANRWKVLanguageModel(nn.Module):
    """An RWKV-8 language model with Kolmogorov-Arnold Network (KAN) ChannelMix feedforward layers."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_layers: int = 2,
        head_size: int = 16,
        deepembed: bool = True,
        de_dim: Optional[int] = None,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            RWKV8KANBlock(
                d_model=d_model,
                i=i,
                num_layers=n_layers,
                head_size=head_size,
                deepembed=deepembed,
                vocab_size=vocab_size,
                de_dim=de_dim,
                grid_size=grid_size,
                spline_order=spline_order,
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
        states: Optional[list] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
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
                    return_state=True,
                )
                next_states.append((next_att, next_ffn))
            else:
                x, v_first = block(
                    x,
                    ids=ids,
                    v_first=v_first,
                )

        x = self.ln_out(x)
        logits = self.head(x)

        if return_state:
            return logits, next_states
        return logits
