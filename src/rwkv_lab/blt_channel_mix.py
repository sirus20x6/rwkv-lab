"""BLT (Byte Latent Token) style ChannelMix drop-in replacement for RWKV-7/8.

Reference: Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than
Tokens", arXiv:2412.09871, https://arxiv.org/abs/2412.09871.

This module replaces the standard word/token-level ChannelMix (FFN) with a
BLT-style patch-level mixer. In parallel training mode, it dynamically pools fine-grained
hidden states (character/byte-level) into variable-sized patches using an entropy-head
predictor, runs ChannelMix on the patches, and unpools them back.
In recurrent inference mode, it utilizes a streaming state tracker to perform causal
pooling, ensuring exact alignment with parallel training at patch boundaries.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.byte_patches import entropy_patch_ids


class RWKV7BLTChannelMix(nn.Module):
    """A BLT-style ChannelMix module that is a drop-in replacement for RWKV8ChannelMixDeltaNet.

    It operates on hidden states, predicts next-byte entropy to identify patch boundaries,
    pools representations into patches, processes them via a standard ChannelMix FFN,
    and unpools them back to the original resolution.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: Optional[int] = None,
        *,
        layer_idx: Optional[int] = None,
        initializer_range: float = 0.02,
        init_output_scale: float = 1e-3,
        threshold: float = 3.0,
        min_patch: int = 1,
        max_patch: int = 8,
        streaming_mode: str = "causal",  # "causal" or "discrete"
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size or hidden_size * 4
        self.layer_idx = layer_idx
        self.init_output_scale = init_output_scale

        self.threshold = threshold
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.streaming_mode = streaming_mode

        # Entropy predictor head (predicts distribution over 256 bytes)
        self.entropy_head = nn.Linear(hidden_size, 256)

        # Standard ChannelMix parameters
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, self.ffn_hidden_size, bias=False)
        self.value = nn.Linear(self.ffn_hidden_size, hidden_size, bias=False)

        # Initialization
        nn.init.normal_(self.entropy_head.weight, mean=0.0, std=initializer_range)
        nn.init.constant_(self.entropy_head.bias, 0.0)
        nn.init.normal_(self.key.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(
            self.value.weight,
            mean=0.0,
            std=initializer_range * init_output_scale,
        )

    def compute_entropy_loss(self, hidden_states: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
        """Compute the next-byte prediction cross-entropy loss for training the entropy head.

        Args:
            hidden_states: [B, T, C] continuous hidden states.
            target_bytes: [B, T] long tensor of byte indices in [0, 255].
        """
        # Align targets for next-byte prediction: input hidden_states[t] predicts target_bytes[t+1]
        logits = self.entropy_head(hidden_states[:, :-1]) # [B, T-1, 256]
        targets = target_bytes[:, 1:] # [B, T-1]
        return F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        hidden_gate: Optional[torch.Tensor] = None,
        shift_state: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Union[torch.Tensor, tuple]]:
        B, T, C = hidden_states.shape

        if T > 1:
            # -----------------------------------------------------------------
            # PARALLEL TRAINING MODE
            # -----------------------------------------------------------------
            # 1. Compute entropy at each position
            logits = self.entropy_head(hidden_states)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1) # [B, T]

            # 2. Extract patch IDs using entropy. Detach to prevent non-differentiable indexing
            patch_ids = entropy_patch_ids(
                entropy.detach(),
                threshold=self.threshold,
                min_patch=self.min_patch,
                max_patch=self.max_patch
            ) # [B, T]

            # Store for downstream inspection/logging
            self.last_entropy = entropy.detach()
            self.last_patch_ids = patch_ids

            # 3. Dynamic Mean Pooling
            n = int(patch_ids.max()) + 1
            values = hidden_states.new_zeros(B, n, C)
            counts = hidden_states.new_zeros(B, n)
            values.scatter_add_(1, patch_ids[..., None].expand(-1, -1, C), hidden_states)
            counts.scatter_add_(1, patch_ids, torch.ones_like(patch_ids, dtype=hidden_states.dtype))
            values = values / counts.clamp_min(1).unsqueeze(-1)

            # 4. Patch-level FFN (ChannelMix)
            prev_values = torch.zeros_like(values)
            prev_values[:, 1:] = values[:, :-1]
            xk = self.x_k.to(dtype=hidden_states.dtype).view(1, 1, -1)
            mixed = values + (prev_values - values) * xk

            k = torch.square(F.relu(self.key(mixed)))
            if hidden_gate is not None:
                # Gated / DeepEmbed mode
                # Gated weights pooled similarly
                gate_values = hidden_gate.new_zeros(B, n, hidden_gate.shape[-1])
                gate_values.scatter_add_(1, patch_ids[..., None].expand(-1, -1, hidden_gate.shape[-1]), hidden_gate)
                gate_counts = hidden_gate.new_zeros(B, n)
                gate_counts.scatter_add_(1, patch_ids, torch.ones_like(patch_ids, dtype=hidden_gate.dtype))
                gate_values = gate_values / gate_counts.clamp_min(1).unsqueeze(-1)
                k = k * gate_values

            out_values = self.value(k)

            # 5. Unpooling (gather back to T resolution)
            index = patch_ids.unsqueeze(-1).expand(*patch_ids.shape, C)
            out = out_values.gather(1, index)

            if return_state:
                # For compatibility, we return last token state as part of RWKV interface
                return out, hidden_states[:, -1:]
            return out

        else:
            # -----------------------------------------------------------------
            # RECURRENT INFERENCE MODE (T == 1)
            # -----------------------------------------------------------------
            # Unpack or initialize recurrent BLT state
            if shift_state is None or (isinstance(shift_state, torch.Tensor) and shift_state.numel() == 0):
                running_sum = torch.zeros(B, C, device=hidden_states.device, dtype=hidden_states.dtype)
                length = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
                prev_patch_state = torch.zeros(B, C, device=hidden_states.device, dtype=hidden_states.dtype)
                last_emitted = torch.zeros(B, C, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                running_sum, length, prev_patch_state, last_emitted = shift_state

            # Compute entropy for current step
            logits = self.entropy_head(hidden_states[:, 0]) # [B, 256]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1) # [B]

            # Update running patch accumulation
            running_sum = running_sum + hidden_states[:, 0]
            length = length + 1.0

            # Check boundary condition (vectorized)
            split = (length >= self.max_patch) | ((length >= self.min_patch) & (entropy >= self.threshold))

            # Compute ChannelMix FFN on the completed or currently accumulating patch state
            # 1. Patch state:
            patch_state = running_sum / length.unsqueeze(-1)
            # 2. Token shift mix with previous patch state:
            xk = self.x_k.to(dtype=hidden_states.dtype).view(1, -1)
            mixed = patch_state + (prev_patch_state - patch_state) * xk
            # 3. FFN value:
            k = torch.square(F.relu(self.key(mixed)))
            if hidden_gate is not None:
                k = k * hidden_gate[:, 0]
            ffn_out = self.value(k)

            # For split boundaries, we lock in the state and update trackers
            new_prev_patch_state = torch.where(split.unsqueeze(-1), patch_state, prev_patch_state)
            new_last_emitted = torch.where(split.unsqueeze(-1), ffn_out, last_emitted)
            new_running_sum = torch.where(split.unsqueeze(-1), torch.zeros_like(running_sum), running_sum)
            new_length = torch.where(split, torch.zeros_like(length), length)

            # Select output based on streaming mode
            if self.streaming_mode == "causal":
                # In causal mode, we emit the current running FFN output, which converges to the parallel
                # training FFN output at the final token of the patch (where split is True).
                out = ffn_out.unsqueeze(1)
            else:
                # In discrete mode, we only output the representation of the last fully completed patch.
                out = new_last_emitted.unsqueeze(1)

            next_state = (new_running_sum, new_length, new_prev_patch_state, new_last_emitted)

            if return_state:
                return out, next_state
            return out
