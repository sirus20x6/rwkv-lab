"""Embedding-space probing for training-free MTP.

Reference: "Efficient Training-Free Multi-Token Prediction via Embedding-Space
Probing" (Goel et al. 2026, arXiv:2603.17942). The trick: append K "mask
token" embeddings to a prompt at inference time, run the forward pass once,
and read the logits at the mask positions as speculative predictions of
token[t+1], token[t+2], etc. No training, no extra params.

Mask-embedding variants:
    "mean"   — synthesize each mask as the mean of the prompt's token embeddings
    "random" — draw a random vocab embedding as the mask
    "zero"   — zero vector (baseline; usually worst)

The paper reports 8–12% accepted-length gains on Qwen3-8B/32B. Here we wire
the core probe into a tiny standalone module that can be called during
generation. Speculative verification against the model's own distribution can
be layered on top by any caller that wants lossless decoding.

Usage:
    from embedding_probe import EmbeddingProbe
    probe = EmbeddingProbe(model, tokenizer, k_masks=4, mask_kind="mean")
    spec_tokens = probe.speculate(prompt_ids)  # [B, k_masks] candidate token ids
    # hand spec_tokens to your verifier / acceptance logic
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

MaskKind = Literal["mean", "random", "zero"]


class EmbeddingProbe:
    """Inference-time helper that injects mask-token embeddings into a prompt
    to elicit multi-step predictions in one forward pass."""

    def __init__(self, model, tokenizer=None, k_masks: int = 4,
                 mask_kind: MaskKind = "mean"):
        self.model = model
        self.tokenizer = tokenizer
        self.k_masks = k_masks
        self.mask_kind = mask_kind

        # Locate the input embedding for mask synthesis
        self.embed = model.get_input_embeddings()
        self.device = next(model.parameters()).device
        self.dtype = self.embed.weight.dtype

    def _make_masks(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Construct [B, K, H] mask embeddings."""
        B = prompt_ids.shape[0]
        H = self.embed.weight.shape[1]
        K = self.k_masks
        if self.mask_kind == "mean":
            # Per-sequence mean of prompt token embeddings
            e = self.embed(prompt_ids).to(self.dtype)   # [B, T, H]
            mean = e.mean(dim=1, keepdim=True)          # [B, 1, H]
            return mean.expand(B, K, H).contiguous()
        if self.mask_kind == "random":
            V = self.embed.weight.shape[0]
            idx = torch.randint(0, V, (B, K), device=self.device)
            return self.embed(idx).to(self.dtype)
        if self.mask_kind == "zero":
            return torch.zeros((B, K, H), dtype=self.dtype, device=self.device)
        raise ValueError(f"unknown mask_kind: {self.mask_kind}")

    @torch.no_grad()
    def probe(self, prompt_ids: torch.Tensor, top_k: int = 1) -> dict:
        """Forward-pass probe.
        Args:
            prompt_ids: [B, T] token ids.
            top_k: keep top_k candidates per mask position.
        Returns dict with:
            logits:     [B, K, V]
            top_ids:    [B, K, top_k]
            top_probs:  [B, K, top_k]
            base_next:  [B, V] — distribution at last prompt token (for verification)
        """
        prompt_e = self.embed(prompt_ids).to(self.dtype)       # [B, T, H]
        masks = self._make_masks(prompt_ids)                   # [B, K, H]
        inputs_embeds = torch.cat([prompt_e, masks], dim=1)    # [B, T+K, H]
        B, L, _ = inputs_embeds.shape
        T = prompt_ids.shape[1]
        out = self.model(inputs_embeds=inputs_embeds)
        logits = out.logits                                    # [B, T+K, V]
        mask_logits = logits[:, T:, :]                         # [B, K, V]
        base_next = logits[:, T - 1, :]                        # [B, V]
        probs = F.softmax(mask_logits.float(), dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)
        return {
            "logits": mask_logits,
            "top_ids": top_ids,
            "top_probs": top_probs,
            "base_next": base_next,
        }

    @torch.no_grad()
    def speculate(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Greedy speculative candidates: [B, K] token ids from top-1 of each
        mask position. Hand these to a verifier for lossless decoding."""
        return self.probe(prompt_ids, top_k=1)["top_ids"][:, :, 0]

    @torch.no_grad()
    def verify_greedy(self, prompt_ids: torch.Tensor,
                      spec_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Lossless greedy verification. Returns (accepted_tokens, num_accepted).

        For greedy decoding, spec tokens are accepted iff they match the
        model's argmax conditioned on all previously-accepted prefix. We run
        one verification forward: concat prompt+spec, compare argmax at each
        position to the spec token. First mismatch stops acceptance.
        """
        B = prompt_ids.shape[0]
        K = spec_tokens.shape[1]
        combined = torch.cat([prompt_ids, spec_tokens], dim=1)
        out = self.model(input_ids=combined)
        # At position T + i (0-indexed), the model predicts token T + i + 1,
        # i.e. spec_tokens[:, i]. Logits at position T + i - 1 is the predictor.
        T = prompt_ids.shape[1]
        verifier_pred = out.logits[:, T - 1 : T - 1 + K, :].argmax(dim=-1)  # [B, K]
        match = (verifier_pred == spec_tokens)                              # [B, K]
        # First mismatch per sequence; accept up to (but not including) it.
        # For B=1, this is simple:
        accepted = []
        for b in range(B):
            n_ok = 0
            for i in range(K):
                if match[b, i].item():
                    n_ok += 1
                else:
                    break
            accepted.append(n_ok)
        n_accept = min(accepted) if B > 0 else 0
        # Take the verifier's own prediction at position n_accept to fill the
        # always-correct "one free token" that comes out of the verification pass.
        return verifier_pred, n_accept


def _smoke() -> None:
    """Quick sanity check. Run from command line after loading a model in an
    interactive shell, or via dedicated script that imports this."""
    print("embedding_probe: module loaded OK")
    print("public API: EmbeddingProbe.probe / speculate / verify_greedy")


if __name__ == "__main__":
    _smoke()
