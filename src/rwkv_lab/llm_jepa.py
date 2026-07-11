"""LLM-JEPA (arXiv:2509.14252, Huang/LeCun/Balestriero) — a Joint-Embedding Predictive
objective for LLMs, added alongside the standard next-token loss.

Idea: a training example is two VIEWS of the same knowledge — (Text, Code) — e.g. a natural-
language request and the code that satisfies it. LLM-JEPA predicts the *embedding* of one view
from the other, in latent space (not token space):

    L = L_LM(next-token)  +  λ · d( Pred(Enc(Text)), Enc(Code) )

Faithful design choices from the paper (each ablation-backed):
  * Enc = the SAME LLM for both views (NO separate target encoder, NO stop-grad, NO EMA — the
    generative loss keeps it from collapsing).
  * Embedding = the last-token hidden state of the last layer.
  * Pred = the LLM ITSELF: append k learnable [PRED] tokens after the Text and read the last
    hidden. k=0 => Pred is identity. (Reusing the LLM weights is cheaper than a separate MLP.)
  * d = COSINE distance (1 − cos). The paper's metric ablation: cosine 71.5% > MSE 70.6% ≫
    ℓ2 2.2% (collapses) — so the metric matters; we default to cosine.
  * Direction: Text → Code (predicting Code from Text beat the reverse).

Scope: this is a PAIRED-DATA objective for a supervised/instruction finetune stage (it needs
two views per example), not the unpaired conversion distillation. A trainer supplies the model
body (returning hidden states), the Text input embeddings, and the Code hidden states. Cheap:
no new large params (the predictor is k extra positions), ~2× forward, zero inference overhead.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def last_token_embedding(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Last-token (last-layer) hidden state as the view embedding. hidden [B,T,C]; attention_mask
    [B,T] (1=real, 0=pad). With a mask, picks each row's last non-pad token; else the last."""
    if attention_mask is None:
        return hidden[:, -1]
    idx = attention_mask.long().sum(dim=1) - 1                  # last real position per row
    idx = idx.clamp_min(0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def cosine_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − cos(pred, target), mean over the batch. target is NOT detached (same encoder, no
    stop-grad, per the paper). Range [0, 2]; 0 when the embeddings point the same way."""
    return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


class LLMJEPA(nn.Module):
    """The LLM-JEPA predictor: k learnable [PRED] token embeddings. `predict` appends them to a
    Text sequence, runs the provided model body, and returns the last [PRED] token's hidden as
    the prediction. k=0 => identity (pool the Text hidden directly)."""

    def __init__(self, d_model: int, k: int = 1):
        super().__init__()
        self.k = int(k)
        self.pred_tokens = (nn.Parameter(torch.randn(self.k, d_model) * 0.02)
                            if self.k > 0 else None)

    def predict(self, text_input_embeds: torch.Tensor,
                model_body: Callable[..., torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """text_input_embeds [B,T,C] input embeddings of the Text view. model_body(inputs_embeds,
        attention_mask) -> hidden [B,L,C]. Returns the prediction embedding [B,C]."""
        if self.k == 0:                                        # identity predictor
            h = model_body(text_input_embeds, attention_mask)
            return last_token_embedding(h, attention_mask)
        B, _, C = text_input_embeds.shape
        pt = self.pred_tokens.to(text_input_embeds.dtype).unsqueeze(0).expand(B, self.k, C)
        seq = torch.cat([text_input_embeds, pt], dim=1)        # [B, T+k, C]
        mask = attention_mask
        if mask is not None:
            mask = torch.cat([mask, mask.new_ones(B, self.k)], dim=1)
        h = model_body(seq, mask)                              # [B, T+k, C]
        return h[:, -1]                                        # last [PRED] token = prediction

    def loss(self, text_input_embeds: torch.Tensor, code_hidden: torch.Tensor,
             model_body: Callable[..., torch.Tensor],
             text_mask: Optional[torch.Tensor] = None,
             code_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """JEPA cosine loss: predict the Code embedding from the Text view. code_hidden [B,T_c,C]
        are the Code view's hidden states (from the same model); its last-token is the target."""
        pred = self.predict(text_input_embeds, model_body, text_mask)
        target = last_token_embedding(code_hidden, code_mask)
        return cosine_jepa_loss(pred, target)


class ActionConditionedJEPA(nn.Module):
    """Action-conditioned latent transition model and dimension diagnostic.

    Motivated by Lu et al., *A Generalization Theory for JEPA-Based World
    Models* (2026), https://arxiv.org/abs/2606.27014: the spectral JEPA
    objective is a low-rank factorization of an action-conditioned co-occurrence
    matrix. This module measures that approximation trade-off; it does not claim
    the paper's planning-regret guarantees without their assumptions.
    """
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(observation_dim, latent_dim)
        self.action_encoder = nn.Linear(action_dim, latent_dim)
        self.predictor = nn.GRUCell(latent_dim, latent_dim)

    def forward(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state = self.encoder(observation)
        for step in range(actions.shape[-2]):
            state = self.predictor(self.action_encoder(actions[..., step, :]), state)
        return state

    def loss(self, observation: torch.Tensor, actions: torch.Tensor,
             future_observation: torch.Tensor, variance_weight: float = 0.01) -> torch.Tensor:
        prediction = self(observation, actions)
        target = self.encoder(future_observation).detach()
        variance = F.relu(1.0 - prediction.float().std(dim=0, unbiased=False)).mean()
        return F.mse_loss(prediction.float(), target.float()) + variance_weight * variance


def action_conditioned_rank_diagnostic(current: torch.Tensor, action: torch.Tensor,
                                       future: torch.Tensor,
                                       dimensions: tuple[int, ...] = (4, 8, 16, 32)) -> list[dict]:
    """SVD residual curve for the empirical [state, action]→future cross-covariance."""
    if current.shape[0] != action.shape[0] or current.shape[0] != future.shape[0]:
        raise ValueError("current, action, and future must share sample dimension")
    source = torch.cat((current.float(), action.float()), -1)
    source, target = source - source.mean(0), future.float() - future.float().mean(0)
    singular = torch.linalg.svdvals(source.T @ target / max(source.shape[0], 1))
    total = singular.square().sum().clamp_min(1e-12)
    return [{"dimension": int(d), "captured_fraction": float(singular[:d].square().sum() / total),
             "approximation_residual": float(singular[d:].square().sum() / total)}
            for d in dimensions if d > 0]
