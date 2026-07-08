"""Coconut — Chain of Continuous Thought (arXiv:2412.06769, Hao et al., Meta).

Reason in a CONTINUOUS latent space instead of decoding tokens. Between a <bot> (begin-of-thought)
and <eot> (end-of-thought) marker, the model's input at each step is the PREVIOUS step's last
hidden state h_t — not a token embedding. A "continuous thought" is never decoded; it stays a
vector, so a chain of them explores reasoning without committing to discrete tokens (a single
latent thought can encode a superposition of next steps — breadth-first search in latent space).

Training is a CURRICULUM (Coconut's key recipe): stage k replaces the first k language reasoning
steps with k·c continuous thoughts; the REMAINING reasoning steps + the answer are supervised with
ordinary next-token cross-entropy. The thoughts get no direct target — gradient reaches them by
back-prop through the later supervised tokens (h_t feeds position t+1). Stage 0 == plain CoT.

This module is model-agnostic: it drives any `model_forward(inputs_embeds) -> (hidden, logits)`
(for us, the Qwen->RWKV CausalLM: hidden = last layer output, logits = lm_head(hidden)). For a
recurrent RWKV core the latent feedback is especially natural — h_t simply becomes the next input
to the recurrence, no KV-cache juggling.

Cost note: the reference `coconut_forward` fills the n continuous thoughts with n sequential
forward passes over a growing prefix (each thought reads the previous thought's hidden). Correct
but O(n) passes; a production path would reuse a KV-cache / recurrent state to fill one step each.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

ModelForward = Callable[[torch.Tensor], tuple]   # inputs_embeds [B,T,C] -> (hidden [B,T,C], logits [B,T,V])


def coconut_forward(inputs_embeds: torch.Tensor, latent_positions: Sequence[int],
                    model_forward: ModelForward):
    """Fill continuous-thought positions by feeding back hidden states, then run the full sequence.

    inputs_embeds [B,T,C]: token embeddings; each latent position holds a placeholder that is
        OVERWRITTEN by the last hidden state of the position immediately before it.
    latent_positions: sorted, ascending int indices (shared across the batch — bucket examples so
        their thoughts align, or use batch size 1). Each must be >= 1 (needs a preceding position).
    Returns (hidden [B,T,C], logits [B,T,V], filled_embeds [B,T,C]). Gradient flows through the
    fed-back hidden states (BPTT into the thoughts), so this is graph-preserving for training."""
    E = inputs_embeds
    for p in latent_positions:
        if p < 1:
            raise ValueError(f"continuous thought at position {p} has no preceding position to read")
        hidden_prefix, _ = model_forward(E[:, :p])          # hidden for positions 0..p-1
        h_prev = hidden_prefix[:, p - 1]                     # last hidden before the thought -> its input
        E = torch.cat([E[:, :p], h_prev.unsqueeze(1), E[:, p + 1:]], dim=1)   # overwrite pos p, keep graph
    hidden, logits = model_forward(E)
    return hidden, logits, E


def coconut_loss(inputs_embeds: torch.Tensor, latent_positions: Sequence[int],
                 labels: torch.Tensor, loss_mask: torch.Tensor, model_forward: ModelForward):
    """Next-token CE over supervised positions after the continuous thoughts are filled.
    labels [B,T]: gold token that logits[:, i] should predict (caller aligns the shift).
    loss_mask [B,T] in {0,1}: 1 on the supervised tokens (remaining reasoning steps + answer);
        0 on the question, the <bot>/<eot> markers, and the latent thoughts (no token target)."""
    _, logits, _ = coconut_forward(inputs_embeds, latent_positions, model_forward)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none").view_as(loss_mask)
    return (ce * loss_mask).sum() / loss_mask.sum().clamp_min(1)


def build_coconut_example(question: List[int], steps: List[List[int]], answer: List[int],
                          stage: int, c: int = 1, bot_id: Optional[int] = None,
                          eot_id: Optional[int] = None, pad_id: int = 0) -> dict:
    """Build ONE curriculum-`stage` training example.

    question: question token ids. steps: per reasoning-step token id lists. answer: answer token ids.
    stage k: replace the first min(k, len(steps)) language steps with c thoughts each. c = thoughts
        per replaced step (Coconut uses c=1 or 2). stage 0 -> no thoughts (plain CoT).
    Returns {input_ids, latent_positions, labels, loss_mask}; latent positions hold `pad_id`
    placeholders whose embeddings are overwritten at forward time. Supervision covers the remaining
    reasoning steps + the answer only (the thoughts and question are masked out)."""
    n_replace = min(max(stage, 0), len(steps))
    n_latent = c * n_replace
    remaining = steps[n_replace:]

    ids: List[int] = list(question)
    if bot_id is not None:
        ids.append(bot_id)
    latent_positions: List[int] = []
    for _ in range(n_latent):
        latent_positions.append(len(ids))
        ids.append(pad_id)                                   # placeholder — overwritten by a hidden state
    if eot_id is not None:
        ids.append(eot_id)
    sup_start = len(ids)                                     # first supervised token index
    for st in remaining:
        ids += list(st)
    ids += list(answer)

    labels = ids[1:] + [pad_id]                              # logits[i] predicts ids[i+1]
    loss_mask = [0] * len(ids)
    for i in range(max(sup_start - 1, 0), len(ids) - 1):     # supervise tokens from sup_start onward
        loss_mask[i] = 1
    return {"input_ids": ids, "latent_positions": latent_positions,
            "labels": labels, "loss_mask": loss_mask}


@torch.no_grad()
def coconut_generate(prompt_embeds: torch.Tensor, n_thoughts: int, max_new_tokens: int,
                     model_forward: ModelForward, embed_token: Callable[[torch.Tensor], torch.Tensor],
                     eot_embed: Optional[torch.Tensor] = None, eos_id: Optional[int] = None):
    """Inference: run the prompt, think `n_thoughts` continuous steps (feed hidden back), optionally
    append an <eot> embedding, then greedily decode up to `max_new_tokens`. embed_token(ids[B])->
    [B,C] maps a decoded token back to an input embedding. Returns the generated token ids [B, <=N]."""
    E = prompt_embeds
    for _ in range(n_thoughts):                              # continuous thinking: no decoding
        hidden, _ = model_forward(E)
        E = torch.cat([E, hidden[:, -1:].detach()], dim=1)
    if eot_embed is not None:
        B = E.shape[0]
        E = torch.cat([E, eot_embed.view(1, 1, -1).expand(B, 1, -1)], dim=1)
    out: List[torch.Tensor] = []
    for _ in range(max_new_tokens):                          # decode answer tokens normally
        _, logits = model_forward(E)
        nxt = logits[:, -1].argmax(-1)                       # [B]
        out.append(nxt)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
        E = torch.cat([E, embed_token(nxt).unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1) if out else prompt_embeds.new_zeros((prompt_embeds.shape[0], 0), dtype=torch.long)
