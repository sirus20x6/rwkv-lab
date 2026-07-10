from __future__ import annotations

import sys

import torch
import torch.nn as nn

from rwkv_lab.rlvr import NumericAnswerVerifier, policy_loss
from rwkv_lab.rlvr_train import (RLVRTask, Rollout, VERIFY_RESPONSE_SCHEMA,
                                 arithmetic_curriculum, optimize_rollouts,
                                 promotion_decision, response_log_probs, sample_response,
                                 split_task_pool, verify_rollouts)


class ToyLM(nn.Module):
    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_arithmetic_curriculum_is_deterministic_and_disjoint():
    a = arithmetic_curriculum(20, seed=7, split="train", difficulty=3)
    b = arithmetic_curriculum(20, seed=7, split="train", difficulty=3)
    e = arithmetic_curriculum(5, seed=8, split="eval", difficulty=3)
    assert a == b
    train, evaluate = split_task_pool(a + e)
    assert len(train) == 20 and len(evaluate) == 5
    assert not ({t.id for t in train} & {t.id for t in evaluate})


def test_numeric_verifier_uses_boxed_or_final_answer():
    verify = NumericAnswerVerifier(7.5)
    assert verify("work: 3 + 4.5 = 7.5") == 1
    assert verify(r"wrong intermediate 8, final \\boxed{15/2}") == 1
    assert verify("7.4") == 0


def test_external_verifier_batch_contract():
    task = RLVRTask("code-1", "return ok", {"kind": "external"})
    rollouts = [Rollout("code-1:0", task, [2], [3], "ok"),
                Rollout("code-1:1", task, [2], [4], "no")]
    script = (
        "import json,sys; p=json.load(sys.stdin); "
        f"print(json.dumps({{'schema':'{VERIFY_RESPONSE_SCHEMA}',"
        "'rewards':[{'id':x['id'],'reward':float(x['response']=='ok')} for x in p['items']]}))"
    )
    rewards, details = verify_rollouts(rollouts, external_command=[sys.executable, "-c", script])
    assert rewards.tolist() == [1.0, 0.0]
    assert all(d["source"] == "external" for d in details)


def test_response_scoring_and_rlvr_update_are_differentiable():
    torch.manual_seed(3)
    model = ToyLM()
    task = RLVRTask("t", "prompt", {"kind": "numeric", "expected": 1})
    rollouts = [Rollout(f"t:{i}", task, [2, 3], [4 + i, 1], str(i % 2)) for i in range(4)]
    logp, mask = response_log_probs(model, rollouts)
    assert logp.shape == mask.shape == (4, 2)
    logp.sum().backward()
    assert model.head.weight.grad is not None

    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.head.weight.detach().clone()
    stats = optimize_rollouts(model, optimizer, rollouts, torch.tensor([1., 0., 1., 0.]),
                              group_size=2, algorithm="gspo", epochs=1,
                              clip_low=.2, clip_high=.2, kl_coef=.01, grad_clip=1.0)
    assert stats["update_applied"] == 1
    assert not torch.equal(before, model.head.weight)


def test_reference_kl_estimator_is_non_negative():
    logp = torch.tensor([[-1.0, -0.5], [-0.2, -1.2]], requires_grad=True)
    old = logp.detach().clone()
    ref = logp.detach() + torch.tensor([[0.4, -0.2], [0.1, -0.3]])
    out = policy_loss(logp, old, torch.tensor([1., 0.]), torch.tensor([0, 0]),
                      torch.ones_like(logp), reference_logp=ref, kl_coef=.1)
    assert out.approx_kl >= 0


def test_dr_grpo_uses_explicit_constant_token_normalizer():
    old = torch.zeros(2, 2)
    logp = torch.tensor([[.05, .05], [-.05, -.05]], requires_grad=True)
    kwargs = dict(rewards=torch.tensor([1., 0.]), group_ids=torch.tensor([0, 0]),
                  mask=torch.ones_like(logp), algorithm="dr_grpo")
    short = policy_loss(logp, old, token_normalizer=2, **kwargs)
    fixed = policy_loss(logp, old, token_normalizer=4, **kwargs)
    assert torch.allclose(fixed.loss, short.loss / 2)


def test_greedy_sampler_includes_stop_as_policy_token():
    class StopLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, ids):
            logits = torch.zeros(*ids.shape, 5) + self.anchor
            logits[..., 1] = 10
            return logits

    assert sample_response(StopLM(), [2], max_new=4, temperature=0, top_p=1,
                           top_k=0, stop_token=1, device="cpu", seed=0) == [1]


def test_promotion_requires_update_signal_and_heldout_gain():
    args = dict(minimum_delta=.05, candidate_checkpoint="candidate.pt",
                rollback_checkpoint="parent.pt")
    passed = promotion_decision({"reward": .2}, {"reward": .3}, updates_applied=2, **args)
    no_signal = promotion_decision({"reward": .2}, {"reward": .3}, updates_applied=0, **args)
    regressed = promotion_decision({"reward": .2}, {"reward": .1}, updates_applied=2, **args)
    assert passed["eligible"] and not no_signal["eligible"] and not regressed["eligible"]
