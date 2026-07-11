from __future__ import annotations

import torch

from rwkv_lab.generate import sample_with_stats
from rwkv_lab.online_memory import install_compiled_online_memory, install_online_memory
from rwkv_lab.production_kernels import qualify_recurrent_generation
from rwkv_lab.rwkv_pretrain import RWKV7Small
from rwkv_lab.speculative import (qualify_speculative_greedy,
                                  EAGLE3DraftHead,
                                  speculative_greedy_decode,
                                  verify_greedy_draft_batched)


def _target_next(context: list[int]) -> int:
    return (sum(context) + 1) % 17


def _target_verify(prefix: list[int], draft: list[int]) -> list[int]:
    context, predictions = list(prefix), []
    for token in draft:
        predictions.append(_target_next(context))
        context.append(token)
    predictions.append(_target_next(context))
    return predictions


def _draft(context: list[int], width: int) -> list[int]:
    # Two good tokens followed by a deliberate rejection exercises both accept paths.
    out, work = [], list(context)
    for index in range(width):
        token = _target_next(work) if index < 2 else (_target_next(work) + 1) % 17
        out.append(token)
        work.append(token)
    return out


def test_batched_speculative_verification_is_exact_and_counted():
    verified, accepted = verify_greedy_draft_batched(
        [1, 2], [4, 8, 0], target_verify=_target_verify)
    assert verified == [4, 8, 16]
    assert accepted == 2
    tokens, stats = speculative_greedy_decode(
        [1, 2], max_new=12, draft_steps=4, draft_propose=_draft,
        target_verify=_target_verify)
    context, expected = [1, 2], []
    for _ in range(12):
        token = _target_next(context)
        expected.append(token)
        context.append(token)
    assert tokens == expected
    assert stats.target_calls < stats.generated_tokens
    assert 0 < stats.acceptance_rate < 1
    report = qualify_speculative_greedy(
        [1, 2], max_new=12, draft_steps=4, target_next=_target_next,
        draft_propose=_draft, target_verify=_target_verify, minimum_speedup=0.0)
    assert report["exact_tokens"] and report["adopted"]


def test_eagle_packed_head_loads_legacy_per_step_weights():
    head = EAGLE3DraftHead(4, 7, draft_steps=3)
    state = {key: value.clone() for key, value in head.state_dict().items()
             if key != "token_head.weight"}
    source = head.token_head.weight.detach().view(3, 7, 4)
    for index in range(3):
        state[f"token_heads.{index}.weight"] = source[index].clone()
    restored = EAGLE3DraftHead(4, 7, draft_steps=3)
    restored.load_state_dict(state)
    assert torch.equal(restored.token_head.weight, head.token_head.weight)


def test_online_memory_recurrent_chunks_match_full_prefix():
    torch.manual_seed(53)
    model = RWKV7Small(32, 8, 1, 4, {})
    memory = install_online_memory(model, d_memory=4, mode="atlas", atlas_window=3)
    with torch.no_grad():
        memory.out_proj.weight.normal_(std=0.05)
        memory.gate.bias.fill_(0.0)
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    expected = model(ids)
    first, state = model.forward_recurrent(ids[:, :2])
    second, state = model.forward_recurrent(ids[:, 2:], state)
    assert torch.allclose(torch.cat((first, second), dim=1), expected, atol=2e-5)
    assert state["online_memory"].steps == ids.shape[1]


def test_compiled_online_memory_uses_live_parameters_without_state_dict_duplication():
    torch.manual_seed(59)
    model = RWKV7Small(32, 8, 1, 4, {})
    memory = install_online_memory(model, d_memory=4)
    with torch.no_grad():
        memory.out_proj.weight.normal_(std=.05)
    ids = torch.tensor([[1, 2, 3]])
    expected = model(ids)
    keys = tuple(model.state_dict())
    install_compiled_online_memory(model, backend="eager")
    actual = model(ids)
    assert torch.allclose(actual, expected, atol=2e-5)
    assert tuple(model.state_dict()) == keys


class _ToyRecurrent(torch.nn.Module):
    def recurrent_incompatibility(self):
        return None

    @staticmethod
    def _logits(ids: torch.Tensor, prior: torch.Tensor | None = None):
        running = ids.cumsum(1)
        if prior is not None:
            running = running + prior[:, None]
        vocab = 19
        logits = torch.full((*ids.shape, vocab), -100.0, device=ids.device)
        target = (running + 1) % vocab
        logits.scatter_(2, target[..., None], 100.0)
        return logits, running[:, -1]

    def forward(self, ids):
        return self._logits(ids)[0]

    def forward_recurrent(self, ids, state=None):
        return self._logits(ids, state)


def test_generation_engine_and_recurrent_qualification_are_token_exact():
    model = _ToyRecurrent()
    prefix, pstats = sample_with_stats(
        model, [1, 2, 3], max_new=8, temperature=0, stop_at_sep=False,
        device="cpu", engine="prefix")
    recurrent, rstats = sample_with_stats(
        model, [1, 2, 3], max_new=8, temperature=0, stop_at_sep=False,
        device="cpu", engine="recurrent")
    assert recurrent == prefix
    assert pstats["engine"] == "prefix" and rstats["engine"] == "recurrent"
    report = qualify_recurrent_generation(
        model, [1, 2, 3], device="cpu", max_new=8, repeats=1)
    assert report["available"] and report["exact_tokens"]
