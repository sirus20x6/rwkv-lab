import pytest
import torch

from rwkv_lab.megakernel import MegakernelBackend, triton_status
from rwkv_lab.rwkv_pretrain import RWKV7Small

pytestmark = pytest.mark.gpu


@torch.no_grad()
def _reference_greedy(model, prompt: torch.Tensor, max_new: int) -> torch.Tensor:
    logits, state = model.forward_recurrent(prompt)
    output = []
    for position in range(max_new):
        token = logits[:, -1].float().argmax(dim=-1, keepdim=True)
        output.append(token)
        if position + 1 < max_new:
            logits, state = model.forward_recurrent(token, state)
    return (torch.cat(output, dim=1) if output
            else prompt.new_empty((prompt.shape[0], 0)))


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_fixed_budget_greedy_graph_is_exact_cached_and_eos_safe(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    torch.manual_seed(173)
    model = RWKV7Small(64, 16, 1, 8, {}).to("cuda", torch.bfloat16).eval()
    prompt = torch.tensor([[2, 3, 4]], device="cuda")
    expected = _reference_greedy(model, prompt, 5)

    backend = MegakernelBackend(model, device="cuda", compile_mode="default")
    actual = backend.generate_greedy(prompt, max_new=5).clone()
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    assert backend.receipt()["cached_greedy_plans"] == 1

    # A cached fixed-budget plan submits the complete decode segment with one
    # host graph replay; prefill remains its own independently cached graph.
    plan = next(iter(backend.greedy_plans.values()))
    real_graph = plan.graph

    class ReplayCounter:
        calls = 0

        def replay(self):
            self.calls += 1
            real_graph.replay()

    counter = ReplayCounter()
    plan.graph = counter
    repeated = backend.generate_greedy(prompt, max_new=5).clone()
    torch.cuda.synchronize()
    assert torch.equal(repeated, expected)
    assert counter.calls == 1
    assert backend.receipt()["cached_greedy_plans"] == 1

    # Exact budgets are separate cache entries; generation neither pads nor
    # advances the recurrent state through a larger bucket.
    one = backend.generate_greedy(prompt, max_new=1).clone()
    zero = backend.generate_greedy(prompt, max_new=0).clone()
    torch.cuda.synchronize()
    assert torch.equal(one, expected[:, :1])
    assert zero.shape == (1, 0)
    assert backend.receipt()["cached_greedy_plans"] == 2

    # Once a row emits its stop token, all later fixed-budget slots remain the
    # stop token on device. Existing callers can truncate at the first token.
    stop = int(expected[0, 0])
    stopped = backend.generate_greedy(
        prompt, max_new=5, stop_token_id=stop).clone()
    torch.cuda.synchronize()
    assert torch.equal(stopped, torch.full_like(stopped, stop))
    assert backend.receipt()["cached_greedy_plans"] == 3
