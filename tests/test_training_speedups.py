import copy

import numpy as np
import pytest
import torch
from torch import nn

from rwkv_lab.distributed import configure_fsdp_prefetch, set_requires_gradient_sync
from rwkv_lab.training_speedups import (
    AsyncCPUBatchPrefetcher,
    context_batch_for_step,
    parse_context_curriculum,
)


def test_context_curriculum_preserves_tokens_per_step():
    stages = parse_context_curriculum(
        "0:128,0.25:256,0.75:512", max_seq_len=512
    )
    assert context_batch_for_step(
        stages, step=0, total_steps=100, max_seq_len=512, base_batch=8
    ) == (128, 32)
    assert context_batch_for_step(
        stages, step=25, total_steps=100, max_seq_len=512, base_batch=8
    ) == (256, 16)
    assert context_batch_for_step(
        stages, step=75, total_steps=100, max_seq_len=512, base_batch=8
    ) == (512, 8)


@pytest.mark.parametrize(
    "spec",
    ["0.1:128", "0:256,0.5:128", "0:128,0:256", "0:0", "bad"],
)
def test_context_curriculum_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        parse_context_curriculum(spec, max_seq_len=512)


def test_prefetch_rng_commits_only_consumed_batches():
    rng = np.random.default_rng(7)
    initial = copy.deepcopy(rng.bit_generator.state)

    def producer(local_rng):
        return torch.from_numpy(local_rng.integers(0, 1000, size=8))

    prefetch = AsyncCPUBatchPrefetcher(producer, rng, pin_memory=False)
    # Merely launching the worker does not move the checkpoint-visible RNG.
    assert rng.bit_generator.state == initial
    first = prefetch.next()
    committed_after_first = copy.deepcopy(rng.bit_generator.state)
    prefetch.close()

    reference = np.random.default_rng()
    reference.bit_generator.state = initial
    expected_first = torch.from_numpy(reference.integers(0, 1000, size=8))
    torch.testing.assert_close(first, expected_first)
    assert committed_after_first == reference.bit_generator.state

    # The unconsumed prefetched batch must be regenerated exactly after resume.
    resumed = np.random.default_rng()
    resumed.bit_generator.state = committed_after_first
    with AsyncCPUBatchPrefetcher(producer, resumed, pin_memory=False) as second:
        actual = second.next()
    expected = torch.from_numpy(reference.integers(0, 1000, size=8))
    torch.testing.assert_close(actual, expected)


class _FakeFSDPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_prefetch = []
        self.backward_prefetch = []

    def set_modules_to_forward_prefetch(self, modules):
        self.forward_prefetch = modules

    def set_modules_to_backward_prefetch(self, modules):
        self.backward_prefetch = modules


class _FakeFSDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeFSDPBlock() for _ in range(4)])
        self.forward_prefetch = []
        self.gradient_sync = None

    def set_modules_to_forward_prefetch(self, modules):
        self.forward_prefetch = modules

    def set_requires_gradient_sync(self, required, *, recurse):
        self.gradient_sync = (required, recurse)


def test_fsdp_prefetch_follows_forward_and_reverse_backward_order():
    model = _FakeFSDPModel()
    configure_fsdp_prefetch(model, depth=2)
    assert model.forward_prefetch == list(model.blocks[:2])
    assert model.blocks[0].forward_prefetch == list(model.blocks[1:3])
    assert model.blocks[3].backward_prefetch == [model.blocks[2], model.blocks[1]]
    set_requires_gradient_sync(model, False)
    assert model.gradient_sync == (False, True)
