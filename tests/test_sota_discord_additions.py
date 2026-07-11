import time
import numpy as np
import torch

from rwkv_lab.kernel_candidates import qualify_kernel_candidate
from rwkv_lab.llm_jepa import ActionConditionedJEPA, action_conditioned_rank_diagnostic
from rwkv_lab.rosa_backends import ROSABackend, qualify_rosa_backend, register_rosa_backend, rosa_backends
from rwkv_lab.routing_free_moe import RoutingFreeMoE
from rwkv_lab.sparse_transfer import LowDimAttentionIndexer, dynamic_top_p_mask, sparse_attention, top_logit_distillation
from rwkv_lab.state_expansion import apply_statex_rwkv, block_diagonal_merged_state, statex_receipt, uniform_statex_plan
from rwkv_lab.supervised_memory import SupervisedMemoryTransition, rollout_drift, supervised_memory_loss


def test_statex_head_merge_preserves_diagonal_states_and_reports_expansion():
    state = torch.arange(16.0).reshape(4, 2, 2)
    merged = block_diagonal_merged_state(state, 2)
    assert merged.shape == (2, 4, 4)
    assert torch.equal(merged[0, :2, :2], state[0])
    assert torch.equal(merged[0, 2:, 2:], state[1])
    assert merged[0, :2, 2:].count_nonzero() == 0
    plan = uniform_statex_plan(12, 4, expanded_count=4)
    assert plan.expanded_layers == (0, 3, 6, 9)
    assert statex_receipt(plan, 64)["expansion_factor"] == 4


def test_statex_replaces_only_selected_rwkv_mixers():
    from rwkv_lab.rwkv_pretrain import RWKV7Small
    model = RWKV7Small(32, 16, 4, 8, {})
    untouched = model.blocks[1].att
    plan = apply_statex_rwkv(model, expanded_count=2)
    assert plan.expanded_layers == (0, 2)
    assert model.blocks[0].att.num_heads == 1 and model.blocks[0].att.head_size == 16
    assert model.blocks[1].att is untouched


def test_supervised_memory_is_parallel_and_tracks_rollout_drift():
    torch.manual_seed(1)
    updater = SupervisedMemoryTransition(6, 4)
    memory, inputs = torch.randn(2, 8, 6), torch.randn(2, 7, 4)
    result = supervised_memory_loss(updater, memory, inputs, detach_labels=True)
    result.total.backward()
    assert updater.net[0].weight.grad is not None
    assert rollout_drift(updater, memory[:, 0], inputs, memory).shape == (7,)


def test_routing_free_moe_self_activates_without_fixed_topk():
    torch.manual_seed(2)
    moe = RoutingFreeMoE(8, n_experts=3, hidden_dim=12, rank=4, threshold=0.1)
    x = torch.randn(2, 5, 8, requires_grad=True)
    y = moe(x)
    (y.square().mean() + 0.01 * moe.aux_loss).backward()
    assert y.shape == x.shape
    assert 0 <= moe.last_stats["activation_density"] <= 1
    assert moe.experts[0].down.weight.grad is not None


def test_dynamic_top_p_and_sparse_transfer_losses():
    scores = torch.tensor([[[[4.0, 1.0, 0.0]]]])
    assert dynamic_top_p_mask(scores, 0.8).sum() == 1
    indexer = LowDimAttentionIndexer(2, 4, 3)
    q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 6)
    output, support = sparse_attention(q, k, v, indexer, p=0.9)
    assert output.shape == (1, 2, 3, 6) and support.any(-1).all()
    assert top_logit_distillation(torch.randn(2, 20), torch.randn(2, 20)).item() >= 0


def test_external_kernel_candidate_requires_parity_and_speed():
    def reference(x):
        time.sleep(0.001)
        return x.square() + 1
    report = qualify_kernel_candidate(reference, lambda x: x.square() + 1,
                                      [(torch.randn(8),)], source="test-candidate", repeats=2,
                                      minimum_speedup=1.0)
    assert report["output_parity"] and report["gradient_parity"]
    assert report["deterministic"] and report["adopted"]


def test_rosa_backend_registration_is_parity_gated():
    from rwkv_lab.rosa_reference import rosa_reference
    name = "test-jax-adapter"
    if name not in rosa_backends():
        register_rosa_backend(ROSABackend(name, rosa_reference, device="jax-cpu"))
    q = np.array([[[0], [1], [0], [1], [0]]], dtype=np.int32)
    report = qualify_rosa_backend(name, q, q)
    assert report["exact"] and report["adopted"]


def test_action_conditioned_jepa_and_rank_curve():
    model = ActionConditionedJEPA(7, 3, 5)
    obs, actions, future = torch.randn(12, 7), torch.randn(12, 2, 3), torch.randn(12, 7)
    model.loss(obs, actions, future).backward()
    curve = action_conditioned_rank_diagnostic(obs, actions[:, 0], future, (1, 3, 5))
    assert model(obs, actions).shape == (12, 5)
    assert curve[0]["captured_fraction"] <= curve[-1]["captured_fraction"]
