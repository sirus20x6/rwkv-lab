"""Fast invariants for capabilities sourced from the RWKV Discord research review."""
from __future__ import annotations

import copy

import torch


def test_balance_state_is_off_by_default_and_changes_recurrence(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet
    base = RWKV8TimeMixDeltaNet(8, num_heads=2, head_size=4, depth_layer_id=0,
                                depth_n_layer=2, out_correct=False)
    balanced = RWKV8TimeMixDeltaNet(8, num_heads=2, head_size=4, depth_layer_id=0,
                                    depth_n_layer=2, out_correct=False, balance_state=True)
    with torch.no_grad():
        base.output.weight.normal_(std=.1)
    balanced.load_state_dict(copy.deepcopy(base.state_dict()))
    x = torch.randn(2, 4, 8)
    ordinary = base(x, return_state=True)[0]
    changed = balanced(x, return_state=True)[0]
    assert not base.balance_state and balanced.balance_state
    assert torch.isfinite(changed).all() and not torch.allclose(ordinary, changed)


def test_balance_state_declarative_lm_lever_is_an_action_flag(tmp_path):
    from rwkv_lab.config import _lm_command
    command = _lm_command(["--data", "tokens.bin"], None, str(tmp_path),
                          {"d_model": 8, "n_layers": 2, "head_size": 4},
                          {"steps": 1}, {"balance_state": True}, 0,
                          str(tmp_path / "model.pt"))
    assert command.count("--balance-state") == 1
    assert command[-1] == "--balance-state"


def test_state_adapter_expands_without_cross_request_aliasing(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    from rwkv_lab.rwkv_pretrain import RWKV7Small
    from rwkv_lab.state_tuning import install_state_adapter
    model = RWKV7Small(32, 8, 2, 4, {})
    adapter = install_state_adapter(model)
    state = adapter.expanded(3)
    assert state[0]["wkv"].shape == (3, 2, 4, 4)
    logits, _ = model.forward_recurrent(torch.tensor([[1, 2], [3, 4], [5, 6]]))
    logits.float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("state_adapter."))


def test_paged_state_fork_stack_and_split_are_isolated():
    from rwkv_lab.online_memory import OnlineMemoryState
    from rwkv_lab.recurrent_serving import (PagedRecurrentStatePool,
                                            split_recurrent_state,
                                            stack_recurrent_states)
    pool = PagedRecurrentStatePool(max_entries=3, pin_memory=False)
    state = {"blocks": [{"wkv": torch.randn(1, 2, 3, 3), "shift": torch.randn(1, 1, 6)}],
             "online_memory": OnlineMemoryState(torch.randn(1, 3, 3), torch.zeros(1, 3, 3))}
    pool.put("prefix", state); pool.fork("prefix", "request")
    child = pool.get("request"); child["blocks"][0]["wkv"].zero_()
    assert pool.get("prefix")["blocks"][0]["wkv"].abs().sum() > 0
    batch = stack_recurrent_states([pool.get("prefix"), pool.get("request")])
    rows = split_recurrent_state(batch, 2)
    assert len(rows) == 2 and rows[0]["blocks"][0]["wkv"].shape[0] == 1
    assert rows[1]["online_memory"].memory.shape[0] == 1


def test_triangular_inversion_oracle_and_qualification():
    from rwkv_lab.triangular_delta import (qualify_triangular_backend,
                                           stable_triangular_inverse)
    matrix = torch.eye(5).repeat(2, 1, 1)
    matrix[:, 1:, :-1] += torch.tril(torch.randn(2, 4, 4) * .03)
    direct = stable_triangular_inverse(matrix)
    neumann = stable_triangular_inverse(matrix, method="neumann")
    assert torch.allclose(direct, neumann, atol=1e-5)
    report = qualify_triangular_backend(matrix, lambda value: stable_triangular_inverse(value),
                                        repeats=1, minimum_speedup=0)
    assert report["parity_passed"] and report["adopted"]


def test_sleep_consolidation_is_bounded_and_explicit():
    from rwkv_lab.online_memory import OnlineAssociativeMemory, SleepConsolidator
    memory = OnlineAssociativeMemory(8, d_memory=4)
    hidden = torch.randn(2, 5, 8)
    state = memory.initial_state(2)
    unchanged = memory.sleep_consolidate(hidden, state, passes=0)
    assert unchanged.steps == 0
    scheduler = SleepConsolidator(memory, interval=5, passes=2, max_context=5)
    scheduler.observe(hidden)
    consolidated = scheduler.consolidate(state)
    assert consolidated.steps == 10 and not scheduler.due()


def test_reasoning_cache_enforces_token_budget_and_materializes_pairs():
    from rwkv_lab.reasoning_cache import reasoning_cache_training_pairs, run_reasoning_cache
    generate = lambda prompt, cache, budget: (f"attempt:{cache}", min(3, budget))
    summarize = lambda response, cache, budget: (response[-8:], min(2, budget))
    result = run_reasoning_cache("problem", generate=generate, summarize=summarize,
                                 max_iterations=10, max_total_tokens=11)
    assert result.total_tokens <= 11 and result.stop_reason == "token_budget"
    assert reasoning_cache_training_pairs("problem", result)


def test_triplet_diffusion_and_hils_reference_paths_backpropagate():
    from rwkv_lab.diffusion_rwkv import TripletBlockDiffusionHead
    from rwkv_lab.hils_attention import HiLSAttention
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    ids = torch.randint(0, 32, (2, 7))
    diffusion = TripletBlockDiffusionHead(8, 32, heads=2)
    loss = diffusion.loss(hidden, ids, torch.full((2,), 127))
    loss.backward(retain_graph=True)
    assert torch.isfinite(loss) and diffusion.output.weight.grad is not None
    hils = HiLSAttention(8, heads=2, chunk_size=3, top_chunks=2)
    output = hils(hidden)
    output.square().mean().backward()
    assert output.shape == hidden.shape and hils.output.weight.grad is not None


def test_decoder_matrix_is_deterministic_and_tracks_state_drift():
    from rwkv_lab.decoding_eval import DecoderConfig, evaluate_decoders
    def step(token, state):
        state = torch.tensor([0.0]) if state is None else state
        state = state + token
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0]) + state * .01
        return logits, state
    configs = [DecoderConfig("greedy", "greedy"),
               DecoderConfig("typical", "typical", typical_p=.8)]
    first = evaluate_decoders([[1, 2]], configs, step=step, max_new=5, seed=7)
    second = evaluate_decoders([[1, 2]], configs, step=step, max_new=5, seed=7)
    assert first["typical"]["samples"] == second["typical"]["samples"]
    assert first["greedy"]["state_divergence"] == 0
    assert first["typical"]["state_divergence"] >= 0


def test_state_offset_applies_each_token_and_keeps_base_frozen(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    from rwkv_lab.rwkv_pretrain import RWKV7Small
    from rwkv_lab.state_tuning import install_state_offset_adapter
    model = RWKV7Small(16, 8, 2, 4, {})
    adapter = install_state_offset_adapter(model)
    assert "state offsets" in model.packing_incompatibility()
    with torch.no_grad():
        adapter.wkv[0].fill_(.01)
    logits, state = model.forward_recurrent(torch.tensor([[1, 2, 3]]))
    logits.float().sum().backward()
    assert logits.shape == (1, 3, 16) and state["offset_step"] == 3
    assert adapter.wkv[0].grad is not None
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if not n.startswith("state_offset_adapter."))


def test_state_offset_is_a_declarative_lm_arm(tmp_path):
    from rwkv_lab.config import _lm_command
    command = _lm_command(["--data", "tokens.bin"], None, str(tmp_path),
                          {"d_model": 8, "n_layers": 2, "head_size": 4},
                          {"steps": 1}, {"state_offset": True,
                                         "state_offset_interval": 1}, 0,
                          str(tmp_path / "model.pt"))
    assert command[-4:] == ["--state-offset", "1", "--state-offset-interval", "1"]


def test_routed_state_bank_reports_entropy_and_isolation():
    from rwkv_lab.state_bank import RoutedStateBank
    example = [{"wkv": torch.zeros(1, 2, 3, 3), "shift": torch.zeros(1, 1, 6)}]
    bank = RoutedStateBank(example, query_dim=4, slots=3, hyper_rank=2)
    seeded = [{"wkv": torch.ones(1, 2, 3, 3), "shift": torch.ones(1, 1, 6)}]
    bank.seed_slot(1, seeded)
    state, stats = bank.route(torch.randn(2, 4))
    assert state[0]["wkv"].shape == (2, 2, 3, 3)
    assert stats["weights"].shape == (2, 3)
    assert torch.isfinite(stats["entropy"]).all() and torch.all(stats["collapse"] < 1)


def test_byte_aware_embedding_is_exact_noop_and_trains():
    from rwkv_lab.tokenizer_experiments import ByteAwareEmbedding, superword_training_command
    base = torch.nn.Embedding(8, 6)
    wrapped = ByteAwareEmbedding(base, {1: b"hello", 2: "λ".encode()}, max_bytes=8, byte_dim=3)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(wrapped(ids), base(ids))
    with torch.no_grad(): wrapped.length.weight[5].fill_(.1); wrapped.project.weight.fill_(.1)
    assert not torch.equal(wrapped(ids), base(ids))
    command = superword_training_command(implementation="superbpe", corpus="data.txt",
                                         output="tok", vocab_size=4096)
    assert command[:4] == ["ztok", "train", "--kind", "superbpe"]


def test_typed_decoding_policy_is_allowlisted_and_bounded():
    from rwkv_lab.decoding_policy import DecodingPolicyMachine, SamplingPolicy
    machine = DecodingPolicyMachine(
        {"normal": SamplingPolicy("normal"),
         "json": SamplingPolicy("json", temperature=0, grammar="json")},
        default="normal", enter_tokens={10: "json"}, exit_tokens=(11,), max_depth=2)
    assert machine.consume(10).name == "json"
    assert machine.consume(11).name == "normal"
    with __import__("pytest").raises(RuntimeError): machine.consume(11)


def test_guarded_adapter_consolidation_requires_promotion(tmp_path):
    from rwkv_lab.adapter_consolidation import GuardedAdapterConsolidator
    context = tmp_path / "day.state"; context.write_bytes(b"immutable state")
    guard = GuardedAdapterConsolidator(tmp_path / "sleep", minimum_improvement=.1)
    snapshot = guard.snapshot({"day.state": context})
    def train(_snapshot, out, budget):
        assert budget == 100_000
        candidate = out / "adapter.bin"; candidate.write_bytes(b"candidate"); return candidate
    receipt = guard.run(snapshot, train=train, evaluate=lambda _: {"reward": 1.2, "safety": .9},
                        baseline_metrics={"reward": 1.0, "safety": .9},
                        primary_metric="reward", regression_metrics=("safety",))
    assert receipt["eligible"] and not receipt["promoted"]
    with __import__("pytest").raises(PermissionError):
        guard.promote(receipt["receipt"], tmp_path / "promoted.bin", approved=False)
    promoted = guard.promote(receipt["receipt"], tmp_path / "promoted.bin", approved=True)
    assert promoted["promoted"]
