import torch

from rwkv_lab.looped_rwkv_rosa_engram_v3 import ModelConfig, LoopedRWKVLanguageModel


@torch.no_grad()
def test_per_loop_logits_are_opt_in():
    torch.manual_seed(2)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_prelude_layers=1, n_loop_layers=1,
        n_coda_layers=1, max_loops=2, num_depth_branches=1,
        loop_mixer_schedule=("rwkv",),
    )
    model = LoopedRWKVLanguageModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    fast, _, fast_diag = model(ids, forced_loops=2)
    diagnostic, _, diag = model(ids, forced_loops=2, return_loop_logits=True)
    torch.testing.assert_close(fast, diagnostic, rtol=0, atol=0)
    assert "per_loop_logits" not in fast_diag
    assert diag["per_loop_logits"].shape == (1, 3, 2, cfg.vocab_size)
