import torch

from rwkv_lab.mla_module import MLAAttention


class _Cache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = key, value
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _rope(length, dim):
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    f = pos * inv
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


@torch.no_grad()
def test_compressed_cache_matches_full_attention():
    torch.manual_seed(4)
    for rope_position in ("first", "last"):
        m = MLAAttention(
            hidden_size=32, num_heads=4, qk_nope_head_dim=6,
            qk_rope_head_dim=4, v_head_dim=5, kv_lora_rank=7,
            has_qk_norm=True, rope_position=rope_position,
            num_kv_rope_heads=2, layer_idx=0,
        ).eval()
        x = torch.randn(2, 6, 32)
        cos, sin = _rope(6, 4)
        full, _ = m(x, position_embeddings=(cos, sin))

        cache = _Cache()
        prefill, _ = m(x[:, :5], position_embeddings=(cos[:5], sin[:5]),
                       past_key_value=cache, use_cache=True)
        decoded, _ = m(x[:, 5:], position_embeddings=(cos[5:], sin[5:]),
                       past_key_value=cache, use_cache=True)

        torch.testing.assert_close(prefill, full[:, :5], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(decoded, full[:, 5:], rtol=3e-5, atol=3e-5)
        # Cache is independent of v_head_dim and never stores expanded per-head V.
        assert cache.value_cache[0].shape == (2, 1, 6, 7)
