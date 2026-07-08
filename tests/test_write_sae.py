"""CPU test for WriteSAE (sparse autoencoder for recurrent state, arXiv:2605.12770).

Run: pytest tests/test_write_sae.py
"""
import torch
from write_sae import WriteSAE

DV, DK, B = 16, 16, 8


def test_shapes_and_topk_sparsity():
    sae = WriteSAE(DV, DK, n_feat=64, k=5)
    S = torch.randn(B, DV, DK)
    a, alls = sae.encode(S)
    assert a.shape == (B, 64) and alls.shape == (B, 64)
    assert (a != 0).sum(-1).max() <= 5, "TopK produced more than k nonzeros"
    assert sae.decode(a).shape == (B, DV, DK)
    print("[write-sae] encode/decode shapes + hard TopK sparsity — OK")


def test_matched_filter_reads_own_write():
    # bilinear encoder: an atom's own rank-1 outer product should activate that atom strongly
    sae = WriteSAE(DV, DK, n_feat=32, k=4, atom_rank=1)
    i = 7
    S = sae.atom(i).unsqueeze(0)                    # state = exactly atom i's write shape
    _, scores = sae.encode(S)
    assert scores[0].argmax().item() == i, "matched-filter encoder didn't read the atom it writes"
    print("[write-sae] matched-filter encoder: read direction == write direction — OK")


def test_reconstruction_learns():
    torch.manual_seed(0)
    sae = WriteSAE(DV, DK, n_feat=48, k=8, k_aux=0)
    # states drawn from a small set of rank-1 writes -> a sparse code should reconstruct them
    base = [torch.randn(DV, 1) @ torch.randn(1, DK) for _ in range(6)]
    def batch():
        idx = torch.randint(0, 6, (B,))
        return torch.stack([base[j] for j in idx]) + 0.01 * torch.randn(B, DV, DK)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    sae.train()
    for _ in range(50):
        sae.update_mean(batch())
    l0 = None
    for step in range(400):
        S = batch()
        loss, info = sae.loss(S)
        opt.zero_grad(); loss.backward(); opt.step()
        if l0 is None: l0 = info["recon"]
    assert info["recon"] < 0.5 * l0, f"reconstruction did not improve ({l0:.3f} -> {info['recon']:.3f})"
    assert sae.V.grad.abs().sum() > 0 and sae.W.grad.abs().sum() > 0
    print(f"[write-sae] TopK recon MSE {l0:.3f} -> {info['recon']:.3f}; grad to V/W atoms — OK")


def test_matched_norm_substitution():
    sae = WriteSAE(DV, DK, n_feat=32, k=4)
    S_prev = torch.randn(DV, DK)
    native = torch.randn(DV, 1) @ torch.randn(1, DK) * 3.0     # a native write with some norm
    S_new = sae.causal_substitute(S_prev, native, atom_idx=3)
    injected = S_new - S_prev
    assert torch.allclose(injected.norm(), native.norm(), atol=1e-4), "substitution not norm-matched"
    print("[write-sae] matched-Frobenius-norm cache substitution (norm-preserving) — OK")


def test_rank2_atoms_for_rwkv7():
    # RWKV-7 writes rank-2; atom_rank=2 makes atoms match that
    sae = WriteSAE(DV, DK, n_feat=16, k=3, atom_rank=2)
    assert sae.V.shape == (16, 2, DV) and sae.W.shape == (16, 2, DK)
    S = torch.randn(B, DV, DK)
    loss, info = sae.train().__call__ if False else sae.loss(S)  # smoke
    assert torch.isfinite(loss)
    print("[write-sae] rank-2 atoms (RWKV-7 write shape) run — OK")


def test_dead_feature_tracking():
    sae = WriteSAE(DV, DK, n_feat=40, k=2, dead_after=3, k_aux=8)
    sae.train()
    for _ in range(6):
        sae.loss(torch.randn(B, DV, DK))
    assert sae.steps_since_fired.max() >= 3, "dead-feature counter not advancing"
    print("[write-sae] dead-feature tracking advances (AuxK revival active) — OK")


if __name__ == "__main__":
    test_shapes_and_topk_sparsity()
    test_matched_filter_reads_own_write()
    test_reconstruction_learns()
    test_matched_norm_substitution()
    test_rank2_atoms_for_rwkv7()
    test_dead_feature_tracking()
    print("\nall WriteSAE tests passed")
