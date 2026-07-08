"""Validate the fast rosa_sam kernel against the golden brute-force ROSA reference.

Run: pytest tests/test_rosa_reference.py
"""
import numpy as np
from rosa_reference import rosa_reference
from rosa_sam import sam_retrieve


def _agree(B, T, R, K, seed):
    rng = np.random.default_rng(seed)
    aq = rng.integers(0, K, (B, T, R))
    ak = rng.integers(0, K, (B, T, R))
    tau, mlen = sam_retrieve(aq, ak, K, return_mlen=True)
    tau_ref, mlen_ref = rosa_reference(aq, ak)
    assert np.array_equal(mlen, mlen_ref), (
        f"matched-length mismatch vs brute force (K={K}): "
        f"first diff at {np.argwhere(mlen != mlen_ref)[:3].tolist()}")
    assert np.array_equal(tau, tau_ref), (
        f"tau (firstpos+1) mismatch vs brute force (K={K}): "
        f"first diff at {np.argwhere(tau != tau_ref)[:3].tolist()}")


def test_binary_canonical():
    # K=2 is the canonical 1-bit ROSA (the FPGA reference operates on a binary sequence)
    for seed in range(6):
        _agree(2, 40, 3, 2, seed)
    print("[rosa-ref] fast SAM == brute-force golden reference on 1-bit (K=2) streams — OK")


def test_larger_alphabet():
    for seed in range(4):
        _agree(2, 50, 4, 8, seed)
    _agree(1, 64, 1, 16, 99)
    print("[rosa-ref] fast SAM == brute-force reference on K=8/16 streams — OK")


def test_self_matching_stream():
    # ROSA's canonical use: query and key are the SAME stream (match a sequence against its own past)
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2, (2, 48, 2))
    tau, mlen = sam_retrieve(a, a, 2, return_mlen=True)
    tau_ref, mlen_ref = rosa_reference(a, a)
    assert np.array_equal(mlen, mlen_ref) and np.array_equal(tau, tau_ref)
    print("[rosa-ref] self-matching (query==key) stream matches reference — OK")


def test_repetitive_structure():
    # a periodic stream stresses long suffix matches
    a = np.tile(np.array([0, 1, 1, 0, 1], np.int64), 12)[None, :, None]
    tau, mlen = sam_retrieve(a, a, 2, return_mlen=True)
    tau_ref, mlen_ref = rosa_reference(a, a)
    assert np.array_equal(mlen, mlen_ref) and np.array_equal(tau, tau_ref)
    assert mlen.max() >= 4, "expected long matches on a periodic stream"
    print(f"[rosa-ref] periodic stream: max matched length {int(mlen.max())}, matches reference — OK")


if __name__ == "__main__":
    test_binary_canonical()
    test_larger_alphabet()
    test_self_matching_stream()
    test_repetitive_structure()
    print("\nall ROSA golden-reference tests passed")
