"""CPU test for Coconut (continuous-thought latent reasoning).

Run: pytest tests/test_coconut.py   (or python tests/test_coconut.py)
"""
import torch
import torch.nn as nn
from rwkv_lab.coconut import coconut_forward, coconut_loss, build_coconut_example, coconut_generate

C, V, B = 24, 40, 2


def _stub_model(seed=0):
    """A tiny causal LM stub: inputs_embeds [B,T,C] -> (hidden [B,T,C], logits [B,T,V]). Uses a
    causal prefix-mean so each position depends on the ones before it (mimics attention) — required
    for the continuous-thought feedback and its gradient path to be exercised."""
    torch.manual_seed(seed)
    lin = nn.Linear(C, C)
    head = nn.Linear(C, V)
    def fwd(embeds):
        L = embeds.shape[1]
        denom = torch.arange(1, L + 1, dtype=embeds.dtype).view(1, -1, 1)
        ctx = embeds.cumsum(1) / denom                      # causal prefix mean
        h = torch.tanh(lin(embeds) + ctx)
        return h, head(h)
    return fwd, lin, head


def test_feedback_fills_latent_with_prev_hidden():
    fwd, _, _ = _stub_model()
    T = 6
    E = torch.randn(B, T, C)
    latent = [3, 4]                                          # thoughts at positions 3,4
    _, _, filled = coconut_forward(E, latent, fwd)
    # position 3 must equal the last hidden of the prefix E[:, :3]
    h_pref, _ = fwd(E[:, :3])
    assert torch.allclose(filled[:, 3], h_pref[:, 2], atol=1e-5), "latent pos 3 != hidden[2]"
    # non-latent positions are untouched
    assert torch.equal(filled[:, 0], E[:, 0]) and torch.equal(filled[:, 5], E[:, 5])
    print("[coconut] latent position fed the previous step's hidden state — OK")


def test_gradient_flows_through_thoughts():
    fwd, lin, head = _stub_model()
    T = 7
    E = torch.randn(B, T, C, requires_grad=True)
    latent = [2, 3]
    labels = torch.randint(0, V, (B, T))
    loss_mask = torch.zeros(B, T); loss_mask[:, 4:] = 1      # supervise tokens after the thoughts
    loss = coconut_loss(E, latent, labels, loss_mask, fwd)
    assert torch.isfinite(loss)
    loss.backward()
    # gradient reaches the model AND the pre-thought token embeddings (BPTT through the thoughts)
    assert lin.weight.grad.abs().sum() > 0 and head.weight.grad.abs().sum() > 0
    assert E.grad[:, :2].abs().sum() > 0, "no gradient to tokens before the thoughts (BPTT broke)"
    print(f"[coconut] CE {float(loss):.3f}; grad reaches model + pre-thought tokens via BPTT — OK")


def test_stage0_equals_plain_ce():
    fwd, _, _ = _stub_model()
    T = 5
    E = torch.randn(B, T, C)
    labels = torch.randint(0, V, (B, T))
    loss_mask = torch.ones(B, T); loss_mask[:, -1] = 0
    # no latent positions == ordinary next-token CE on the same logits
    ln = coconut_loss(E, [], labels, loss_mask, fwd)
    _, logits = fwd(E)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none").view(B, T)
    ref = (ce * loss_mask).sum() / loss_mask.sum()
    assert torch.allclose(ln, ref, atol=1e-6), "stage-0 Coconut != plain masked CE"
    print("[coconut] stage 0 (no thoughts) == plain masked next-token CE — OK")


def test_curriculum_builder():
    q = [1, 2, 3]; steps = [[10, 11], [12], [13, 14]]; ans = [20, 21]
    ex = build_coconut_example(q, steps, ans, stage=2, c=2, bot_id=5, eot_id=6, pad_id=0)
    # stage 2, c=2 -> first 2 steps replaced by 4 thoughts; step 3 + answer remain (supervised)
    assert len(ex["latent_positions"]) == 4
    assert ex["input_ids"][4] == 0 and ex["latent_positions"][0] == 4  # after q(3)+bot(1)
    # supervised token count == remaining reasoning + answer tokens
    assert sum(ex["loss_mask"]) == len([13, 14, 20, 21])
    # stage 0 -> no thoughts, everything after the (absent) marker is CoT
    ex0 = build_coconut_example(q, steps, ans, stage=0, c=1)
    assert len(ex0["latent_positions"]) == 0
    print("[coconut] curriculum builder: stage/c -> thought count, loss mask, alignment — OK")


def test_generate_runs():
    fwd, _, _ = _stub_model()
    emb = nn.Embedding(V, C)
    prompt = torch.randn(B, 4, C)
    out = coconut_generate(prompt, n_thoughts=3, max_new_tokens=5, model_forward=fwd,
                           embed_token=lambda ids: emb(ids))
    assert out.shape == (B, 5) and out.dtype == torch.long
    print("[coconut] generate: n thoughts then greedy decode — OK")


if __name__ == "__main__":
    test_feedback_fills_latent_with_prev_hidden()
    test_gradient_flows_through_thoughts()
    test_stage0_equals_plain_ce()
    test_curriculum_builder()
    test_generate_runs()
    print("\nall Coconut tests passed")
