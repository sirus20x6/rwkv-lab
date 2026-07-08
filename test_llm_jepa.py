"""CPU test for LLM-JEPA (paired-view embedding objective).

Run: python test_llm_jepa.py
"""
import torch
import torch.nn as nn
from llm_jepa import LLMJEPA, cosine_jepa_loss, last_token_embedding

C, B, T = 32, 4, 10


def _stub_body():
    """A tiny stand-in for the LLM body: inputs_embeds [B,L,C] -> hidden [B,L,C]. Includes a
    CAUSAL prefix-mean so each position (incl. the appended [PRED] token) depends on the tokens
    before it — mimicking attention, so the [PRED] token can read the Text (as in a real LLM)."""
    lin = nn.Linear(C, C)
    def body(inputs_embeds, attention_mask=None):
        L = inputs_embeds.shape[1]
        denom = torch.arange(1, L + 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, -1, 1)
        ctx = inputs_embeds.cumsum(dim=1) / denom              # causal prefix mean (crude attention)
        return torch.tanh(lin(inputs_embeds) + ctx)
    return body, lin


def test_cosine_loss():
    x = torch.randn(B, C)
    assert float(cosine_jepa_loss(x, x)) < 1e-5, "cos loss of x with itself should be ~0"
    assert abs(float(cosine_jepa_loss(x, -x)) - 2.0) < 1e-5, "cos loss of x,-x should be ~2"
    print("[jepa] cosine loss: 0 for aligned, 2 for opposite — OK")


def test_last_token_mask():
    h = torch.arange(B * T * C, dtype=torch.float32).view(B, T, C)
    m = torch.ones(B, T); m[0, 5:] = 0                          # row 0 last real token is index 4
    e = last_token_embedding(h, m)
    assert torch.equal(e[0], h[0, 4]) and torch.equal(e[1], h[1, T - 1])
    print("[jepa] last-token embedding respects the pad mask — OK")


def test_predictor_and_grads():
    torch.manual_seed(0)
    body, lin = _stub_body()
    text = torch.randn(B, T, C, requires_grad=True)
    code_hidden = torch.randn(B, T, C, requires_grad=True)
    # k>0: [PRED] tokens appended, run through body, last hidden = prediction
    jepa = LLMJEPA(C, k=2)
    pred = jepa.predict(text, body)
    assert pred.shape == (B, C)
    l = jepa.loss(text, code_hidden, body)
    assert torch.isfinite(l) and 0.0 <= float(l) <= 2.0
    l.backward()
    assert jepa.pred_tokens.grad.abs().sum() > 0, "no grad to [PRED] tokens"
    assert lin.weight.grad.abs().sum() > 0, "no grad to the model body (shared encoder)"
    assert text.grad.abs().sum() > 0 and code_hidden.grad.abs().sum() > 0, "no grad to views (no stop-grad)"
    print(f"[jepa] k=2 predictor + cosine loss {float(l):.3f}; grads -> PRED tokens, body, BOTH views — OK")


def test_k0_identity():
    body, _ = _stub_body()
    jepa = LLMJEPA(C, k=0)
    assert jepa.pred_tokens is None
    text = torch.randn(B, T, C)
    # k=0 => identity predictor: just the pooled encoded Text
    p = jepa.predict(text, body)
    assert torch.equal(p, last_token_embedding(body(text), None))
    print("[jepa] k=0 predictor is identity (Pred(x)=x) — OK")


if __name__ == "__main__":
    test_cosine_loss()
    test_last_token_mask()
    test_predictor_and_grads()
    test_k0_identity()
    print("\nall LLM-JEPA tests passed")
