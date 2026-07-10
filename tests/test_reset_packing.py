import torch

from rwkv_lab.adapters import AdapterConfig, adapter_parameters, inject_lora
from rwkv_lab.posttrain_data import IGNORE_INDEX, TokenizedExample, TokenizedVariant
from rwkv_lab.posttrain_train import _qualify_reset_packing
from rwkv_lab.rwkv_pretrain import RWKV7Small


def test_reset_mask_packing_matches_individual_forward_and_gradients(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    torch.manual_seed(41)
    model = RWKV7Small(vocab=32, d=8, n_layers=2, head_size=4, loop_kw={}).float()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[5, 6, 7]])
    packed = torch.cat((first, second), dim=1)
    reset = torch.zeros_like(packed, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, first.shape[1]] = True

    expected_first = model(first)
    expected_second = model(second)
    actual = model(packed, reset_mask=reset)
    assert torch.allclose(actual[:, :first.shape[1]], expected_first, atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[:, first.shape[1]:], expected_second, atol=2e-5, rtol=2e-5)

    parameters = [model.blocks[0].att.receptance.weight, model.blocks[0].ffn.key.weight]
    separate_loss = expected_first.float().square().mean() + expected_second.float().square().mean()
    separate_gradients = torch.autograd.grad(separate_loss, parameters, retain_graph=True)
    packed_loss = (actual[:, :first.shape[1]].float().square().mean() +
                   actual[:, first.shape[1]:].float().square().mean())
    packed_gradients = torch.autograd.grad(packed_loss, parameters)
    for expected, candidate in zip(separate_gradients, packed_gradients):
        assert torch.allclose(candidate, expected, atol=3e-5, rtol=3e-5)


def test_sft_reset_multipack_passes_loss_and_adapter_gradient_gate(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    torch.manual_seed(43)
    model = RWKV7Small(vocab=32, d=8, n_layers=2, head_size=4, loop_kw={}).float()
    inject_lora(model, AdapterConfig("pack", rank=2, alpha=2, targets=("output",)))
    first = TokenizedVariant((1, 2, 3, 4), (IGNORE_INDEX, IGNORE_INDEX, 3, 4), ("",) * 4)
    second = TokenizedVariant((5, 6, 7), (IGNORE_INDEX, 6, 7), ("",) * 3)
    rows = [TokenizedExample("a", "sft", "train", {"sft": first}, {}),
            TokenizedExample("b", "sft", "train", {"sft": second}, {})]
    report = _qualify_reset_packing(model, None, rows, "sft", "pack", "cpu", 0.1, 1.0,
                                    list(adapter_parameters(model)), tolerance=5e-4)
    assert report["passed"], report
