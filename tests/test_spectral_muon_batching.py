import torch

from rwkv_lab.spectral_muon import SpectralMuon


def _parameters(seed):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(8, 6, generator=generator))
        for _ in range(3)
    ]


def test_batched_muon_matches_scalar_same_shape_updates():
    scalar = _parameters(1)
    batched = [torch.nn.Parameter(value.detach().clone()) for value in scalar]
    scalar_opt = SpectralMuon(
        [{"params": scalar, "use_muon": True, "lr": 0.02}],
        momentum=0.9, nesterov=True, ns_steps=3, scale=0.4, batched=False,
    )
    batched_opt = SpectralMuon(
        [{"params": batched, "use_muon": True, "lr": 0.02}],
        momentum=0.9, nesterov=True, ns_steps=3, scale=0.4, batched=True,
    )

    gradient_generator = torch.Generator().manual_seed(2)
    for _ in range(3):
        gradients = [
            torch.randn(value.shape, generator=gradient_generator)
            for value in scalar
        ]
        for left, right, gradient in zip(scalar, batched, gradients):
            left.grad = gradient.clone()
            right.grad = gradient.clone()
        scalar_opt.step()
        batched_opt.step()

    for left, right in zip(scalar, batched):
        torch.testing.assert_close(left, right, rtol=3e-3, atol=3e-4)
        torch.testing.assert_close(
            scalar_opt.state[left]["mom"],
            batched_opt.state[right]["mom"],
            rtol=0,
            atol=0,
        )


def test_batched_muon_falls_back_for_advanced_variants():
    params = _parameters(3)
    optimizer = SpectralMuon(
        [{"params": params, "use_muon": True, "lr": 0.01}],
        spectral_power=0.25, batched=True,
    )
    for parameter in params:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert all("mom" in optimizer.state[parameter] for parameter in params)
