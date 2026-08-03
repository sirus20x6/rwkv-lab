"""Qualification gates for the fused AdamW optimizer component.

The component surface already declares `fused` and `foreach` as mutually
exclusive, validated fields, so fused AdamW has always been *selectable*. What
did not exist was evidence that selecting it is safe. These tests supply the
gates the fused-kernel card names, and they are written to fail rather than to
reassure.

One gate is deliberately absent. "Resumed trajectory" cannot be expressed as
bit-identity: the two kernels agree to float32 epsilon on a single step and
then diverge as training amplifies that agreement-level difference, about an
order of magnitude every few steps. No kernel change removes this, and every
non-bit-identical implementation behaves the same way. Defining trajectory
equivalence statistically is its own decision and is tracked separately; until
it lands, this file measures the divergence and pins its SHAPE rather than
asserting an equality that can never hold.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

# Single-step agreement is a kernel property and should sit at float32 epsilon.
# 1e-6 relative is far looser than the ~1e-7 observed, so a real regression in
# the update math fails while ordinary reassociation does not.
SINGLE_STEP_RELATIVE_TOLERANCE = 1.0e-6


def _requires_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("fused optimizer qualification requires a CUDA device")


def _model(width: int = 512, depth: int = 2) -> "torch.nn.Module":
    torch.manual_seed(0)
    layers = [torch.nn.Linear(width, width) for _ in range(depth)]
    return torch.nn.Sequential(*layers).cuda()


def _train(fused: bool, steps: int, width: int = 512, depth: int = 2):
    model = _model(width, depth)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, fused=fused, foreach=not fused)
    torch.manual_seed(1)
    batch = torch.randn(32, width, device="cuda")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward()
        optimizer.step()
    weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    return model, optimizer, weights


def _relative_difference(left, right) -> float:
    return (left - right).abs().max().item() / left.abs().max().item()


def test_single_step_update_matches_the_reference():
    """The update gate: one fused step must equal one reference step."""
    _requires_cuda()
    _, _, reference = _train(fused=False, steps=1)
    _, _, candidate = _train(fused=True, steps=1)
    difference = _relative_difference(reference, candidate)
    assert difference < SINGLE_STEP_RELATIVE_TOLERANCE, (
        f"fused AdamW diverges from the reference on a single step by "
        f"{difference:.3e}, beyond {SINGLE_STEP_RELATIVE_TOLERANCE:.0e}; that "
        f"is an update-math difference, not reassociation noise")


def test_optimizer_state_round_trips_in_both_directions():
    """The state gate: a checkpoint written by one mode resumes in the other.

    fused keeps `step` as a CUDA tensor while foreach keeps it on CPU, so the
    two write structurally different state. This asserts the difference is
    benign rather than assuming it: a run must be able to switch modes across a
    resume without the optimizer refusing the state or failing its next step.
    """
    _requires_cuda()
    _, fused_optimizer, _ = _train(fused=True, steps=3)
    written_by_fused = fused_optimizer.state_dict()

    reference_model, reference_optimizer, _ = _train(fused=False, steps=3)
    reference_optimizer.load_state_dict(written_by_fused)
    torch.manual_seed(1)
    batch = torch.randn(32, 512, device="cuda")
    reference_optimizer.zero_grad(set_to_none=True)
    reference_model(batch).square().mean().backward()
    reference_optimizer.step()

    written_by_reference = reference_optimizer.state_dict()
    resumed_model, resumed_optimizer, _ = _train(fused=True, steps=3)
    resumed_optimizer.load_state_dict(written_by_reference)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_model(batch).square().mean().backward()
    resumed_optimizer.step()

    for state in (written_by_fused, written_by_reference):
        entry = next(iter(state["state"].values()))
        assert {"step", "exp_avg", "exp_avg_sq"} <= set(entry)


def test_trajectory_divergence_starts_at_epsilon_and_grows():
    """Not a parity gate. Pins the SHAPE of the divergence.

    Bit-identity over a trajectory is unachievable for any non-bit-identical
    kernel, so asserting it would reject every fast optimizer for a reason
    unrelated to its quality. What is checkable is where the divergence starts
    and that it grows: starting at float32 epsilon means the two kernels agree
    on the update and differ only in accumulation order, and growth confirms
    the trajectory amplifies that rather than the kernels disagreeing outright.

    Deliberately NOT asserted: step-by-step monotonicity. Measured across
    1/2/5 steps at this size the sequence is 1.6e-07, 1.2e-05, 9.6e-06 — it
    grows overall but dips, because a training trajectory is chaotic rather
    than smooth. An earlier version of this test asserted monotonicity and
    failed against the real numbers; the assumption was wrong, not the kernel.
    """
    _requires_cuda()
    first_step = _relative_difference(
        _train(fused=False, steps=1)[2], _train(fused=True, steps=1)[2])
    later = _relative_difference(
        _train(fused=False, steps=25)[2], _train(fused=True, steps=25)[2])

    assert first_step < SINGLE_STEP_RELATIVE_TOLERANCE, (
        f"divergence starts at {first_step:.3e}, above single-step tolerance, "
        f"so the update math differs rather than the accumulation order")
    assert later > first_step, (
        f"divergence did not grow over 25 steps ({first_step:.3e} -> "
        f"{later:.3e}); if the trajectories track exactly, this gate is "
        f"measuring nothing")


def test_fused_is_not_slower_than_the_reference():
    """The speed gate. A candidate that is not faster has no case at all.

    Deliberately asserts only 'not slower' rather than a specific speedup:
    the measured 8x is shape and device dependent, and pinning it would make
    this test a hardware assertion rather than a qualification one.
    """
    _requires_cuda()
    import time

    def measure(fused: bool) -> float:
        model = _model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0e-3, fused=fused, foreach=not fused)
        torch.manual_seed(1)
        batch = torch.randn(32, 512, device="cuda")
        for _ in range(5):  # disposable warmup, outside the timed window
            optimizer.zero_grad(set_to_none=True)
            model(batch).square().mean().backward()
            optimizer.step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            model(batch).square().mean().backward()
            optimizer.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / 20

    reference = measure(fused=False)
    candidate = measure(fused=True)
    assert candidate <= reference * 1.1, (
        f"fused {candidate * 1e3:.3f} ms/step is slower than the reference "
        f"{reference * 1e3:.3f} ms/step")
