import pytest

from rwkv_lab.powercool import powercool_lr


def test_powercool_has_warmup_plateau_and_cooldown():
    args = dict(peak_lr=1.0, min_lr=0.1, total_steps=100, warmup_steps=10,
                cooldown_fraction=0.2, power=2.0)
    assert powercool_lr(0, **args) == pytest.approx(0.1)
    assert powercool_lr(9, **args) == pytest.approx(1.0)
    assert powercool_lr(79, **args) == pytest.approx(1.0)
    assert powercool_lr(90, **args) < 1.0
    assert powercool_lr(100, **args) == pytest.approx(0.1)


def test_powercool_is_monotone_after_warmup():
    args = dict(peak_lr=3e-4, min_lr=3e-5, total_steps=50, warmup_steps=5,
                cooldown_fraction=0.4, power=1.5)
    rates = [powercool_lr(i, **args) for i in range(50)]
    assert all(a <= b for a, b in zip(rates, rates[1:10]))
    assert all(a >= b for a, b in zip(rates[29:], rates[30:]))


def test_powercool_rejects_bad_configuration():
    with pytest.raises(ValueError):
        powercool_lr(0, peak_lr=1, total_steps=0)
    with pytest.raises(ValueError):
        powercool_lr(0, peak_lr=1, min_lr=2, total_steps=10)
