"""MuonClip helpers, extracted from train_mla.py so the live conversion pipeline
(convert_train.py) no longer depends on the 2200-line MLA trainer.

  * _ParamProxy: a fake nn.Module that yields a curated (name, param) list to feed
    MuonClip a subset of the model's params without touching the model.
  * _make_guarded_muonclip_class(): lazy factory -> a MuonClip subclass whose
    per-param effective step is capped at lr*RMS(update)/RMS(p) <= max_ratio
    (the 0.4*sqrt(max_dim) MuonClip amplifier is otherwise unstable early —
    see the muonclip-lr-units note). Imports muon/utils_muon only when called.
"""
from __future__ import annotations

import torch

class _ParamProxy(torch.nn.Module):
    """Fake nn.Module that yields a fixed (name, param) list from
    `named_parameters()`. Used to feed MuonClip a curated subset of the model's
    params (e.g. excluding embed_tokens / lm_head) without modifying the model
    itself or MuonClip's internal routing."""
    def __init__(self, named_params):
        super().__init__()
        self._named_params = list(named_params)

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        for n, p in self._named_params:
            yield (prefix + n if prefix else n), p


def _make_guarded_muonclip_class():
    """Lazy class factory: imports the muon-clip pkg only if the user actually
    enables `guarded_muonclip=1`. Returns a subclass of MuonClip with a guarded
    `single_muon_step`."""
    from muon import MuonClip
    from utils_muon import muon_update, adam_update

    class GuardedMuonClip(MuonClip):
        """MuonClip subclass whose per-param effective step is capped at
        `lr * RMS(update) / RMS(p) <= max_<muon|adam>_ratio`.

        Direction is unchanged (Newton-Schulz output is preserved); only the
        per-param scalar `alpha` is reduced. Set max_*_ratio big enough and the
        guard is a no-op; small enough and it acts as a hard stability cap that
        prevents the corrected-RMS amplifier from blowing up early-step updates.
        """
        def __init__(self, *args, max_muon_ratio: float = 5e-4,
                     max_adam_ratio: float = 1e-4, guard_stats_every: int = 10, **kwargs):
            super().__init__(*args, **kwargs)
            assert not self.enable_clipping, (
                "GuardedMuonClip assumes enable_clipping=False — QK-clipping is "
                "incompatible with MLA's q_a_proj/q_b_proj naming."
            )
            self.max_muon_ratio = float(max_muon_ratio)
            self.max_adam_ratio = float(max_adam_ratio)
            self.guard_stats_every = max(1, int(guard_stats_every))
            # Per-step diagnostics — read by the train loop and logged.
            self.last_guard_saturation = {
                "muon_sat": 0, "muon_total": 0,
                "adam_sat": 0, "adam_total": 0,
            }

        @torch.no_grad()
        def single_muon_step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # Diagnostics materialize only at log-like cadence; scalar .item() otherwise
            # serializes every optimizer step with the GPU.
            dev = next((p.device for g in self.param_groups for p in g["params"]
                        if p.requires_grad), torch.device("cpu"))
            collect_stats = (getattr(self, "_step", 0) + 1) % self.guard_stats_every == 0
            muon_sat_t = torch.zeros((), device=dev, dtype=torch.long) if collect_stats else None
            adam_sat_t = torch.zeros((), device=dev, dtype=torch.long) if collect_stats else None
            muon_total = 0
            adam_total = 0

            for group in self.param_groups:
                lr = float(group["lr"])
                wd = float(group.get("weight_decay", 0.0))
                if group["use_muon"]:
                    cap = self.max_muon_ratio
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["velocity_buffer"] = torch.zeros(
                                (p.size(-2), 1), device=p.device,
                            )
                            state["step"] = 0
                        state["step"] += 1
                        update = muon_update(
                            p.grad, state["momentum_buffer"], state["velocity_buffer"],
                            step=state["step"], beta=group["beta"], eps=group["eps"],
                            ortho_polynomials=self.ortho_polynomials,
                            ns_steps=self.ns_steps, cans_ortho=self.cans_ortho,
                        )
                        # GPU-resident scaling: lr * RMS(update) / RMS(p) <= cap.
                        p_rms = p.detach().float().square().mean().sqrt().clamp_min(1e-12)
                        u_rms = update.float().square().mean().sqrt().clamp_min(1e-12)
                        rho = lr * u_rms / p_rms
                        scale = torch.clamp(cap / rho.clamp_min(1e-12), max=1.0)
                        update.mul_(scale)
                        if collect_stats:
                            muon_sat_t += (scale < 0.99).long()
                        muon_total += 1
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(update.reshape(p.shape), alpha=-lr)
                else:  # Adam side
                    cap = self.max_adam_ratio
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(
                            p.grad, state["exp_avg"], state["exp_avg_sq"],
                            state["step"], group["betas"], group["eps"],
                        )
                        p_rms = p.detach().float().square().mean().sqrt().clamp_min(1e-12)
                        u_rms = update.float().square().mean().sqrt().clamp_min(1e-12)
                        rho = lr * u_rms / p_rms
                        scale = torch.clamp(cap / rho.clamp_min(1e-12), max=1.0)
                        update.mul_(scale)
                        if collect_stats:
                            adam_sat_t += (scale < 0.99).long()
                        adam_total += 1
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(update, alpha=-lr)

            if collect_stats:
                self.last_guard_saturation = {
                    "muon_sat": int(muon_sat_t.item()),
                    "muon_total": muon_total,
                    "adam_sat": int(adam_sat_t.item()),
                    "adam_total": adam_total,
                }
            self._step += 1
            return loss

    return GuardedMuonClip
