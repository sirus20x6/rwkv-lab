"""grok_autopilot.py — reactive recovery for anti-grokking collapse.

When held-out ppl regresses from its OWN best while training keeps improving
(late-stage "un-grokking", 2602.02859), this escalates instead of stopping:

  rung 2  weight-EMA best: track an EMA of the student weights and, each eval,
          keep whichever of {raw, EMA} ppl is lower as the saved best. The
          average resists the post-grok wobble, so the saved model keeps
          capturing lower ppl with zero impact on the training trajectory.
  rung 3  on a detected collapse: bump regularization (nuc_weight / weight_decay)
          and, if restore_best, roll the live model back to out/best/ckpt.pt
          (+clear optimizer momentum) so it re-descends from the minimum rather
          than training forward out of the degenerate basin.

Gated by --grok-autopilot (default off) and fail-safe — every method swallows its
own errors so a recovery quirk can never break the loop, and out/best/ckpt.pt is
always on disk regardless. The dashboard detector owns the cheap in-place LR cool
(lr_scale via the control table); this owns the structural moves the trainer must
do (EMA, reg escalation, restore-best). EMA is disabled under schedulefree, which
already averages its iterate.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import torch


class GrokAutopilot:
    def __init__(self, student, codec, opt, out_dir, is_sf, *, enabled=False,
                 ema_decay=0.999, collapse_thresh=0.02, patience=2, stall_patience=3,
                 max_restarts=3, reg_mult=2.0, restore_best=True, lookahead=None):
        self.enabled = bool(enabled)
        self.student, self.codec, self.opt = student, codec, opt
        self.lookahead = lookahead       # NextLat/TOP aux heads: rolled back WITH the backbone
        self.out, self.is_sf = Path(out_dir), is_sf
        self.ema_decay = ema_decay
        self.thresh, self.patience = collapse_thresh, max(1, patience)
        self.stall_patience = max(1, stall_patience)
        self.max_restarts, self.reg_mult = max_restarts, reg_mult
        self.restore_best = restore_best
        self.overrides: dict = {}        # effective-knob overrides (nuc_weight, weight_decay, grokfast_lamb)
        self.restarts = 0
        self.recent = deque(maxlen=self.patience)
        self.no_improve = 0              # evals since best_ppl last improved (stall signal)
        self.best_seen = float("inf")
        self.cur_nuc = 0.0
        self.cur_wd = 0.0
        self.cur_gf = 0.0
        self.use_ema = self.enabled and not is_sf
        self.ema = None
        if self.use_ema:
            self.ema = {n: p.detach().float().clone()
                        for n, p in student.named_parameters() if p.requires_grad}

    # ---- rung 2: weight EMA ------------------------------------------------
    def update_ema(self):
        if not self.use_ema:
            return
        d = self.ema_decay
        with torch.no_grad():
            for n, p in self.student.named_parameters():
                e = self.ema.get(n)
                if e is not None:
                    e.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def eval_ema(self, eval_fn, best_ppl, save_fn):
        """Load EMA weights, run eval_fn()->dict; if its ppl beats best_ppl call
        save_fn(ppl) while the student still holds the EMA weights (so the saved
        best IS the average), then restore. Returns (ev_or_None, saved). Never raises."""
        if not self.use_ema:
            return None, False
        backup, ev, saved = {}, None, False
        try:
            with torch.no_grad():
                for n, p in self.student.named_parameters():
                    e = self.ema.get(n)
                    if e is not None:
                        backup[n] = p.detach().clone()
                        p.copy_(e.to(p.dtype))
            ev = eval_fn()
            if ev and ev.get("ppl") is not None and ev["ppl"] < best_ppl:
                save_fn(ev["ppl"])
                saved = True
        except Exception:
            ev, saved = None, False
        finally:
            with torch.no_grad():
                for n, p in self.student.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])
        return ev, saved

    # ---- rung 3: collapse detection + recovery -----------------------------
    def on_eval(self, step, ppl, best_ppl):
        """Escalate the generalizing pressure when training is (a) STUCK in a
        memorization plateau (held-out ppl not improving for `stall_patience` evals)
        or (b) COLLAPSING (held-out regressing above its own best). Both bump the
        low-rank/decay pressure and kick GrokFast to escape faster; collapse also
        restores the best checkpoint. Returns an action dict to emit (empty if idle)."""
        if not self.enabled or ppl is None:
            return {}
        # stall signal: how long since best_ppl actually improved
        if best_ppl < self.best_seen - 1e-9:
            self.best_seen, self.no_improve = best_ppl, 0
        else:
            self.no_improve += 1
        self.recent.append(ppl)
        collapsed = (best_ppl < float("inf") and len(self.recent) >= self.patience
                     and all(p > best_ppl * (1.0 + self.thresh) for p in self.recent))
        stalled = self.no_improve >= self.stall_patience
        if not ((collapsed or stalled) and self.restarts < self.max_restarts):
            return {}
        self.restarts += 1
        self.cur_nuc = (self.cur_nuc or 1e-5) * self.reg_mult     # more low-rank pressure
        self.cur_wd = (self.cur_wd or 1e-3) * self.reg_mult
        self.cur_gf = max(self.cur_gf, 2.0)                       # kick GrokFast to escape faster
        self.overrides["nuc_weight"] = self.cur_nuc
        self.overrides["weight_decay"] = self.cur_wd
        self.overrides["grokfast_lamb"] = self.cur_gf
        mode = "collapse" if collapsed else "stall"
        action = {"autopilot": mode, "ap_restart": self.restarts, "ap_nuc_weight": self.cur_nuc,
                  "ap_weight_decay": self.cur_wd, "ap_grokfast_lamb": self.cur_gf}
        if collapsed and self.restore_best:
            action["ap_restored_best"] = int(self._restore_best())
        self.recent.clear()
        self.no_improve = 0
        return action

    def _restore_best(self) -> bool:
        try:
            bp = self.out / "best" / "ckpt.pt"
            if not bp.exists():
                return False
            state = torch.load(bp, map_location="cpu")
            self.student.load_state_dict(state["student"])
            if self.codec is not None and "codec" in state:
                self.codec.load_state_dict(state["codec"])
            if self.lookahead is not None and "lookahead" in state:
                # aux heads must roll back with the backbone: their predictions/rankings
                # are functions of the restored hidden states, and stale heads would
                # re-shape the backbone toward the collapsed trajectory
                self.lookahead.load_state_dict(state["lookahead"])
            if self.use_ema:                    # re-seed EMA at the restored weights
                self.ema = {n: p.detach().float().clone()
                            for n, p in self.student.named_parameters() if p.requires_grad}
            if not self.is_sf:                  # drop momentum so we don't re-descend in
                try:
                    self.opt.state.clear()
                except Exception:
                    pass
            return True
        except Exception:
            return False
