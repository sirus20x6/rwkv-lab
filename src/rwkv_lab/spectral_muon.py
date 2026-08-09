"""spectral_muon.py — one configurable Muon-family optimizer collecting the 2026
spectral-optimizer levers as flags. 2D weight matrices get the Muon update; all
other params (norms/biases/embeddings/scalars) use a built-in AdamW fallback.
Routing is by per-group `use_muon` (matches this repo's GuardedMuonClip convention).

Defaults reproduce vanilla Muon (orthogonalize momentum via Newton–Schulz → UVᵀ).
Per 2D matrix each step:
  1. momentum   m = μ·m + g'            (g' = g + α·EMA(Δg) if MONA)        [2605.26842]
  2. Muon²      m ← m / (√v + ε)        Adam-style precond before NS         [2604.09967]
  3. MuonEq     equilibrate rows/cols of m                                   [2603.28254]
  4. orthog.    p==0 → Newton–Schulz polar (UVᵀ);  p∈(0,1] → U·Σ^p·Vᵀ (eigh)  [2606.13867 — Muonᵖ spectral-power orthogonalization]
  5. Aurora     equal-row-norm for tall matrices                            [2606.27715]
  6. MUON+      row/col-normalize the orthogonalized update                 [2602.21545]
  7. scale by `scale·√(max(m,n))` (the repo's MuonClip amplifier) and lr; decoupled WD
`cubic=True` uses the odd-cubic NS schedule (~1/3 fewer matmuls).            [2606.00371]

RSAV (`rsav=True`) — SpecMuon-inspired relaxed scalar-auxiliary-variable step gate
[2602.16167 — SpecMuon RSAV, adapted]. A single GLOBAL scalar `r` tracks the gradient "energy"
E = Σ‖g‖² (over the Muon-routed 2D matrices) via the SAV chain rule, and every
Muon update this step is scaled by ξ = clamp(r/√(E+C), 1±cap). ξ≈1 at equilibrium
and dips below 1 when the energy spikes faster than `r` has tracked it — a
stability damper that costs one extra reduction over the grads per step. NOTE:
this is the gradient-energy variant (self-contained — the optimizer already sees
every grad), NOT the paper's faithful loss-energy SAV; the r-update chain rule is
therefore heuristic (exact only when E is the loss). `rsav_cap=0` forces ξ≡1
(inert = vanilla). `r` is a global scalar kept on `self` rather than per-parameter
state; the optimizer's versioned state dictionary persists it and rejects a
resume whose global RSAV configuration differs.

The per-matrix lever knobs live in each param group, so a trainer may live-tune
them by setting e.g. opt.param_groups[i]["spectral_power"] = v between steps; the
GLOBAL RSAV knobs live on the optimizer (opt.rsav_c, opt.rsav_cap, opt.rsav_relax).

Precision: all optimizer state (momentum, MONA acc, Muon² v, Adam moments) is kept
in fp32 regardless of param dtype — bf16 moments quantize away small updates (the
known bf16-AdamW failure). The NS polar iteration runs in bf16 (KJ Muon convention,
~2x faster than fp32+TF32); the eigh/SVD power paths need fp32 (cuSOLVER).
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

_QUINTIC = (3.4445, -4.7750, 2.0315)  # Keller-Jordan Muon NS coefficients
_GLOBAL_STATE_KEY = "spectral_muon_global_state_v1"


def _ns_quintic(X, steps):
    a, b, c = _QUINTIC
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _ns_cubic(X, steps):
    # odd cubic: 2 matmuls/step vs 3 for quintic (~1/3 cheaper) [2606.00371]
    a, b = 1.5, -0.5
    for _ in range(steps):
        X = a * X + b * ((X @ X.mT) @ X)
    return X


@torch.no_grad()
def _power_eigh(G, power, rtol=1e-3):
    """U.Sigma^p.Vt via eigh of the smaller-dim symmetric Gram (a matmul + symmetric
    eigensolver instead of a full SVD of G). Identity: U.Sigma^p.Vt = G.(GtG)^((p-1)/2).
    Forming the Gram squares the condition number, so directions with Sigma below
    rtol*Sigma_max sit beneath the eigh noise floor and are ZEROED — mirroring SVD's
    Sigma^p->0 on the null space. Without that, eigh noise on those directions gets
    amplified by the negative inner power Sigma^(p-1) and the update explodes (a dense
    full-rank momentum is unaffected; this is the robustness guard for ill-conditioned /
    rank-deficient G). Solves GtG when n<=m else GGt — the smaller dimension, so the
    rank-r low-rank factors cost only an r x r eigh."""
    m, n = G.shape[-2], G.shape[-1]
    if n <= m:
        evals, V = torch.linalg.eigh(G.mT @ G)            # GtG = V Sigma^2 Vt  (n x n)
        s = evals.clamp_min(0.0).sqrt()                   # Sigma
        inner = torch.where(s > rtol * s.amax(), s.pow(power - 1.0), s.new_zeros(()))
        return G @ ((V * inner) @ V.mT)                   # U Sigma . Sigma^(p-1) Vt = U Sigma^p Vt
    evals, U = torch.linalg.eigh(G @ G.mT)                # GGt = U Sigma^2 Ut  (m x m)
    s = evals.clamp_min(0.0).sqrt()
    inner = torch.where(s > rtol * s.amax(), s.pow(power - 1.0), s.new_zeros(()))
    return ((U * inner) @ U.mT) @ G


_power_fallback_warned = [False]


def orthogonalize(G, steps=5, cubic=False, power=0.0, power_method="eigh"):
    """(Fractional-power) orthogonalized factor of G. power>0 uses the math-identical
    eigh-on-Gram path (power_method 'eigh', default — see _power_eigh; measured ~14x
    faster than gesvd at 4096², 161ms vs 2.3s on Blackwell) OR exact gesvd ('svd',
    debug/verification only). The NS polar path runs in bf16 (~2x faster than
    fp32+TF32); the power paths need fp32 (cuSOLVER)."""
    if power and power > 0.0:
        try:
            if power_method == "eigh":
                return _power_eigh(G.float(), power)
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            return (U * S.clamp_min(0).pow(power)) @ Vh
        except Exception as e:  # solver failure -> NS polar (spectral_power OFF for this matrix/step)
            if not _power_fallback_warned[0]:
                _power_fallback_warned[0] = True
                print(f"[spectral_muon] WARNING: power_method={power_method!r} failed ({e!r}); "
                      "falling back to plain NS polar — spectral_power is silently ignored "
                      "wherever this recurs (warning printed once).", flush=True)
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    X = _ns_cubic(X, steps) if cubic else _ns_quintic(X, steps)
    if transpose:
        X = X.mT
    return X


def orthogonalize_batched(G, steps=5, cubic=False):
    """Full-matrix Newton–Schulz for a same-shaped matrix batch.

    This is mathematically the scalar ``orthogonalize`` path with independent
    Frobenius normalization per matrix, but launches each NS matmul as one
    batched GEMM.
    """
    if G.ndim != 3:
        raise ValueError("batched Muon expects [batch, rows, cols]")
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    norms = X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)
    X = X / norms
    X = _ns_cubic(X, steps) if cubic else _ns_quintic(X, steps)
    if transpose:
        X = X.mT
    return X


def _rms(x, dim):
    return x.pow(2).mean(dim=dim, keepdim=True).clamp_min(1e-12).sqrt()


def _ddc_project(U, W, mode, strength):
    """Dead-Direction Conditioner (abelian subset, 2606.29176 — DDC (Dead-Direction Conditioner)): remove the part of the
    update U that merely RESCALES channels of W — the per-channel gauge / "dead"
    directions where the loss is flat. 'row' = output-channel scale (RMSNorm-after /
    next-layer rescale gauge), 'col' = input-channel scale (RMSNorm-before), 'both' =
    both. Keeps the step on the loss-relevant quotient (resists over-training drift into
    degenerate flat minima). strength in [0,1] = fraction of the gauge component removed."""
    out = U
    if "row" in mode or mode == "both":
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        out = out - strength * (out * Wn).sum(dim=1, keepdim=True) * Wn
    if "col" in mode or mode == "both":
        Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
        out = out - strength * (out * Wn).sum(dim=0, keepdim=True) * Wn
    return out


def himuon_orthogonalize(G, steps=5, tile=512, cubic=False):
    """Hierarchical/Tiled Muon (2606.27216): block-diagonal Newton-Schulz. Pad G [H,W] to a grid
    of tile×tile blocks, run the SAME quintic NS INDEPENDENTLY on each block (each Frobenius-
    normalized on its own), reassemble, strip padding. Returns the (unscaled) update — the caller
    uses scale c·√tile, NOT c·√max(H,W). Cheaper: replaces the min(H,W) NS factor with `tile`."""
    H, W = G.shape[-2], G.shape[-1]
    T = int(tile)
    if T >= max(H, W):                             # single tile == whole matrix -> exact full NS
        return orthogonalize(G, steps=steps, cubic=cubic)   # (handles the tall-matrix transpose)
    R, C = (H + T - 1) // T, (W + T - 1) // T
    Gp = G.new_zeros(R * T, C * T)
    Gp[:H, :W] = G
    tiles = Gp.reshape(R, T, C, T).permute(0, 2, 1, 3).reshape(R * C, T, T).bfloat16()
    n = tiles.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)
    X = tiles / n
    X = _ns_cubic(X, steps) if cubic else _ns_quintic(X, steps)   # batched over the tile axis
    Up = X.reshape(R, C, T, T).permute(0, 2, 1, 3).reshape(R * T, C * T)
    return Up[:H, :W]


def _sinkhorn_normalize(X, iters=5):
    """f_Sink (ARO): `iters` rounds of row-then-col L2 normalization. Rotation-NON-equivariant,
    so ARO's rotate→f→rotate-back actually does something (unlike orthogonalization)."""
    for _ in range(iters):
        X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        X = X / (X.norm(dim=0, keepdim=True) + 1e-8)
    return X


def _cholesky_qr(A, eps=1e-6):
    """Shifted Cholesky-QR: Q-factor of A [m,m] via P=AᵀA+εI=LLᵀ, Q=A·L⁻ᵀ. Returns an ORTHONORMAL
    Q. Degenerate inputs select identity entirely on-device, keeping the per-matrix optimizer path
    free of Python tensor predicates and CUDA stream synchronization."""
    m = A.shape[-1]
    Im = torch.eye(m, device=A.device, dtype=A.dtype)
    gram = A.transpose(-1, -2) @ A
    L, info = torch.linalg.cholesky_ex(gram + eps * Im, check_errors=False)
    Q = torch.linalg.solve_triangular(L, A.transpose(-1, -2), upper=False).transpose(-1, -2)
    # One polar Newton correction repairs the small orthogonality error introduced by the shift.
    Q = Q @ (1.5 * Im - 0.5 * (Q.transpose(-1, -2) @ Q))
    valid = (info == 0) & torch.isfinite(Q).all() & (gram.abs().amax() > eps)
    return torch.where(valid, Q, Im)


def _aro_core(M, R, sink_iters: int):
    """Stable ARO tensor subgraph, separated so qualified CUDA runs can compile it."""
    z_prev = _sinkhorn_normalize(R.t() @ M, sink_iters)
    new_R = _cholesky_qr(M @ z_prev.t())
    dW = new_R @ _sinkhorn_normalize(new_R.t() @ M, sink_iters)
    return new_R, dW / (dW.norm() + 1e-12)


class SpectralMuon(Optimizer):
    def __init__(self, param_groups, *, momentum=0.95, nesterov=False,
                 ns_steps=5, cubic=False, spectral_power=0.0, power_method="eigh",
                 second_moment=False, sm_beta2=0.99, sm_eps=1e-8,
                 equilibrate="none", plus_norm="none", row_uniform=False,
                 mona=False, mona_beta=0.9, mona_alpha=0.1, scale=0.4,
                 ddc_strength=0.0, ddc_mode="both",
                 rsav=False, rsav_c=1.0, rsav_cap=0.2, rsav_relax=0.0,
                 tile_size=0, da_muon=False, da_eta_max=0.01, da_r0=1e-3,
                 aro=False, aro_sink_iters=5, aro_compile=False,
                 batched=False, compile_ns=False,
                 row_update_floor=0.0, radial_brake=0.0, radius_pin=False,
                 cautious_weight_decay=False, adam_update_interval=1,
                 weight_decay=0.0, adam_betas=(0.9, 0.95), adam_eps=1e-8):
        if int(adam_update_interval) < 1:
            raise ValueError("adam_update_interval must be >= 1")
        if float(row_update_floor) < 0.0:
            raise ValueError("row_update_floor must be non-negative")
        if not 0.0 <= float(radial_brake) <= 1.0:
            raise ValueError("radial_brake must be in [0, 1]")
        defaults = dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, cubic=cubic,
                        spectral_power=spectral_power, power_method=power_method, second_moment=second_moment,
                        sm_beta2=sm_beta2, sm_eps=sm_eps, equilibrate=equilibrate,
                        plus_norm=plus_norm, row_uniform=row_uniform, mona=mona,
                        mona_beta=mona_beta, mona_alpha=mona_alpha, scale=scale,
                        ddc_strength=ddc_strength, ddc_mode=ddc_mode,
                        tile_size=tile_size, da_muon=da_muon, da_eta_max=da_eta_max, da_r0=da_r0,
                        aro=aro, aro_sink_iters=aro_sink_iters,
                        row_update_floor=row_update_floor, radial_brake=radial_brake,
                        radius_pin=radius_pin, cautious_weight_decay=cautious_weight_decay,
                        adam_update_interval=int(adam_update_interval),
                        weight_decay=weight_decay, adam_betas=adam_betas, adam_eps=adam_eps,
                        use_muon=False, lr=3e-4)
        super().__init__(param_groups, defaults)
        # RSAV is a GLOBAL scalar coupling (not per-param), so its knobs + state live
        # on the optimizer, not in param groups. r re-inits on resume (see docstring).
        self.rsav = bool(rsav)
        self.rsav_c = float(rsav_c)
        self.rsav_cap = float(rsav_cap)
        self.rsav_relax = float(rsav_relax)
        self._rsav_r = None          # fp32 scalar tensor once seen
        self._rsav_dE = None         # per-step energy-change accumulator (fp32 scalar)
        self._rsav_last_xi = 1.0     # diagnostics / tests
        self.aro_compile = bool(aro_compile)
        self._compiled_aro_core = None
        self.batched = bool(batched)
        self.compile_ns = bool(compile_ns)
        self._compiled_ns: dict[tuple[int, bool], object] = {}
        self._compile_ns_failed = False

    def state_dict(self):
        result = super().state_dict()
        rsav_r = self._rsav_r
        rsav_xi = self._rsav_last_xi
        result[_GLOBAL_STATE_KEY] = {
            "rsav": self.rsav,
            "rsav_c": self.rsav_c,
            "rsav_cap": self.rsav_cap,
            "rsav_relax": self.rsav_relax,
            "rsav_r": (
                rsav_r.detach().clone() if torch.is_tensor(rsav_r) else None
            ),
            "rsav_last_xi": (
                rsav_xi.detach().clone()
                if torch.is_tensor(rsav_xi)
                else float(rsav_xi)
            ),
        }
        return result

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict casts float state to each param's dtype (bf16 for a
        # bf16 model) BEFORE any post-hook can run, silently re-quantizing the fp32
        # state on every resume. Upcasting afterwards cannot restore the lost bits, so
        # re-install the ORIGINAL tensors (as fp32, on the param's device) after the
        # parent load. Old bf16-state ckpts upcast losslessly through the same path.
        saved_state = {k: dict(v) for k, v in state_dict.get("state", {}).items()}
        global_state = state_dict.get(_GLOBAL_STATE_KEY)
        if global_state is not None:
            if not isinstance(global_state, dict) or set(global_state) != {
                "rsav",
                "rsav_c",
                "rsav_cap",
                "rsav_relax",
                "rsav_r",
                "rsav_last_xi",
            }:
                raise ValueError("SpectralMuon global checkpoint state is invalid")
            expected = {
                "rsav": self.rsav,
                "rsav_c": self.rsav_c,
                "rsav_cap": self.rsav_cap,
                "rsav_relax": self.rsav_relax,
            }
            if any(global_state[key] != value for key, value in expected.items()):
                raise ValueError(
                    "SpectralMuon checkpoint RSAV configuration is incompatible"
                )
        elif self.rsav:
            raise ValueError("SpectralMuon RSAV checkpoint omits global state")
        super().load_state_dict(
            {
                "state": state_dict["state"],
                "param_groups": state_dict["param_groups"],
            }
        )
        id_map = {
            old_id: p
            for old_ids, g in zip(
                (sg["params"] for sg in state_dict["param_groups"]), self.param_groups
            )
            for old_id, p in zip(old_ids, g["params"])
        }
        for old_id, st in saved_state.items():
            p = id_map.get(old_id)
            if p is None or p not in self.state:
                continue
            for k, v in st.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    self.state[p][k] = v.to(device=p.device, dtype=torch.float32)
        if global_state is not None:
            first_parameter = next(
                (
                    parameter
                    for group in self.param_groups
                    for parameter in group["params"]
                ),
                None,
            )
            device = first_parameter.device if first_parameter is not None else "cpu"
            saved_r = global_state["rsav_r"]
            if saved_r is not None and not torch.is_tensor(saved_r):
                raise ValueError("SpectralMuon RSAV scalar state is invalid")
            self._rsav_r = (
                saved_r.to(device=device, dtype=torch.float32)
                if saved_r is not None
                else None
            )
            saved_xi = global_state["rsav_last_xi"]
            if torch.is_tensor(saved_xi):
                self._rsav_last_xi = saved_xi.to(device=device, dtype=torch.float32)
            elif isinstance(saved_xi, (int, float)) and not isinstance(
                saved_xi, bool
            ):
                self._rsav_last_xi = float(saved_xi)
            else:
                raise ValueError("SpectralMuon RSAV diagnostic state is invalid")

    @staticmethod
    def _is_muon(grp, p):
        return bool(grp.get("use_muon")) and p.ndim == 2 and min(p.shape) > 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        optimizer_step = max(
            (int(group.get("_optimizer_step", 0)) for group in self.param_groups),
            default=0,
        ) + 1
        for group in self.param_groups:
            group["_optimizer_step"] = optimizer_step
        # --- RSAV pre-pass: global gradient energy E = Σ‖g‖² over Muon matrices ---
        xi, sqrt_Et = 1.0, None
        if self.rsav:
            E = None
            for grp in self.param_groups:
                for p in grp["params"]:
                    if p.grad is not None and self._is_muon(grp, p):
                        s = (p.grad.float() ** 2).sum()
                        E = s if E is None else E + s.to(E.device)  # colocate (multi-device safe; no-op single-device)
            if E is not None:
                sqrt_Et = (E + self.rsav_c).sqrt()
                if self._rsav_r is None:
                    self._rsav_r = sqrt_Et.detach().clone()   # first sighting: r = √(E+C), ξ=1
                else:
                    cap = self.rsav_cap
                    # Keep the gate on-device. Converting it to float here forced
                    # one full CUDA-stream synchronization per optimizer step.
                    xi = (self._rsav_r / sqrt_Et).clamp(1.0 - cap, 1.0 + cap)
                self._rsav_dE = torch.zeros((), dtype=torch.float32, device=sqrt_Et.device)
            self._rsav_last_xi = xi
        for grp in self.param_groups:
            adam_params = []
            muon_params = []
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if self._is_muon(grp, p):
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            if self.batched and self._can_batch_group(grp, xi):
                buckets = {}
                for p in muon_params:
                    buckets.setdefault((p.device, p.dtype, tuple(p.shape)), []).append(p)
                for bucket in buckets.values():
                    if len(bucket) > 1:
                        self._batched_muon_step(bucket, grp)
                    else:
                        p = bucket[0]
                        self._muon_step(p, p.grad, grp, self.state[p], xi)
            else:
                for p in muon_params:
                    self._muon_step(p, p.grad, grp, self.state[p], xi)
            self._adam_group_step(
                adam_params,
                grp,
                update=(optimizer_step % int(grp["adam_update_interval"]) == 0),
            )
        # --- RSAV post-step: evolve r by the SAV chain rule, then relax toward √(E+C) ---
        if self.rsav and sqrt_Et is not None:
            r = self._rsav_r + self._rsav_dE / (2.0 * sqrt_Et)
            if self.rsav_relax > 0.0:
                r = (1.0 - self.rsav_relax) * r + self.rsav_relax * sqrt_Et
            self._rsav_r = r.clamp_min(1e-8)
        return loss

    def _can_batch_group(self, grp, xi) -> bool:
        """The fast path intentionally covers the common vanilla/Nesterov NS case."""
        return (
            not self.rsav
            and not torch.is_tensor(xi)
            and float(xi) == 1.0
            and not grp["mona"]
            and not grp["second_moment"]
            and grp["equilibrate"] == "none"
            and grp["plus_norm"] == "none"
            and not grp["row_uniform"]
            and not grp["ddc_strength"]
            and not grp["spectral_power"]
            and not grp["tile_size"]
            and not grp["da_muon"]
            and not grp["aro"]
            and not grp["row_update_floor"]
            and not grp["radial_brake"]
            and not grp["radius_pin"]
            and not grp["cautious_weight_decay"]
        )

    def _batched_orthogonalize(self, stacked, grp):
        steps, cubic = int(grp["ns_steps"]), bool(grp["cubic"])
        if self.compile_ns and stacked.is_cuda and not self._compile_ns_failed:
            key = (steps, cubic)
            if key not in self._compiled_ns:
                try:
                    def core(value):
                        return orthogonalize_batched(value, steps=steps, cubic=cubic)
                    self._compiled_ns[key] = torch.compile(
                        core, dynamic=False, fullgraph=True)
                except Exception:
                    self._compile_ns_failed = True
            if not self._compile_ns_failed:
                try:
                    return self._compiled_ns[key](stacked)
                except Exception as exc:
                    self._compile_ns_failed = True
                    print("[spectral_muon] compiled batched NS failed "
                          f"({type(exc).__name__}: {exc}); using eager batched NS", flush=True)
        return orthogonalize_batched(stacked, steps=steps, cubic=cubic)

    def _batched_muon_step(self, params, grp):
        """Update same-shaped matrices with one batched Newton–Schulz graph."""
        mu = float(grp["momentum"])
        momenta = []
        grads = []
        for p in params:
            st = self.state[p]
            if "mom" not in st:
                st["mom"] = torch.zeros_like(p.grad, dtype=torch.float32)
            momenta.append(st["mom"])
            grads.append(p.grad.float())
        torch._foreach_mul_(momenta, mu)
        torch._foreach_add_(momenta, grads)
        updates = (torch._foreach_add(grads, momenta, alpha=mu)
                   if grp["nesterov"] else momenta)
        orthogonal = self._batched_orthogonalize(
            torch.stack(updates), grp).to(params[0].dtype).unbind(0)
        lr = float(grp["lr"])
        if grp["weight_decay"]:
            torch._foreach_mul_(params, 1.0 - lr * float(grp["weight_decay"]))
        scale = float(grp["scale"]) * max(params[0].shape) ** 0.5
        torch._foreach_add_(params, orthogonal, alpha=-lr * scale)

    def _muon_step(self, p, g, grp, st, xi=1.0):
        lr, mu = grp["lr"], grp["momentum"]
        if "mom" not in st:
            st["mom"] = torch.zeros_like(g, dtype=torch.float32)  # fp32 state: see module docstring
            if grp["mona"]:
                st["gprev"] = torch.zeros_like(g, dtype=torch.float32)
                st["acc"] = torch.zeros_like(g, dtype=torch.float32)
            if grp["second_moment"]:
                st["v"] = torch.zeros_like(g, dtype=torch.float32)
        gg = g
        if grp["mona"]:                                   # MONA curvature/Nesterov term
            d = g - st["gprev"]; st["gprev"].copy_(g)
            st["acc"].mul_(grp["mona_beta"]).add_(d, alpha=1 - grp["mona_beta"])
            gg = g + grp["mona_alpha"] * st["acc"]
        m = st["mom"]; m.mul_(mu).add_(gg)
        u = gg.add(m, alpha=mu) if grp["nesterov"] else m
        if grp["second_moment"]:                          # Muon²
            v = st["v"]; v.mul_(grp["sm_beta2"]).addcmul_(gg, gg, value=1 - grp["sm_beta2"])
            u = u / (v.sqrt() + grp["sm_eps"])
        eq = grp["equilibrate"]                           # MuonEq (pre-orthogonalization)
        if "R" in eq:
            u = u / _rms(u, 1)
        if "C" in eq:
            u = u / _rms(u, 0)
        # --- transform stage: ARO (replaces NS) | Hierarchical tiled NS | standard NS/power ---
        if grp["aro"]:                                     # ARO-Sinkhorn: non-orthonormal update
            o, base_scale = self._aro_update(u, p, st, grp)   # (Aurora/MUON+/DDC don't apply)
        else:
            tile = int(grp["tile_size"])
            tiling = 0 < tile < max(p.shape)               # Hierarchical Muon (tiled NS)
            if tiling:
                o = himuon_orthogonalize(u, grp["ns_steps"], tile, grp["cubic"]).to(p.dtype)
            else:
                o = orthogonalize(u, steps=grp["ns_steps"], cubic=grp["cubic"],
                                  power=grp["spectral_power"], power_method=grp["power_method"]).to(p.dtype)
            if grp["row_uniform"] and o.size(0) >= o.size(1):  # Aurora (tall matrices)
                o = o / _rms(o, 1)
            pn = grp["plus_norm"]                          # MUON+ (post-orthogonalization)
            if pn == "row":
                o = o / _rms(o, 1)
            elif pn == "col":
                o = o / _rms(o, 0)
            if grp["ddc_strength"] > 0.0:                  # DDC: project out the rescale gauge
                o = _ddc_project(o, p, grp["ddc_mode"], grp["ddc_strength"])
            base_scale = grp["scale"] * ((tile if tiling else max(p.shape)) ** 0.5)  # HiMuon: √tile
        eta = self._da_eta(p, st, grp) if grp["da_muon"] else 1.0   # Distance-Aware radius
        step_scale = xi * eta
        if torch.is_tensor(step_scale):
            step_scale = step_scale.to(device=o.device, dtype=torch.float32)
        elif step_scale != 1.0:
            step_scale = float(step_scale)

        # Shape-aware postconditioning from the Track-3 optimizer line. These
        # operations are all opt-in; with defaults this reduces to vanilla Muon.
        update = o.float().mul(float(base_scale))
        if torch.is_tensor(step_scale):
            update.mul_(step_scale)
        elif step_scale != 1.0:
            update.mul_(step_scale)
        floor = float(grp["row_update_floor"])
        if floor > 0.0:
            p_rows = p.detach().float().norm(dim=1, keepdim=True)
            u_rows = update.norm(dim=1, keepdim=True)
            update.mul_((floor * p_rows / u_rows.clamp_min(1e-12)).clamp_min_(1.0))

        needs_reference = bool(
            grp["radial_brake"] or grp["radius_pin"] or grp["cautious_weight_decay"]
        )
        reference = p.detach().float().clone() if needs_reference else None
        brake = float(grp["radial_brake"])
        if brake > 0.0:
            denom = reference.square().sum().clamp_min(1e-20)
            coefficient = (update * reference).sum() / denom
            radial = reference * coefficient
            # p <- p - lr*u moves outward exactly when <p,u> is negative.
            update = torch.where(
                coefficient < 0,
                update - radial + brake * radial,
                update,
            )

        target_norm = None
        if grp["radius_pin"]:
            norm = reference.norm().clamp_min(1e-12)
            target_norm = (norm - lr * (reference * update).sum() / norm).clamp_min(1e-12)

        if grp["weight_decay"] and not grp["cautious_weight_decay"]:
            p.mul_(1.0 - lr * grp["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-lr)
        if target_norm is not None:
            p.mul_((target_norm / p.detach().float().norm().clamp_min(1e-12)).to(p.dtype))
        if grp["weight_decay"] and grp["cautious_weight_decay"]:
            mask = ((update * reference) > 0).to(p.dtype)
            p.mul_(1.0 - lr * grp["weight_decay"] * mask)
        if self.rsav and self._rsav_dE is not None:         # SAV chain rule: dE ≈ Σ ⟨g, Δx⟩
            d = (g.float() * update).sum().to(self._rsav_dE.device)
            self._rsav_dE.add_(d, alpha=-lr)

    def _da_eta(self, p, st, grp):
        """Distance-Aware Muon (2605.18999) adaptive radius: η = clamp(r̄/√(k+1), 0, η_max), with
        r̄ the running-MAX Frobenius distance ‖W − W0‖ from the optimizer's starting weight. Adds
        one W0 snapshot per matrix. (W0 = weight at first step; = init for a fresh run.)"""
        if "da_W0" not in st:
            st["da_W0"] = p.detach().float().clone()
            st["da_rbar"] = torch.tensor(float(grp["da_r0"]), device=p.device)
            st["da_k"] = 0
        st["da_k"] += 1
        d = (p.detach().float() - st["da_W0"]).norm()
        st["da_rbar"] = torch.maximum(st["da_rbar"], d)
        return (st["da_rbar"] / (st["da_k"] ** 0.5)).clamp_max(float(grp["da_eta_max"]))

    def _aro_update(self, u, p, st, grp):
        """ARO-Sinkhorn (2602.09006): rotate momentum into a learned frame R, apply Sinkhorn
        row/col normalization there (rotation-NON-equivariant, so — unlike on NS orthogonalization,
        where ARO is a no-op — this actually reshapes the update), rotate back. Persistent R (m×m)
        is re-estimated each step by orthogonal Procrustes (shifted Cholesky-QR). Returns the
        unit-Frobenius update and ARO's own RMS budget scale 0.2·√(mn)."""
        M = u.float()
        m_dim = M.shape[0]
        if "aro_R" not in st:
            st["aro_R"] = torch.eye(m_dim, device=M.device, dtype=torch.float32)
        R = st["aro_R"]
        it = int(grp["aro_sink_iters"])
        if self.aro_compile and M.is_cuda:
            if self._compiled_aro_core is None:
                self._compiled_aro_core = torch.compile(_aro_core, dynamic=False)
            R, dW = self._compiled_aro_core(M, R, it)
        else:
            R, dW = _aro_core(M, R, it)
        st["aro_R"].copy_(R)
        base_scale = 0.2 * float(M.shape[0] * M.shape[1]) ** 0.5   # ARO's AdamW-budget RMS match
        return dW.to(p.dtype), base_scale

    def _adam_group_step(self, params, grp, *, update=True):
        """Foreach AdamW fallback; DDC matrices retain their per-parameter projection."""
        if not params:
            return
        cadence = int(grp["adam_update_interval"])
        effective_grads = {}
        if cadence > 1:
            for p in params:
                st = self.state[p]
                accumulator = st.get("adam_grad_accum")
                if accumulator is None:
                    accumulator = st["adam_grad_accum"] = torch.zeros_like(
                        p.grad, dtype=torch.float32
                    )
                    st["adam_grad_count"] = 0
                accumulator.add_(p.grad.float())
                st["adam_grad_count"] += 1
                if update:
                    effective_grads[p] = accumulator / max(
                        int(st["adam_grad_count"]), 1
                    )
            if not update:
                return
        else:
            effective_grads = {p: p.grad for p in params}
        ordinary = []
        for p in params:
            st = self.state[p]
            if "exp_avg" not in st:
                st["exp_avg"] = torch.zeros_like(effective_grads[p], dtype=torch.float32)
                st["exp_sq"] = torch.zeros_like(effective_grads[p], dtype=torch.float32)
                st["t"] = 0
            st["t"] += 1
            if grp["ddc_strength"] > 0.0 and p.ndim == 2 and min(p.shape) > 1:
                self._adam_step(p, effective_grads[p], grp, st, increment=False)
            else:
                ordinary.append(p)
        buckets = {}
        for p in ordinary:
            st = self.state[p]
            buckets.setdefault((p.device, st["t"]), []).append(p)
        b1, b2 = grp["adam_betas"]
        for (_, step), bucket in buckets.items():
            grads = [effective_grads[p].float() for p in bucket]
            exp_avg = [self.state[p]["exp_avg"] for p in bucket]
            exp_sq = [self.state[p]["exp_sq"] for p in bucket]
            torch._foreach_mul_(exp_avg, b1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - b1)
            torch._foreach_mul_(exp_sq, b2)
            torch._foreach_addcmul_(exp_sq, grads, grads, value=1 - b2)
            denom = torch._foreach_sqrt(exp_sq)
            torch._foreach_div_(denom, (1 - b2 ** step) ** 0.5)
            torch._foreach_add_(denom, grp["adam_eps"])
            updates = torch._foreach_div(exp_avg, 1 - b1 ** step)
            torch._foreach_div_(updates, denom)
            if grp["weight_decay"]:
                torch._foreach_mul_(bucket, 1.0 - grp["lr"] * grp["weight_decay"])
            # Not named `update`: that is this method's keyword parameter, and
            # shadowing it here would silently break any later use of the flag.
            for p, delta in zip(bucket, updates):
                p.add_(delta.to(p.dtype), alpha=-grp["lr"])
        if cadence > 1:
            for p in params:
                self.state[p]["adam_grad_accum"].zero_()
                self.state[p]["adam_grad_count"] = 0

    def _adam_step(self, p, g, grp, st, *, increment=True):
        b1, b2 = grp["adam_betas"]; eps = grp["adam_eps"]; lr = grp["lr"]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(g, dtype=torch.float32)  # fp32: bf16 moments
            st["exp_sq"] = torch.zeros_like(g, dtype=torch.float32)   # quantize away fine updates
            st["t"] = 0
        if increment:
            st["t"] += 1
        t = st["t"]
        ea, es = st["exp_avg"], st["exp_sq"]
        ea.mul_(b1).add_(g, alpha=1 - b1)
        es.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (es.sqrt() / (1 - b2 ** t) ** 0.5).add_(eps)
        upd = (ea / (1 - b1 ** t)) / denom
        if grp["ddc_strength"] > 0.0 and p.ndim == 2 and min(p.shape) > 1:
            upd = _ddc_project(upd, p, grp["ddc_mode"], grp["ddc_strength"])
        if grp["weight_decay"]:
            p.mul_(1.0 - lr * grp["weight_decay"])
        p.add_(upd, alpha=-lr)
