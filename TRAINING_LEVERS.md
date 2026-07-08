# `convert_train.py` — Training Levers Manual

Reference for the loop / recurrent-depth, grokking, spectral-optimizer, and distillation-
objective levers on the conversion trainer. The canonical entrypoint is
`python -m rwkv_lab.convert_train`; trainboard launches the same package module.

## Read this first
- **Everything defaults OFF / weight 0.** At default flags the trainer behaves exactly
  as the original — every lever below is opt-in. The only on-by-default item is
  diagnostics (`--log-grokking-metrics 1`).
- **None of this is validated on *your* model.** It is literature-backed and
  correctness-tested (the optimizers/losses run and reduce toy problems; CKA is provably
  rotation-invariant; etc.), but **not A/B-tested on the RWKV conversion**. Treat the
  "sensible default" column as a *starting point to sweep*, not a known optimum.
- **Live-tuning:** knobs marked **Live** can be changed mid-run from the trainboard
  live-tuning panel. The dashboard writes the value to `run_controls` in
  `dashboard/trainboard.db`; the trainer polls it every `--control-poll-every` steps
  (default 10) via `live_controls.py`. No restart needed.

---

## 1. Grokking / generalization levers
Encourage the student to reach the *generalizing* solution (and reach it fast), and keep
it from getting stuck memorizing. Background sweep: 2026 grokking literature.

| Flag | Default | What it does | Source | When to use |
|---|---|---|---|---|
| `--weight-decay` | `0.0` | Decoupled weight decay. Now applied **manually for all optimizers** (adamw/muonclip carry WD=0 internally), so it actually bites. Under RMSNorm it's largely *radial/inert* — prefer `--nuc-weight`. **Live** (`weight_decay`). | 2603.13331 | A mild constant compress pressure; keep small. |
| `--tail-weight-decay` / `--wd-tail-frac` | `0.0` / `0.3` | Weight decay ramped in **only over the last `wd_tail_frac`** of training (0 during the fit phase) — "fit first, compress later." **Live** (`tail_weight_decay`, `wd_tail_frac`). | 2605.04396, 2606.05863 | When constant WD fights the fit phase / good init. |
| `--nuc-weight` / `--nuc-every` | `0.0` / `1` | **Nuclear-norm (spectral) penalty** on the student's 2D mix matrices — the *tangential* low-rank pressure L2 can't give under RMSNorm. SVD per matrix, so amortize with `--nuc-every`. **Live** (`nuc_weight`, `nuc_every`). | 2606.04405 | The main "encourage low-rank/generalizing structure" lever for RMSNorm nets. Try ~1e-5. |
| `--grokfast` + `--grokfast-lamb` / `--grokfast-alpha` | `0` + `2.0` / `0.98` | **GrokFast**: low-pass-filter the gradient and amplify the slow-varying component → reach the generalizing solution sooner (collapses the delay). **Live** (`grokfast_lamb`, `grokfast_alpha`). | GrokFast (Lee 2024); used in 2604.20923 | The headline "best model fastest" accelerator. Pairs with Muon, fights schedulefree averaging. |
| `--readout-lr-mult` | `1.0` | Higher LR on the **readout group** (`out_proj`, and the codec once it unfreezes at consolidation) — closes the representation→behavior lag. Scheduled opts only. **Live** (`readout_lr_mult`). | 2604.13082 | When block-match is good but `lm_ce` lags. |
| `--fixed-trainset` / `--disjoint-eval` | `0` / `1` | Cache N windows and **cycle** them (reuse-data) → the memorize→generalize regime. **DIAGNOSTIC ONLY** — it *manufactures* the memorization plateau; do **not** use for production. `--disjoint-eval` keeps the eval pool disjoint. | 2605.09724, 2605.14659 | Studying the transition on a per-layer fit; never in a real run. |
| `--log-grokking-metrics` / `--grok-spec-every` | `1` / `0` | Emit `gen_gap` (held-out − train block-MSE) at eval + (every N steps) `wnorm_rms`, `stable_rank`. Pure observability. | 2602.18649, 2605.28975 | Leave on; set `--grok-spec-every 200` to watch rank collapse. |

---

## 2. Grokking autopilot — anti-collapse + memorization-stall escape
A reactive controller (off by default). Detects (a) **collapse** — held-out regressing
from its own best while train improves (late-stage un-grokking) — and (b) **stall** —
held-out not improving for N evals (stuck memorizing). On either it escalates
regularization + kicks GrokFast; on collapse it also **restores the best checkpoint**.

| Flag | Default | What it does | Source |
|---|---|---|---|
| `--grok-autopilot` | `0` | Master switch (EMA-best + collapse/stall recovery). | 2602.02859 |
| `--ap-ema-decay` | `0.999` | Weight-EMA decay; the averaged model resists post-grok wobble (non-schedulefree). | — |
| `--ap-collapse-thresh` | `0.02` | Held-out must exceed its own best by this for `--ap-patience` evals = collapse. | 2602.02859 |
| `--ap-patience` / `--ap-stall-patience` | `2` / `3` | Consecutive regressing / non-improving evals before acting. | — |
| `--ap-reg-mult` | `2.0` | Multiply `nuc_weight`/`weight_decay` on each recovery. | — |
| `--ap-restore-best` | `1` | On collapse, roll the live model back to `out/best/ckpt.pt` (+clear optimizer momentum). | 2606.21514 |
| `--ap-max-restarts` | `3` | Cap on escalations. | — |

The **dashboard detector** independently writes an `lr_scale=0.5` cool to the control
table on `anti_grokking_collapse` (warn, not SIGINT). Autopilot owns the structural moves.

---

## 3. Spectral optimizer — `--optimizer spectral_muon`
Orthogonalize the momentum (Newton–Schulz polar factor `UVᵀ` — "spectral flattening"):
larger usable LR + lower curvature penalty → ~2× over Adam on LLM-ish data. `spectral_muon`
runs the Muon update on 2D matrices and a built-in AdamW on everything else. **With no
`--sm-*` flags it is plain Muon.** Calibrated to this repo's `GuardedMuonClip` via
`--sm-scale 0.4` (so `--muon-lr 1e-4` transfers). Core: Muon (Jordan 2024), 2605.13079,
2606.04662.

| Flag | Default | What it does | Source | When to use |
|---|---|---|---|---|
| `--sm-plus-norm {none,row,col}` | `none` | **MUON+**: row/col-normalize the orthogonalized update (fixes finite-NS imbalance). Near-free. | 2602.21545 | Almost always — up to 37% faster pretrain, ~0 cost. Start `row`. |
| `--sm-equilibrate {none,R,C,RC}` | `none` | **MuonEq**: row/col equilibration *before* NS. Near-free. | 2603.28254 | Pair with MUON+; use `R` for hidden weights. |
| `--sm-second-moment` | `0` | **Muon²**: Adam-style 2nd-moment precondition *before* NS → ~40% fewer NS iters. | 2604.09967 | Scale/MoE/distributed where NS cost matters. |
| `--sm-row-uniform` | `0` | **Aurora**: equal-row-norm for tall matrices — fixes dead neurons in wide MLPs. | 2606.27715 | FFN-heavy / wide-MLP layers. |
| `--sm-spectral-power` | `0.0` | **Muon^p**: `U·Σ^p·Vᵀ` (0 = Muon). p≈1/3 helps **finetune**; hurts pretrain. SVD path. **Live** (`sm_spectral_power`). | 2606.13867 | Finetuning only. |
| `--sm-mona` / `--sm-mona-alpha` | `0` / `0.1` | **MONA**: Nesterov/curvature term in the gradient before NS. | 2605.26842 | Large MoE pretrain/SFT; tune α. |
| `--sm-cheap-cubic` | `0` | Odd-cubic NS schedule (~1/3 fewer matmuls; weaker orthogonalization). | 2606.00371 | Only if NS is the bottleneck and you can afford a small quality risk. |
| `--sm-nesterov` | `0` | Nesterov momentum on the Muon update. | — | Optional. |
| `--sm-ns-steps` / `--sm-scale` | `5` / `0.4` | NS iterations / update amplifier `scale·√(max_dim)`. **Live** (`sm_scale`). | — | Leave unless re-calibrating LR. |
| `--sm-ddc-strength` / `--sm-ddc-mode` | `0.0` / `both` | **DDC** (abelian subset): remove the fraction [0,1] of the update along the **per-channel rescale gauge** (dead directions). Resists over-training collapse, cleaner minima. **Live** (`ddc_strength`). | 2606.29176 | When you see over-training/anti-grokking collapse; the optimizer-level analog of the autopilot. |
| `--sm-rsav` / `--sm-rsav-c` / `--sm-rsav-cap` / `--sm-rsav-relax` | `0` / `1.0` / `0.2` / `0.0` | **SpecMuon RSAV**: a global scalar-auxiliary-variable `r` tracks gradient energy and gates the step size (ξ, capped) — energy-adaptive without per-param state. | 2602.16167 | Stability under noisy/nonstationary gradients; cheap. |
| `--sm-tile-size T` | `0` | **Hierarchical/tiled Muon**: block-diagonal Newton–Schulz over T×T tiles (scale → c·√T); replaces the min(m,n) NS cost with `T`. 0 = full-matrix. | 2606.27216 | Very wide matrices where NS is the bottleneck. |
| `--sm-da-muon` / `--sm-da-eta-max` / `--sm-da-r0` | `0` / `0.01` / `1e-3` | **Distance-Aware Muon**: per-matrix adaptive radius `η = clamp(r̄/√k, η_max)`, `r̄` = running-max ‖W−W₀‖. Adds a W₀ snapshot per matrix. | 2605.18999 | When a fixed LR under-/over-shoots across layers. |
| `--sm-aro` / `--sm-aro-iters` | `0` / `5` | **ARO-Sinkhorn**: replace NS orthogonalization with a learned rotation (orthogonal-Procrustes) + Sinkhorn base optimizer — a non-orthonormal update (a *mode*, not a knob; NS-on-NS is a no-op). Adds an m×m rotation state. | 2602.09006 | Experimental alternative to orthogonalization. |
| `--sm-ns-steps-final` | `0` | **Spectral-Scaling** guard: route the readout/output projection to a separate Muon group with THIS many NS steps (e.g. 10) so its fast-shrinking momentum stays orthonormalizable. Frontier-scale concern. | 2606.04058 | Large-scale runs where the final layer's spectrum collapses. |

> The full cross-matrix DDC and the rotation gauge need the model graph; the rotation
> gauge is ~N/A for RWKV (no softmax attention). What's implemented is the single-matrix
> per-channel-scale projection — the dominant effect for your architecture.

---

## 4. PC-Layer, LLR, and loop-level levers

| Flag | Default | What it does | Source | When to use |
|---|---|---|---|---|
| `--pc-layer` / `--pc-strength` | `0` / `1.0` | **PC-Layer**: reparameterize student Linears with polynomial spectral preconditioning (soft spectrum flatten + norm-recovery), **mergeable at inference** (zero inference cost). `pc_layer` = degree level (2–4 typical). **Live** (`pc_strength`). | 2606.06470 | Big win **under AdamW**; *marginal under Muon* (already does spectral control). You have the VRAM. |
| `--llr` / `--llr-smax` / `--llr-every` / `--llr-active-frac` | `0` / `5.0` / `200` / `0.2` | **LLR**: per-param-group LR multiplier from each group's spectral heavy-tail (Hill-α). **Live** (`llr_smax`). | 2605.22297 | **Consolidation stage** (many layer-groups) — ~N/A on a single layer. Up to 1.5×. |
| `--hyperball` | `0` | Project each 2D weight back to its initial Frobenius sphere each step — removes WD tuning. | 2606.16899 | Normalized transformers, long/over-trained runs. |
| `--muon-to-adamw-frac` | `0.0` | River-valley switch: at this fraction of training, switch a muon-family optimizer → AdamW for the refinement tail (Muon is a poor late refiner). | 2606.21514 | The last ~10–20% of a Muon run. |

---

## 4b. Loop / recurrent-depth levers (`LoopedRWKV`)

Weight-tied N-pass refinement wrapped around the RWKV layer. **All exact no-ops at init**
(zero-init gates ⇒ `--loop-count N` with everything else off is bit-identical to a single pass),
so any subset can be A/B'd. See [`looped_rwkv.py`](src/rwkv_lab/looped_rwkv.py).

| Flag | Default | What it does | Source |
|---|---|---|---|
| `--loop-count N` | `1` | N weight-tied refinement passes (recurrent depth). Refine on a pre-normed `input + running output`. | 2604.21106 |
| `--loop-gate` / `--loop-gate-cap` | `scalar` / `0` | Gate granularity (scalar/head/channel/factored) + soft spectral-radius cap on the loop's contribution. | OpenMythos |
| `--loop-index` | off | Per-pass zero-init learned input offset so the tied core can specialize each pass. | OpenMythos |
| `--loop-hyper K` | `0` | K≥2 hyper-connection lanes at the loop boundary (largest loop-capacity lever; φ 0.45→0.65). | 2409.19606 |
| `--loop-lora-rank R` | `0` | Per-pass LoRA on the shared core's linears (unshare loop weights), zero-init ⇒ no-op until trained. | (CART/MoDr) |
| `--loop-sample` | off | Per-step loop-count sampling (uniform / poisson) — dynamic beats fixed depth. | (4 recurrent-depth papers) |
| `--loop-iter-consist` | off | Equilibrium-internalization: pull each earlier iterate toward the (stop-grad) final one. | 2605.12466 |
| `--loop-iter-readout` | off | Supervise **every** loop iterate against the teacher target (not just the final). | 2606.24898 |
| `--loop-adaptive-halt` / `--loop-ponder-weight` / `--loop-ponder-prior` | off | PonderNet per-token adaptive depth: halt head → halt-weighted output + KL-to-geometric ponder loss. | 2107.05407 |
| `--loop-cart-anchor` / `--loop-cart-gate-init` | off / `4.0` | CART contractive LTI gate `out = σ(g)⊙out + inc` — carry term is a contraction, damping deep-loop drift. Mutually exclusive with `--loop-hyper`. | 2606.01495 |
| `--loop-deq` / `--loop-deq-window k` | off / `1` | DEQ / 1-step gradient: approach the fixed point detached (no BPTT, O(1) memory) then take k graded steps (Neumann-k). Same forward value; cheaper gradient. Pair with `--loop-cart-anchor`. Incompatible with halt/hyper/iter-consist. | 2506.21734, 2606.18206 |
| `--loop-fp-halt` / `--loop-fp-tol` / `--loop-fp-min-iters` / `--loop-fp-patience` / `--loop-fp-damp` | off / `1e-3` / `1` / `2` / `0.5` | FPRM fixed-point-residual halting: stop when `‖out−prev‖/‖out‖ < tol` (convergence-based, no learned head), with damped-patience for oscillation. Incompatible with adaptive-halt/hyper/deq. | 2606.18206 |
| `--loop-lr-mult` / `--loop-anneal-rw` / `--loop-anneal-steps` | `1.0` / off / — | LR multiplier for the loop gates (their own `rwkv_loop` group); optional residual-weight anneal-in schedule. | — |

---

## 5. Distillation objectives (cross-architecture, alignment-invariant)
Pointwise MSE `‖h_S−h_T‖²` assumes a shared basis the transformer/GDN teacher and RWKV
student **don't** have. These match **direction / structure / dynamics** instead, so the
basis & dim mismatch is tolerated without a learned projector. All operate on the
block-output pair you already compute (no new forward passes). Module: `distill_objectives.py`.

| Flag | Default | What it does | Source | When to use |
|---|---|---|---|---|
| `--w-cos` | `0.0` | Cosine block/state match — direction-invariant, scale-robust. **Live** (`w_cos`). | 2602.05262, 2606.26488 | Cheapest relational add; start ~0.1. |
| `--w-cka` | `0.0` | **CKA/Gram** match — invariant to orthogonal transform + isotropic scaling, dim-agnostic. **Live** (`w_cka`). | 2606.05682 (Mamba-validated) | The core cross-arch fix; start ~0.25. Watch `carry_fidelity`. |
| `--w-flow` | `0.0` | **PHF** transition-flow — match how features *move* along the sequence (direction + Gram), not where. **Live** (`w_flow`). | 2606.29340 | Best fit for the DMT rollout; start ~0.3. |
| `--w-bridge` / `--bridge-rank` | `0.0` / `8` | **OPRD-Bridge**: match in a **frozen low-rank PCA subspace** (rank ~8; higher degrades). **Live** (`w_bridge`). | 2606.06021 | When same-coordinate MSE fails from dim mismatch. |
| `--agreement-gate` | `0.0` | **Trust-region** gating: reweight block MSE by teacher-student cosine agreement, down-weighting diverged tokens. **Live** (`agreement_gate`). | 2606.01249 | Stabilizes compounding GDN→RWKV mismatch in the rollout. |
| `--distill-fidelity-log` | `0` | Emit `carry_fidelity` — **label-free cosine drift monitor** (≈<0.8 flags global-dynamics collapse even while block-MSE looks fine). | 2606.26488 | Leave on when doing relational matching; it's a dashboard panel. |

**Provided as helpers (not auto-wired — need extra plumbing):**
- `distill_objectives.entropy_gated_kl(...)` (EOPD, 2603.07079) — FKL where the teacher is
  uncertain, RKL else. Needs **teacher logits** (a second forward you don't currently do).
- `distill_objectives.taylor_calibrate_decay_bias / _value_rescale` (2606.16429) — closed-form
  gate **init** from teacher attention stats (88× better worst-case init PPL). The formulas
  are there; wiring needs your RWKV time-mix gate param names + the teacher's attention
  distance/entropy. The single biggest cross-arch lever if you wire it.

---

## 6. Live-tunable keys
Settable mid-run from the trainboard panel (whitelisted in both the trainer and
`dashboard/internal/server/livetune.go`):

```
w_lmce w_block w_smt w_dmt grad_clip lr_scale eval_every save_every log_every
weight_decay tail_weight_decay wd_tail_frac nuc_weight nuc_every
grokfast_lamb grokfast_alpha readout_lr_mult
sm_spectral_power sm_scale ddc_strength pc_strength llr_smax
w_cos w_cka w_flow w_bridge agreement_gate
```

---

## 7. Dashboard additions (`dashboard`)
- **Panels:** memory-activation (ROSA/Engram injection RMS), memorization-vs-generalization
  (`gen_gap`), plus `carry_fidelity` from the distill monitor.
- **Alerts:** `anti_grokking_collapse` (warn + auto-cool `lr_scale`, *not* SIGINT),
  `memory_path_dead` (recall path never activated), extending the existing
  `gnorm_spike`/`codec_collapse`/`ppl_regress`/`stall` set.
- Metrics ride in `train.jsonl` `extra_json` → no schema change. Rebuild the Go binary
  after pulling new whitelist keys: `cd dashboard && go build ./cmd/trainboard`.

---

## 8. Recommended configs (starting points — sweep before trusting)

**Conservative starter** (low-risk subset):
```
--optimizer spectral_muon --sm-plus-norm row --sm-equilibrate R   # near-free Muon wins
--w-cka 0.25 --w-cos 0.1 --agreement-gate 1                       # cross-arch relational matching
--distill-fidelity-log 1                                          # watch carry_fidelity -> 1
```
(Confirm `--muon-lr 1e-4` calibrates like your `GuardedMuonClip`; tune the `w-*` live.)

**Encourage-grokking / best-model-fastest add-on:**
```
--grokfast 1 --nuc-weight 1e-5 --grok-autopilot 1
```

**Consolidation stage (multi-layer):** add `--llr 1 --llr-smax 5`.

**Do NOT default-on:** `--fixed-trainset` (diagnostic; manufactures memorization),
`--sm-spectral-power` (finetune-only), `--sm-cheap-cubic` (quality risk), `--pc-layer`
(marginal under Muon).

**Sweep-first (regime/scale/weight-dependent):** `--sm-second-moment`, `--sm-row-uniform`,
`--sm-mona`, `--sm-ddc-strength`, `--tail-weight-decay`, `--muon-to-adamw-frac`,
`--hyperball`, and all the `--w-*` magnitudes.

---

## 9. Module map
| File | Role |
|---|---|
| `spectral_muon.py` | Configurable Muon optimizer: MUON+/MuonEq/Muon²/Aurora/Muon^p/MONA/DDC/cheap-cubic. |
| `distill_objectives.py` | Relational/cross-arch losses: cosine, CKA, flow(PHF), OPRD-Bridge, agreement-gate, carry-fidelity, entropy-gated-KL, Taylor-Calibrate helpers. |
| `grok_autopilot.py` | Anti-collapse + stall-escape recovery (EMA-best, restore-best, reg escalation). |
| `grokking_metrics.py` | `gen_gap` / `wnorm_rms` / `stable_rank` diagnostics. |
| `pc_layer.py` | PC-Layer parametrization (mergeable at inference). |
| `llr.py` | Heavy-tail layerwise-LR controller. |
| `live_controls.py` | Trainer-side consumer of the trainboard live-tune control table. |
