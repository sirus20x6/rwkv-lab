"""
GQA -> MLA conversion via SVD initialization.

Given a single attention layer's GQA weights (q_proj, k_proj, v_proj, o_proj),
produce an MLA layer (DeepSeek-V2/V3 convention) whose forward pass
approximates the original. A short finetune then recovers the residual gap.

Shapes follow PyTorch nn.Linear layout: weight is [out_features, in_features].
Internally we transpose to [in, out] because the math is cleaner that way.

Conventions for the produced MLA state_dict (names match DeepSeek HF impl):
    kv_a_proj_with_mqa : [R + D_rope, H]   hidden -> [kv_latent | k_rope_shared]
    kv_a_layernorm     : [R]               RMSNorm over kv_latent (init to 1)
    kv_b_proj          : [Nh*(D_nope+D_v), R]   kv_latent -> per-head [k_nope | v]
    q_proj             : [Nh*(D_nope+D_rope), H]   (uncompressed Q variant)
    o_proj             : [H, Nh*D_v]

If q_lora_rank is set (compressed Q variant):
    q_a_proj           : [Rq, H]
    q_a_layernorm      : [Rq]
    q_b_proj           : [Nh*(D_nope+D_rope), Rq]
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GQAConfig:
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    # Qwen-style extras. Safe defaults = plain GQA / DeepSeek layout.
    has_output_gate: bool = False           # q_proj outputs 2*Nh*head_dim, per-head [Q, gate]
    has_qk_norm: bool = False               # RMSNorm(head_dim) applied to Q and K per head
    rope_position: str = "last"             # "first" (Qwen) or "last" (DeepSeek)


@dataclass
class MLAConfig:
    hidden_size: int
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    q_lora_rank: Optional[int] = None
    has_output_gate: bool = False           # carry the sigmoid-gate through to MLA
    has_qk_norm: bool = False               # carry q_norm/k_norm through to MLA
    num_kv_rope_heads: int = 1              # 1=canonical MLA (shared),
                                            # Nkv=preserves GQA K_rope fidelity (recommended for Qwen)


def _expand_gqa(w: torch.Tensor, n_kv: int, n_q: int, d: int) -> torch.Tensor:
    """Repeat GQA KV heads to match query head count. w: [H, n_kv*d] -> [H, n_q*d]."""
    H = w.shape[0]
    n_rep = n_q // n_kv
    return (
        w.view(H, n_kv, d)
        .unsqueeze(2)
        .expand(H, n_kv, n_rep, d)
        .reshape(H, n_q * d)
        .contiguous()
    )


def _truncated_svd(W: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A, B) such that A @ B approximates W with rank `rank`.
    A: [H, rank], B: [rank, N]. Singular values folded into A."""
    W32 = W.float()
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    A = U[:, :rank] * S[:rank].unsqueeze(0)
    B = Vh[:rank, :]
    return A, B


def gqa_to_mla_svd(
    gqa_sd: dict,
    gqa_cfg: GQAConfig,
    mla_cfg: MLAConfig,
    *,
    q_key: str = "q_proj.weight",
    k_key: str = "k_proj.weight",
    v_key: str = "v_proj.weight",
    o_key: str = "o_proj.weight",
    head_expand_noise_std: float = 1e-3,
    rng_seed: int = 0,
    balance_kv: Optional[float] = None,
) -> dict:
    """
    Initialize an MLA attention layer from GQA weights via SVD.

    If mla_cfg.num_heads > gqa_cfg.num_q_heads, expand heads by duplication:
    each original head becomes N replicas (with small Gaussian noise on Q/UK/UV
    to break symmetry during training), and W_O is scaled by 1/N so the forward
    pass matches the original GQA at initialization exactly.

    balance_kv (BKV, TransMLA NeurIPS 2025):
        When provided, the scalar K/V L2-norm ratio (||K_nope|| / ||V||,
        typically measured from activations on calibration data — see
        build_bkv_stats.py). Rescales W_K_nope by 1/ratio before the joint
        SVD so K and V contribute equally to principal components. W_UK is
        then multiplied by ratio on the way out, so the forward pass is
        exactly preserved at full rank — but when the SVD truncates to R,
        V's principal directions survive instead of being dominated by K's
        larger norm. Paper: MHA2MLA had 21.85% accuracy drop at 93% KV
        compression without BKV; TransMLA had 1.65% with BKV.
    """
    assert mla_cfg.num_heads % gqa_cfg.num_q_heads == 0, (
        "mla_cfg.num_heads must be an integer multiple of gqa_cfg.num_q_heads"
    )
    assert mla_cfg.qk_nope_head_dim + mla_cfg.qk_rope_head_dim == gqa_cfg.head_dim, (
        "nope + rope dims must equal original head_dim for a clean carve-out"
    )
    assert mla_cfg.v_head_dim == gqa_cfg.head_dim, (
        "v_head_dim must equal original head_dim so W_O can be reused as-is"
    )

    assert not (mla_cfg.has_output_gate and not gqa_cfg.has_output_gate), (
        "can't synthesize a gate that wasn't in the source"
    )
    assert not (mla_cfg.has_qk_norm and not gqa_cfg.has_qk_norm), (
        "can't synthesize qk_norm that wasn't in the source"
    )
    assert gqa_cfg.rope_position in ("first", "last")

    H = mla_cfg.hidden_size
    Nq = gqa_cfg.num_q_heads
    Nh = mla_cfg.num_heads
    expand = Nh // Nq
    D_nope, D_rope, D_v = mla_cfg.qk_nope_head_dim, mla_cfg.qk_rope_head_dim, mla_cfg.v_head_dim
    R = mla_cfg.kv_lora_rank
    dtype = gqa_sd[k_key].dtype
    gen = torch.Generator().manual_seed(rng_seed)

    # Transpose to [in, out]
    # Q layout differs when has_output_gate: per-head interleaved [Q | gate],
    # so q_proj output dim is 2 * Nq * head_dim (per Qwen3_5Moe convention).
    W_Q_raw = gqa_sd[q_key].T.contiguous()        # [H, Nq*head_dim] or [H, 2*Nq*head_dim]
    W_K = gqa_sd[k_key].T.contiguous()
    W_V = gqa_sd[v_key].T.contiguous()
    W_O = gqa_sd[o_key]                           # [H, Nq*D_v]

    if gqa_cfg.has_output_gate:
        # W_Q_raw's per-head layout is [Q_head (head_dim), gate_head (head_dim)].
        W_Q_interleaved = W_Q_raw.view(H, Nq, 2, gqa_cfg.head_dim)
        W_Q = W_Q_interleaved[:, :, 0, :].reshape(H, Nq * gqa_cfg.head_dim).contiguous()
        W_gate = W_Q_interleaved[:, :, 1, :].reshape(H, Nq * gqa_cfg.head_dim).contiguous()
    else:
        W_Q = W_Q_raw
        W_gate = None

    # Expand GQA -> MHA view (one K/V per *original* query head)
    W_K_full = _expand_gqa(W_K, gqa_cfg.num_kv_heads, Nq, gqa_cfg.head_dim)
    W_V_full = _expand_gqa(W_V, gqa_cfg.num_kv_heads, Nq, gqa_cfg.head_dim)

    # Slice K per-head into nope / rope, respecting rope_position.
    W_K_heads = W_K_full.view(H, Nq, gqa_cfg.head_dim)
    if gqa_cfg.rope_position == "first":
        W_K_rope_per_head = W_K_heads[:, :, :D_rope]          # [H, Nq, D_rope]
        W_K_nope = W_K_heads[:, :, D_rope:].reshape(H, Nq * D_nope).contiguous()
    else:
        W_K_nope = W_K_heads[:, :, :D_nope].reshape(H, Nq * D_nope).contiguous()
        W_K_rope_per_head = W_K_heads[:, :, D_nope:]

    # K_rope handling. Canonical MLA shares one rope key across all heads, but
    # for GQA sources whose KV heads carry distinct rope signals (Qwen: Nkv=2
    # with near-orthogonal rope slices), averaging destroys information.
    # If num_kv_rope_heads == Nkv, preserve all KV heads' rope slices.
    # If == 1, average. Otherwise raise.
    Nkr = mla_cfg.num_kv_rope_heads
    if Nkr == 1:
        W_KR = W_K_rope_per_head.mean(dim=1)                      # [H, D_rope]
    elif Nkr == gqa_cfg.num_kv_heads:
        # Take the original (pre-expansion) per-KV-head rope slices.
        W_K_raw_heads = W_K.view(H, gqa_cfg.num_kv_heads, gqa_cfg.head_dim)
        if gqa_cfg.rope_position == "first":
            W_KR = W_K_raw_heads[:, :, :D_rope].reshape(H, Nkr * D_rope)   # [H, Nkr*D_rope]
        else:
            W_KR = W_K_raw_heads[:, :, D_rope:].reshape(H, Nkr * D_rope)
    else:
        raise ValueError(
            f"num_kv_rope_heads must be 1 or gqa_cfg.num_kv_heads ({gqa_cfg.num_kv_heads}), "
            f"got {Nkr}"
        )

    # Joint low-rank factorization of [K_nope | V] at ORIGINAL head count.
    # If balance_kv is provided, scale K_nope down by `ratio` before SVD so the
    # principal components aren't dominated by K's larger activation norm;
    # rescale W_UK up by `ratio` after to preserve the forward pass exactly
    # (at full rank). At truncated rank R, the effect is that V's principal
    # directions survive instead of being clobbered — the core BKV trick.
    if balance_kv is not None and balance_kv > 0:
        W_K_nope_for_svd = W_K_nope / float(balance_kv)
    else:
        W_K_nope_for_svd = W_K_nope
    W_kv_combined = torch.cat([W_K_nope_for_svd, W_V_full], dim=1)  # [H, Nq*(D_nope+D_v)]
    W_DKV, W_up = _truncated_svd(W_kv_combined, R)  # [H, R], [R, Nq*(D_nope+D_v)]
    W_UK_orig = W_up[:, : Nq * D_nope]
    W_UV_orig = W_up[:, Nq * D_nope :]
    if balance_kv is not None and balance_kv > 0:
        # Undo the K rescaling in the up-projection: forward now reconstructs
        # W_K_nope ≈ W_DKV @ (W_UK_orig * ratio) — original scale.
        W_UK_orig = W_UK_orig * float(balance_kv)

    # Q per-head, respecting rope_position.
    W_Q_heads = W_Q.view(H, Nq, gqa_cfg.head_dim)
    if gqa_cfg.rope_position == "first":
        W_Q_rope_orig = W_Q_heads[:, :, :D_rope]
        W_Q_nope_orig = W_Q_heads[:, :, D_rope:]
    else:
        W_Q_nope_orig = W_Q_heads[:, :, :D_nope]
        W_Q_rope_orig = W_Q_heads[:, :, D_nope:]

    # ---- Head expansion: replicate per-head slices N times, add symmetry-breaking noise ----
    def _expand_and_noise(w_per_head: torch.Tensor, head_axis: int, noise_std: float) -> torch.Tensor:
        # Duplicate N=expand times along `head_axis`, add Gaussian noise to the duplicates.
        base = w_per_head.unsqueeze(head_axis + 1)  # insert replica axis after head axis
        shape = list(base.shape)
        shape[head_axis + 1] = expand
        expanded = base.expand(*shape).contiguous()
        if noise_std > 0 and expand > 1:
            noise = (
                torch.randn(expanded.shape, generator=gen)
                .to(expanded.device, expanded.dtype)
                * noise_std
            )
            # Keep the first replica un-noised so at zero-noise-limit we recover exact init.
            noise[(slice(None),) * (head_axis + 1) + (0,)] = 0
            expanded = expanded + noise
        # Merge head axis and replica axis -> new head axis of size Nh
        new_shape = list(w_per_head.shape)
        new_shape[head_axis] = w_per_head.shape[head_axis] * expand
        return expanded.reshape(*new_shape)

    W_UK = _expand_and_noise(
        W_UK_orig.view(R, Nq, D_nope), head_axis=1, noise_std=head_expand_noise_std
    ).contiguous()
    W_UV = _expand_and_noise(
        W_UV_orig.view(R, Nq, D_v), head_axis=1, noise_std=head_expand_noise_std
    ).contiguous()
    W_Q_nope = _expand_and_noise(W_Q_nope_orig, head_axis=1, noise_std=head_expand_noise_std).contiguous()
    W_Q_rope = _expand_and_noise(W_Q_rope_orig, head_axis=1, noise_std=head_expand_noise_std).contiguous()

    # Gate is expanded head-wise too. The gate's effect is multiplicative on the
    # per-head attention output, so we do NOT scale it by 1/expand (unlike W_O):
    # each replica head should still produce a sensibly-scaled gate signal.
    if W_gate is not None:
        W_gate_heads = W_gate.view(H, Nq, gqa_cfg.head_dim)
        W_gate_expanded = _expand_and_noise(
            W_gate_heads, head_axis=1, noise_std=head_expand_noise_std
        ).contiguous()  # [H, Nh, head_dim]
    else:
        W_gate_expanded = None

    out: dict = {}

    # --- KV path ---
    kv_a = torch.cat([W_DKV, W_KR.float()], dim=1)  # [H, R + Nkr*D_rope]
    out["kv_a_proj_with_mqa.weight"] = kv_a.T.contiguous().to(dtype)

    kv_b_heads = torch.cat([W_UK, W_UV], dim=2)  # [R, Nh, D_nope+D_v]
    out["kv_b_proj.weight"] = kv_b_heads.reshape(R, Nh * (D_nope + D_v)).T.contiguous().to(dtype)

    # --- Q path ---
    # Per-head layout order follows MLA's own rope_position convention. To keep the
    # MLA module's forward simple, we emit [rope_first | nope_rest] if rope_position
    # is "first" (matching Qwen), and [nope_first | rope_rest] if "last" (DeepSeek).
    mla_rope_position = gqa_cfg.rope_position
    if mla_rope_position == "first":
        q_per_head = torch.cat([W_Q_rope, W_Q_nope], dim=2)  # [H, Nh, D_rope+D_nope]
    else:
        q_per_head = torch.cat([W_Q_nope, W_Q_rope], dim=2)  # [H, Nh, D_nope+D_rope]
    q_full = q_per_head.reshape(H, Nh * (D_nope + D_rope)).contiguous()

    if mla_cfg.q_lora_rank is not None:
        Rq = mla_cfg.q_lora_rank
        W_DQ, W_UQ_full = _truncated_svd(q_full, Rq)
        out["q_a_proj.weight"] = W_DQ.T.contiguous().to(dtype)
        out["q_b_proj.weight"] = W_UQ_full.T.contiguous().to(dtype)
    else:
        out["q_proj.weight"] = q_full.T.contiguous().to(dtype)

    # Re-layout kv_b_proj too if rope_position=first: MLA's K per head is
    # [k_nope | k_rope_shared], not affected by internal rope_position since
    # k_rope is emitted separately via kv_a_proj_with_mqa. But we need to make
    # sure the MLA module's k_nope slice ordering matches: in DeepSeek convention,
    # per-head kv_b is [k_nope (D_nope) | v (D_v)] — stays the same. OK.

    # --- Gate projection (if present) ---
    if W_gate_expanded is not None and mla_cfg.has_output_gate:
        # gate_proj.weight: [Nh * head_dim, H]
        gate_flat = W_gate_expanded.reshape(H, Nh * gqa_cfg.head_dim)
        out["gate_proj.weight"] = gate_flat.T.contiguous().to(dtype)

    # --- QK-norm weights (if present) ---
    # q_norm/k_norm are per-head-dim RMSNorm applied to the full head vector
    # (RMS computed over all head_dim). Carry the weight as-is; the MLA module
    # applies it to the assembled per-head [nope|rope] vector before RoPE.
    # Reordering note: the MLA module's per-head layout may place rope before
    # nope (rope_position="first"). We must reorder the norm weight to match.
    if gqa_cfg.has_qk_norm and mla_cfg.has_qk_norm:
        q_norm_raw = gqa_sd[q_key.replace("q_proj.weight", "q_norm.weight")]
        k_norm_raw = gqa_sd[k_key.replace("k_proj.weight", "k_norm.weight")]
        # The MLA per-head layout mirrors the source's rope_position convention
        # (see mla_rope_position above): "first" -> [rope | nope] on both sides,
        # "last" -> [nope | rope] on both sides. Either way the norm weight
        # carries over unchanged.
        out["q_norm.weight"] = q_norm_raw.contiguous().to(dtype)
        out["k_norm.weight"] = k_norm_raw.contiguous().to(dtype)

    # --- Output projection ---
    W_O_heads = W_O.view(H, Nq, D_v)
    W_O_expanded = W_O_heads.unsqueeze(2).expand(H, Nq, expand, D_v).contiguous()
    W_O_new = (W_O_expanded / expand).reshape(H, Nh * D_v)
    out["o_proj.weight"] = W_O_new.contiguous().to(W_O.dtype)

    return out


# ---------------------------------------------------------------------------
# Verification: does the SVD init actually approximate the original GQA?
# We compare the KV reconstruction directly, which is the only lossy step
# (RoPE and softmax are identical by construction once RoPE is plumbed).
# ---------------------------------------------------------------------------

def reconstruction_error(
    gqa_sd: dict,
    mla_sd: dict,
    gqa_cfg: GQAConfig,
    mla_cfg: MLAConfig,
    *,
    batch_tokens: int = 512,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    H = mla_cfg.hidden_size
    Nq = gqa_cfg.num_q_heads
    Nh = mla_cfg.num_heads
    expand = Nh // Nq
    D_nope = mla_cfg.qk_nope_head_dim
    D_rope = mla_cfg.qk_rope_head_dim
    D_v = mla_cfg.v_head_dim
    R = mla_cfg.kv_lora_rank

    x = torch.randn(batch_tokens, H, dtype=torch.float32)

    # ---- Original GQA forward (no RoPE, no softmax — just the projections) ----
    W_K = gqa_sd["k_proj.weight"].float().T  # [H, n_kv*d]
    W_V = gqa_sd["v_proj.weight"].float().T
    W_K_full = _expand_gqa(W_K, gqa_cfg.num_kv_heads, Nq, gqa_cfg.head_dim)
    W_V_full = _expand_gqa(W_V, gqa_cfg.num_kv_heads, Nq, gqa_cfg.head_dim)
    K_gqa_full = x @ W_K_full  # [T, Nq*d]
    V_gqa_full = x @ W_V_full

    K_gqa_heads = K_gqa_full.view(batch_tokens, Nq, gqa_cfg.head_dim)
    if gqa_cfg.rope_position == "first":
        K_gqa_rope = K_gqa_heads[:, :, :D_rope]  # [T, Nq, D_rope]
        K_gqa_nope = K_gqa_heads[:, :, D_rope:].reshape(batch_tokens, Nq * D_nope)
    else:
        K_gqa_nope = K_gqa_heads[:, :, :D_nope].reshape(batch_tokens, Nq * D_nope)
        K_gqa_rope = K_gqa_heads[:, :, D_nope:]  # [T, Nq, D_rope]

    # ---- MLA forward. Take one replica per original head for head-by-head comparison. ----
    kv_a = mla_sd["kv_a_proj_with_mqa.weight"].float().T  # [H, R+Nkr*D_rope]
    kv_b = mla_sd["kv_b_proj.weight"].float().T           # [R, Nh*(D_nope+D_v)]
    kv_a_out = x @ kv_a
    kv_lat = kv_a_out[:, :R]
    Nkr = mla_cfg.num_kv_rope_heads
    k_rope_heads = kv_a_out[:, R:].view(batch_tokens, Nkr, D_rope)

    # Select the first replica of each original head (replica 0 is noise-free by construction).
    kv_b_heads = kv_b.view(R, Nq, expand, D_nope + D_v)[:, :, 0, :]  # [R, Nq, D_nope+D_v]
    UK = kv_b_heads[:, :, :D_nope].reshape(R, Nq * D_nope)
    UV = kv_b_heads[:, :, D_nope:].reshape(R, Nq * D_v)

    K_mla_nope = kv_lat @ UK
    V_mla = kv_lat @ UV
    # Expand rope heads to one per original query head, mirroring _expand_gqa's
    # KV->Q head mapping (each rope head serves Nq // Nkr consecutive q heads).
    K_mla_rope = (
        k_rope_heads.unsqueeze(2)
        .expand(batch_tokens, Nkr, Nq // Nkr, D_rope)
        .reshape(batch_tokens, Nq, D_rope)
    )

    def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).norm().item() / (b.norm().item() + 1e-9)

    return {
        "k_nope_rel_err": rel_err(K_mla_nope, K_gqa_nope),
        "v_rel_err":      rel_err(V_mla, V_gqa_full),
        "k_rope_rel_err": rel_err(K_mla_rope, K_gqa_rope),  # lossy: shared vs per-head
        "kv_rank":        R,
        "kv_full_rank":   min(H, Nq * (D_nope + D_v)),
        "head_expand":    expand,
    }


# ---------------------------------------------------------------------------
# KV singular-value spectrum utility: pick `kv_lora_rank` empirically.
#
# Returns the cumulative energy retained vs rank, plus the minimum rank needed
# to hit common thresholds. Call this on one real full-attention layer's
# weights before committing to a kv_lora_rank for the whole model.
# ---------------------------------------------------------------------------

def kv_rank_spectrum(
    gqa_sd: dict,
    gqa_cfg: GQAConfig,
    qk_nope_head_dim: int,
    *,
    k_key: str = "k_proj.weight",
    v_key: str = "v_proj.weight",
    thresholds: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99, 0.999),
) -> dict:
    D_nope = qk_nope_head_dim
    Nq = gqa_cfg.num_q_heads
    d = gqa_cfg.head_dim
    H = gqa_cfg.hidden_size
    D_v = d

    W_K = gqa_sd[k_key].float().T
    W_V = gqa_sd[v_key].float().T
    W_K_full = _expand_gqa(W_K, gqa_cfg.num_kv_heads, Nq, d)
    W_V_full = _expand_gqa(W_V, gqa_cfg.num_kv_heads, Nq, d)
    if gqa_cfg.rope_position == "first":
        W_K_nope = W_K_full.view(H, Nq, d)[:, :, d - D_nope:].reshape(H, Nq * D_nope)
    else:
        W_K_nope = W_K_full.view(H, Nq, d)[:, :, :D_nope].reshape(H, Nq * D_nope)
    W_combined = torch.cat([W_K_nope, W_V_full], dim=1)    # [H, Nq*(D_nope+D_v)]

    _, S, _ = torch.linalg.svd(W_combined, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    ranks = {}
    for t in thresholds:
        # Smallest k such that energy[k-1] >= t.
        idx = int((energy >= t).nonzero(as_tuple=True)[0][0].item()) + 1
        ranks[f"rank_at_{t:.3f}"] = idx
    return {
        "singular_values": S,
        "cumulative_energy": energy,
        "full_rank":        S.numel(),
        **ranks,
    }


# ---------------------------------------------------------------------------
# Smoke test with synthetic GQA weights.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    gqa = GQAConfig(hidden_size=2048, num_q_heads=16, num_kv_heads=4, head_dim=128)
    torch.manual_seed(0)
    gqa_sd = {
        "q_proj.weight": torch.randn(gqa.num_q_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "k_proj.weight": torch.randn(gqa.num_kv_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "v_proj.weight": torch.randn(gqa.num_kv_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "o_proj.weight": torch.randn(gqa.hidden_size, gqa.num_q_heads * gqa.head_dim) * 0.02,
    }

    # --- 1) Same-heads, low rank (the memory-saving config) ---
    mla = MLAConfig(
        hidden_size=2048, num_heads=16,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128,
        kv_lora_rank=512,
    )
    print("[1] same heads, R=512:")
    for k, v in reconstruction_error(gqa_sd, gqa_to_mla_svd(gqa_sd, gqa, mla), gqa, mla).items():
        print(f"    {k:20s} {v}")

    # --- 2) Same-heads, full rank (sanity: exact reconstruction) ---
    mla_full = MLAConfig(**{**mla.__dict__,
                            "kv_lora_rank": min(2048, 16 * (64 + 128))})
    print("\n[2] same heads, full R (should be ~0):")
    for k, v in reconstruction_error(gqa_sd, gqa_to_mla_svd(gqa_sd, gqa, mla_full), gqa, mla_full).items():
        print(f"    {k:20s} {v}")

    # --- 3) 2x head expansion (the quality-focused config): R stays = full
    #        and we check that picking out replica-0 of every expanded head
    #        recovers the original GQA behavior exactly. ---
    mla_2x = MLAConfig(**{**mla.__dict__, "num_heads": 32,
                          "kv_lora_rank": min(2048, 16 * (64 + 128))})
    sd_2x = gqa_to_mla_svd(gqa_sd, gqa, mla_2x, head_expand_noise_std=0.0)
    print("\n[3] 2x heads, full R, zero noise (should be ~0 on replica-0):")
    for k, v in reconstruction_error(gqa_sd, sd_2x, gqa, mla_2x).items():
        print(f"    {k:20s} {v}")
    # Shapes sanity
    print("    shapes:")
    for k, v in sd_2x.items():
        print(f"      {k:30s} {tuple(v.shape)}")

    # --- 4) Spectrum utility: tells you what rank is actually needed. ---
    print("\n[4] KV singular-value spectrum (random weights, so no low-rank structure):")
    spec = kv_rank_spectrum(gqa_sd, gqa, qk_nope_head_dim=64)
    for k, v in spec.items():
        if k.startswith("rank_at"):
            print(f"    {k:22s} {v}  / {spec['full_rank']}")


if __name__ == "__main__":
    _smoke()
