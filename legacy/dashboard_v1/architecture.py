"""Per-run model architecture breakdown.

Combines four data sources:
  1. The run's saved cfg (sidecar config.json next to each ckpt.pt, written
     by train_mla.save_checkpoint). Falls back to parsing /proc/<pid>/cmdline
     for live runs that haven't saved yet.
  2. The base model's HF config.json (from cfg.model_dir).
  3. The base model's safetensors metadata (model.safetensors.index.json +
     each shard's tensor shapes) — ground truth for per-tensor parameter
     counts. Cached per model_dir on first read.
  4. The patch manifest (from cfg.patch_dir) — provides MLA layer indices,
     MTP install state, dropped GQA tensor names and added MLA tensor names.

Cache key is (model_dir, mtime_of_index_json) — invalidates if model files
change. Tensor shape reads use safe_open's get_slice() which is metadata-only
(no data load). Each model is read once per dashboard process lifetime.

Returned dict is JSON-serializable.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import psutil

# safetensors is a hard dep of transformers; safe to import unconditionally
# in this venv. Wrap anyway for portability.
try:
    from safetensors import safe_open  # type: ignore
except ImportError:  # pragma: no cover
    safe_open = None

# Repo root — resolve relative paths from cfg (e.g. patch_dir="converted_9b_bkv_mtp")
# against this anchor instead of the dashboard's cwd.
REPO_ROOT = Path("/thearray/git/moe-mla")


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


# ---------------------------------------------------------------------------
# Cfg discovery (sidecar JSON > live cmdline).
# ---------------------------------------------------------------------------
def _latest_step_dir(run_dir: Path) -> Optional[Path]:
    if not run_dir.is_dir():
        return None
    cands = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if not cands:
        return None
    return max(cands, key=lambda d: d.stat().st_mtime)


def _read_sidecar_config(run_dir: Path) -> Optional[dict]:
    step_dir = _latest_step_dir(run_dir)
    if step_dir is None:
        return None
    cfg_path = step_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return None


def _arg_pairs(cmdline: list[str]) -> dict[str, str]:
    """--flag value / --flag=value parser. Last write wins."""
    out: dict[str, str] = {}
    i = 0
    while i < len(cmdline):
        a = cmdline[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
                out[k.replace("-", "_")] = v
            else:
                k = a[2:].replace("-", "_")
                if i + 1 < len(cmdline) and not cmdline[i + 1].startswith("--"):
                    out[k] = cmdline[i + 1]
                    i += 1
                else:
                    out[k] = "1"
        i += 1
    return out


_INT_FLAGS = {
    "install_mtp", "engram_enabled", "freeze_non_mla", "train_mla_only",
    "train_mtp_only", "train_aux_only", "train_engram_only",
    "guarded_muonclip", "prodigy_aux", "muon_exclude_embed_lmhead",
    "gradient_checkpointing",
}


def _live_run_cfg(run_name: str) -> Optional[dict]:
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = p.info.get("cmdline") or []
            if "train_mla" not in " ".join(cmdline):
                continue
            args = _arg_pairs(cmdline)
            out_dir = args.get("out_dir", "")
            if out_dir and Path(out_dir).name == run_name:
                cfg = {}
                for k, v in args.items():
                    if k in _INT_FLAGS:
                        try:
                            cfg[k] = int(v)
                        except ValueError:
                            cfg[k] = 0
                    else:
                        cfg[k] = v
                return cfg
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


_CKPT_CFG_CACHE: dict[tuple[str, int], dict] = {}


def _read_ckpt_config(run_dir: Path) -> Optional[dict]:
    """Last-resort fallback for runs whose ckpts predate the sidecar JSON.
    torch.load the latest ckpt.pt with weights_only=False just for its 'config'
    key. Slow (multi-second on a 17 GB ckpt) — cached on (path, mtime)."""
    step_dir = _latest_step_dir(run_dir)
    if step_dir is None:
        return None
    ckpt = step_dir / "ckpt.pt"
    if not ckpt.exists():
        return None
    cache_key = (str(ckpt), ckpt.stat().st_mtime_ns)
    if cache_key in _CKPT_CFG_CACHE:
        return _CKPT_CFG_CACHE[cache_key]
    try:
        import torch  # imported lazily so the dashboard can boot without GPU
        # weights_only=False is required because the saved config was an
        # arbitrary dict (asdict of a dataclass). Trusted source — these are
        # our own checkpoints.
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            _CKPT_CFG_CACHE[cache_key] = cfg
            return cfg
    except Exception:
        pass
    return None


def get_run_cfg(run_dir: Path) -> Optional[dict]:
    sc = _read_sidecar_config(run_dir)
    if sc is not None:
        return sc.get("config", sc)
    live = _live_run_cfg(run_dir.name)
    if live is not None:
        return live
    # Last resort: read from inside the ckpt.pt itself.
    return _read_ckpt_config(run_dir)


# ---------------------------------------------------------------------------
# Safetensors metadata cache: (model_dir, index.json mtime_ns) → tensor table.
# tensor table = list of (name, shape_tuple, numel).
# ---------------------------------------------------------------------------
_TENSOR_CACHE: dict[tuple[str, int], list[tuple[str, tuple[int, ...], int]]] = {}


def _read_safetensors_tensors(model_dir: Path) -> list[tuple[str, tuple[int, ...], int]]:
    """Return [(name, shape, numel), ...] for every tensor in the model's
    safetensors shards. Reads metadata only (no tensor data). Cached."""
    if safe_open is None:
        return []
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        cache_key = (str(model_dir), idx_path.stat().st_mtime_ns)
        if cache_key in _TENSOR_CACHE:
            return _TENSOR_CACHE[cache_key]
        idx = json.loads(idx_path.read_text())
        weight_map: dict[str, str] = idx.get("weight_map", {})
        # Group by shard so we open each file once.
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            by_shard[shard].append(name)
        out: list[tuple[str, tuple[int, ...], int]] = []
        for shard, names in by_shard.items():
            shard_path = model_dir / shard
            try:
                with safe_open(str(shard_path), framework="pt") as f:
                    for n in names:
                        try:
                            sl = f.get_slice(n)
                            shape = tuple(sl.get_shape())
                            numel = 1
                            for d in shape:
                                numel *= int(d)
                            out.append((n, shape, numel))
                        except Exception:
                            continue
            except Exception:
                continue
        _TENSOR_CACHE[cache_key] = out
        return out
    # Single-file fallback (model.safetensors with no index)
    single = model_dir / "model.safetensors"
    if not single.exists():
        return []
    cache_key = (str(model_dir), single.stat().st_mtime_ns)
    if cache_key in _TENSOR_CACHE:
        return _TENSOR_CACHE[cache_key]
    out = []
    try:
        with safe_open(str(single), framework="pt") as f:
            for n in f.keys():
                sl = f.get_slice(n)
                shape = tuple(sl.get_shape())
                numel = 1
                for d in shape:
                    numel *= int(d)
                out.append((n, shape, numel))
    except Exception:
        return []
    _TENSOR_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Layer grouping.
# ---------------------------------------------------------------------------
_RX_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(.+)$")
_RX_MTP = re.compile(r"^mtp\.(.+)$")


def _categorize_tensor(name: str) -> tuple[str, Optional[int], Optional[str]]:
    """Return (group, layer_idx, sub) for a tensor name. group is one of:
    'embed', 'lm_head', 'final_norm', 'decoder', 'mtp', 'engram',
    'vision', 'other'."""
    if "embed_tokens" in name:
        return "embed", None, None
    if name.endswith("lm_head.weight") or name.endswith(".lm_head.weight"):
        return "lm_head", None, None
    # Vision encoder: model.visual.* (Qwen3.5 multimodal). Loaded into
    # safetensors but not put on GPU during text-only training.
    if ".visual." in name or name.startswith("model.visual.") or name.startswith("visual."):
        return "vision", None, name
    m = _RX_LAYER.match(name.replace("model.language_model.", "model.").lstrip("."))
    # Try a more robust match against the full name.
    m2 = re.search(r"layers\.(\d+)\.(.+)$", name)
    if m2 and "mtp." not in name and "engram" not in name:
        idx = int(m2.group(1))
        sub = m2.group(2)
        return "decoder", idx, sub
    if name.startswith("mtp.") or "mtp." in name:
        return "mtp", None, None
    if "engram" in name.lower():
        em = re.search(r"engram[._]layer[._](\d+)|engram\.(\d+)\.|layers\.(\d+).*engram", name)
        idx = None
        if em:
            for g in em.groups():
                if g is not None:
                    idx = int(g)
                    break
        return "engram", idx, None
    if name.endswith("model.norm.weight") or name == "model.norm.weight":
        return "final_norm", None, None
    return "other", None, None


def _classify_decoder_sub(sub: str) -> str:
    """Within a decoder layer, classify sub-name into a sub-component."""
    s = sub
    if s.startswith("self_attn.") or s.startswith("attn.") or "self_attn" in s:
        return "attention"
    if s.startswith("linear_attn.") or "linear_attn" in s:
        return "linear_attn"
    if s.startswith("mlp.") or "mlp." in s:
        return "mlp"
    if "layernorm" in s.lower() or "norm" in s.lower():
        return "norm"
    return "other"


def _is_mla_tensor(sub: str) -> bool:
    """MLA-specific tensor names produced by convert.py."""
    return any(k in sub for k in ("q_a_proj", "q_b_proj", "kv_a_proj",
                                  "kv_b_proj", "q_a_norm", "kv_a_norm"))


# ---------------------------------------------------------------------------
# Patch + manifest.
# ---------------------------------------------------------------------------
def _load_model_config(model_dir: str) -> Optional[dict]:
    p = _resolve(model_dir) / "config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_patch_manifest(patch_dir: str) -> Optional[dict]:
    if not patch_dir:
        return None
    p = _resolve(patch_dir) / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_patch_tensors(patch_dir: str) -> list[tuple[str, tuple[int, ...], int]]:
    if not patch_dir or safe_open is None:
        return []
    p = _resolve(patch_dir) / "patch.safetensors"
    if not p.exists():
        return []
    cache_key = (str(p), p.stat().st_mtime_ns)
    if cache_key in _TENSOR_CACHE:
        return _TENSOR_CACHE[cache_key]
    out = []
    try:
        with safe_open(str(p), framework="pt") as f:
            for n in f.keys():
                sl = f.get_slice(n)
                shape = tuple(sl.get_shape())
                numel = 1
                for d in shape:
                    numel *= int(d)
                out.append((n, shape, numel))
    except Exception:
        return []
    _TENSOR_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Build the per-layer breakdown.
# ---------------------------------------------------------------------------
def _freeze_mode(cfg: dict) -> str:
    if int(cfg.get("train_mla_only", 0)):
        return "mla_only"
    if int(cfg.get("train_mtp_only", 0)):
        return "mtp_only"
    if int(cfg.get("train_aux_only", 0)):
        return "aux_only"
    if int(cfg.get("train_engram_only", 0)):
        return "engram_only"
    if int(cfg.get("train_rwkv8_only", 0)):
        return "rwkv8_only"
    if str(cfg.get("train_rwkv8_layers", "")).strip():
        return "rwkv8_layers"
    if int(cfg.get("freeze_non_mla", 1)) == 0:
        return "full_unfreeze"
    return "mla_default_freeze"


def _parse_rwkv8_layer_indices(cfg: dict) -> set[int]:
    """Parse the comma-sep `rwkv8_deltanet_layers` field from cfg into a set."""
    spec = cfg.get("rwkv8_deltanet_layers", "")
    if not spec:
        return set()
    out: set[int] = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return out


def _parse_train_rwkv8_layers(cfg: dict) -> set[int]:
    """Parse `train_rwkv8_layers` (comma-sep layer indices) into a set."""
    spec = cfg.get("train_rwkv8_layers", "")
    if not spec:
        return set()
    out: set[int] = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return out


def compute_architecture(cfg: dict, model_cfg: dict, model_dir: Path,
                         patch_manifest: Optional[dict],
                         patch_dir: Optional[Path]) -> dict:
    text_cfg = model_cfg.get("text_config", model_cfg)
    hidden = text_cfg["hidden_size"]
    n_layers = text_cfg["num_hidden_layers"]
    n_q = text_cfg.get("num_attention_heads", 32)
    n_kv = text_cfg.get("num_key_value_heads", n_q)
    head_dim = text_cfg.get("head_dim", hidden // n_q)
    intermediate = text_cfg.get("intermediate_size", 4 * hidden)
    vocab = text_cfg.get("vocab_size", 0) or model_cfg.get("vocab_size", 0)
    layer_types: list[str] = (text_cfg.get("layer_types") or
                              ["full_attention"] * n_layers)
    output_gate = bool(text_cfg.get("attn_output_gate", False))
    is_moe = "num_experts_per_tok" in text_cfg
    n_experts = text_cfg.get("num_experts", 0) if is_moe else 0
    n_experts_per_tok = text_cfg.get("num_experts_per_tok", 0) if is_moe else 0
    moe_intermediate = text_cfg.get("moe_intermediate_size", intermediate) if is_moe else None
    tie_word_embeddings = bool(model_cfg.get("tie_word_embeddings", False))

    mla_layer_indices: list[int] = []
    mtp_installed_in_patch = False
    if patch_manifest is not None:
        mla_layer_indices = list(patch_manifest.get("full_attn_layer_indices", []))
        mtp_installed_in_patch = bool(patch_manifest.get("mtp_converted", False))
    install_mtp = bool(int(cfg.get("install_mtp", 0)))
    mtp_in_use = install_mtp and mtp_installed_in_patch

    # RWKV-8: linear-attention layers may be runtime-replaced with an RWKV-8
    # module. The swap happens in load_converted_model() and isn't reflected
    # in safetensors metadata, so we read it from cfg instead.
    rwkv8_layer_indices = _parse_rwkv8_layer_indices(cfg)
    rwkv8_swap_mode = str(cfg.get("rwkv8_swap_mode", "timemix"))
    rwkv8_train_only_layers = _parse_train_rwkv8_layers(cfg)

    # Read base model tensors (cached). For MLA-converted layers, replace
    # the GQA tensors with the patch's MLA tensors.
    base_tensors = _read_safetensors_tensors(model_dir)
    patch_tensors = _load_patch_tensors(str(patch_dir) if patch_dir else "")

    dropped_keys: set[str] = set(patch_manifest.get("dropped_keys", [])) if patch_manifest else set()
    # Effective tensor list = base tensors with dropped_keys removed, plus patch tensors.
    eff: list[tuple[str, tuple[int, ...], int]] = []
    for n, shape, numel in base_tensors:
        if n in dropped_keys:
            continue
        eff.append((n, shape, numel))
    for n, shape, numel in patch_tensors:
        eff.append((n, shape, numel))

    # Group tensors per layer/component.
    group_totals: dict[tuple[str, Optional[int]], int] = defaultdict(int)
    decoder_breakdown: dict[int, dict] = defaultdict(lambda: {
        "attention": 0, "linear_attn": 0, "mlp": 0, "norm": 0, "other": 0,
        "mla_tensors": False,
        "has_conv1d": False,        # Gated DeltaNet signature
        "has_linear_inproj": False, # generic linear-attn signature
    })
    detail_tensors: dict[tuple[str, Optional[int]], list[tuple[str, str, int]]] = defaultdict(list)

    for name, shape, numel in eff:
        group, idx, sub = _categorize_tensor(name)
        group_totals[(group, idx)] += numel
        detail_tensors[(group, idx)].append((name, "x".join(str(d) for d in shape), numel))
        if group == "decoder" and sub is not None and idx is not None:
            sc = _classify_decoder_sub(sub)
            decoder_breakdown[idx][sc] += numel
            if sc == "attention" and _is_mla_tensor(sub):
                decoder_breakdown[idx]["mla_tensors"] = True
            if sc == "linear_attn":
                if "conv1d" in sub:
                    decoder_breakdown[idx]["has_conv1d"] = True
                if "in_proj_" in sub:
                    decoder_breakdown[idx]["has_linear_inproj"] = True

    fmode = _freeze_mode(cfg)

    def trainable_for(kind: str) -> bool:
        if fmode == "full_unfreeze":
            return True
        if fmode == "mla_only":
            return kind == "mla_module"
        if fmode == "mla_default_freeze":
            return kind == "mla_module"
        if fmode == "mtp_only":
            return kind == "mtp"
        if fmode == "aux_only":
            return kind == "aux_head"
        if fmode == "engram_only":
            return kind == "engram"
        return True

    layers: list[dict] = []

    # Embedding
    embed_p = group_totals.get(("embed", None), 0)
    layers.append({
        "kind": "embedding",
        "name": "model.embed_tokens",
        "params": embed_p,
        "shape": f"{vocab}×{hidden}" if vocab else "",
        "trainable": (fmode == "full_unfreeze"),
        "tensors": detail_tensors.get(("embed", None), []),
    })

    # Decoder layers
    for i in range(n_layers):
        bd = decoder_breakdown.get(i, {"attention": 0, "linear_attn": 0,
                                        "mlp": 0, "norm": 0, "other": 0,
                                        "mla_tensors": False,
                                        "has_conv1d": False,
                                        "has_linear_inproj": False})
        is_mla = (i in mla_layer_indices) or bd["mla_tensors"]
        is_rwkv8 = (i in rwkv8_layer_indices)
        ltype_cfg = layer_types[i] if i < len(layer_types) else "full_attention"
        if is_mla:
            attn_kind = "MLA"
            attn_params = bd["attention"]
        elif is_rwkv8:
            # Runtime-swapped RWKV-8 module replacing a linear_attention slot.
            # Param count from safetensors still reflects the (now-discarded)
            # DeltaNet tensors; trained weights live in the run's ckpt.
            attn_kind = "RWKV-8 TimeMix" if rwkv8_swap_mode == "timemix" else "RWKV-8 ChannelMix"
            attn_params = bd["linear_attn"] if bd["linear_attn"] else bd["attention"]
        elif ltype_cfg == "linear_attention" or bd["linear_attn"] > 0:
            # Gated DeltaNet signature: short causal Conv1d on the QKV stream
            # plus low-rank a/b projections for the recurrent state. If we see
            # both, label as "DeltaNet". If only in_proj_* without conv1d,
            # fall back to a generic "LinearAttn" label.
            if bd["has_conv1d"] and bd["has_linear_inproj"]:
                attn_kind = "DeltaNet"
            elif bd["has_linear_inproj"]:
                attn_kind = "LinearAttn"
            else:
                attn_kind = "linear_attention"
            attn_params = bd["linear_attn"] if bd["linear_attn"] else bd["attention"]
        else:
            attn_kind = "GQA"
            attn_params = bd["attention"]

        # Trainability
        if fmode == "full_unfreeze":
            attn_t = mlp_t = norm_t = True
            tr_state = "trainable"
        elif fmode in ("mla_only", "mla_default_freeze"):
            attn_t = is_mla
            mlp_t = norm_t = False
            tr_state = "partial" if is_mla else "frozen"
        elif fmode == "rwkv8_only":
            attn_t = is_rwkv8
            mlp_t = norm_t = False
            tr_state = "partial" if is_rwkv8 else "frozen"
        elif fmode == "rwkv8_layers":
            attn_t = is_rwkv8 and (i in rwkv8_train_only_layers)
            mlp_t = norm_t = False
            tr_state = "partial" if attn_t else "frozen"
        else:
            attn_t = mlp_t = norm_t = False
            tr_state = "frozen"

        total = bd["attention"] + bd["linear_attn"] + bd["mlp"] + bd["norm"] + bd["other"]

        layers.append({
            "kind": "decoder_layer",
            "index": i,
            "name": f"model.layers.{i}",
            "layer_type": ltype_cfg,
            "is_mla": is_mla,
            "is_rwkv8": is_rwkv8,
            "attention": {
                "kind": attn_kind,
                "n_q_heads": n_q,
                "n_kv_heads": n_kv if attn_kind != "MLA" else None,
                "head_dim": head_dim,
                "params": attn_params,
                "trainable": attn_t,
            },
            "mlp": {
                "kind": "MoE" if is_moe else "dense",
                "intermediate_size": moe_intermediate or intermediate,
                "n_experts": n_experts if is_moe else None,
                "n_experts_per_tok": n_experts_per_tok if is_moe else None,
                "params": bd["mlp"],
                "trainable": mlp_t,
            },
            "norm_params": bd["norm"],
            "other_params": bd["other"],
            "params": total,
            "trainable_state": tr_state,
        })

    # Final norm
    final_norm_p = group_totals.get(("final_norm", None), 0)
    layers.append({
        "kind": "norm",
        "name": "model.norm",
        "params": final_norm_p,
        "trainable": (fmode == "full_unfreeze"),
    })

    # lm_head (if not tied)
    lm_head_p = group_totals.get(("lm_head", None), 0)
    if lm_head_p > 0 or not tie_word_embeddings:
        layers.append({
            "kind": "lm_head",
            "name": "lm_head",
            "params": lm_head_p,
            "shape": f"{hidden}×{vocab}" if vocab else "",
            "trainable": (fmode == "full_unfreeze"),
        })

    # MTP
    if mtp_in_use:
        mtp_p = group_totals.get(("mtp", None), 0)
        layers.append({
            "kind": "mtp",
            "name": "mtp",
            "params": mtp_p,
            "trainable": fmode in ("full_unfreeze", "mtp_only"),
            "tensors_count": len(detail_tensors.get(("mtp", None), [])),
        })

    # Engram
    engram_layer_indices: list[int] = []
    if int(cfg.get("engram_enabled", 0)):
        eng_dir = cfg.get("engram_patch_dir", "")
        eng_manifest = _load_patch_manifest(eng_dir) if eng_dir else None
        if eng_manifest:
            engram_layer_indices = list(eng_manifest.get("layer_indices", []))
        eng_per_module = 0
        # Engram tensors live in the engram patch (separate from MLA patch); we
        # don't load them here. Read approx from manifest if present.
        if eng_manifest:
            eng_per_module = int(eng_manifest.get("approx_params_per_layer", 0)) or 0
            # Fallback: try the engram patch.safetensors.
            if eng_per_module == 0:
                eng_tensors = _load_patch_tensors(eng_dir)
                if eng_tensors and engram_layer_indices:
                    total_eng = sum(t[2] for t in eng_tensors)
                    eng_per_module = total_eng // max(1, len(engram_layer_indices))
        for li in engram_layer_indices:
            layers.append({
                "kind": "engram",
                "layer_id": li,
                "name": f"engram.layer_{li}",
                "params": eng_per_module,
                "params_approx": True,
                "trainable": fmode in ("full_unfreeze", "engram_only"),
            })

    # Vision encoder (Qwen3.5 multimodal, not loaded for text-only training)
    vision_p = group_totals.get(("vision", None), 0)
    if vision_p > 0:
        v_tensors = detail_tensors.get(("vision", None), [])
        # Top-10 largest tensors so the user can verify what's in there
        top = sorted(v_tensors, key=lambda t: -t[2])[:10]
        layers.append({
            "kind": "vision",
            "name": "model.visual (vision encoder)",
            "params": vision_p,
            "trainable": False,
            "loaded_in_training": False,  # train_mla.py doesn't load this branch
            "tensors_count": len(v_tensors),
            "top_tensors": [{"name": n, "shape": s, "params": p} for n, s, p in top],
            "note": "vision encoder — present in safetensors but not loaded for text-only training",
        })

    # Other truly-unaccounted tensors (should be empty for known model types)
    other_p = sum(v for (g, _), v in group_totals.items() if g == "other")
    if other_p > 0:
        o_tensors = detail_tensors.get(("other", None), [])
        top = sorted(o_tensors, key=lambda t: -t[2])[:10]
        layers.append({
            "kind": "other",
            "name": "(unaccounted tensors)",
            "params": other_p,
            "trainable": False,
            "tensors_count": len(o_tensors),
            "top_tensors": [{"name": n, "shape": s, "params": p} for n, s, p in top],
        })

    # Totals
    total_params = 0
    trainable_params = 0
    for L in layers:
        if L["kind"] == "decoder_layer":
            attn_t = L["attention"]["trainable"]
            mlp_t = L["mlp"]["trainable"]
            tr = (L["attention"]["params"] if attn_t else 0)
            tr += (L["mlp"]["params"] if mlp_t else 0)
            if attn_t and mlp_t:
                tr += L["norm_params"] + L["other_params"]
            total_params += L["params"]
            trainable_params += tr
        else:
            total_params += L["params"]
            if L.get("trainable"):
                trainable_params += L["params"]

    return {
        "model_name": Path(cfg.get("model_dir", "unknown")).name,
        "model_type": model_cfg.get("model_type", "unknown"),
        "is_moe": is_moe,
        "config": {
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "n_experts": n_experts if is_moe else None,
            "n_experts_per_tok": n_experts_per_tok if is_moe else None,
            "tie_word_embeddings": tie_word_embeddings,
            "attn_output_gate": output_gate,
        },
        "modifications": {
            "mla_layer_indices": mla_layer_indices,
            "mtp_installed": mtp_in_use,
            "engram_layer_indices": engram_layer_indices,
            "rwkv8_layer_indices": sorted(rwkv8_layer_indices),
            "rwkv8_swap_mode": rwkv8_swap_mode if rwkv8_layer_indices else None,
            "rwkv8_train_only_layers": sorted(rwkv8_train_only_layers) if rwkv8_train_only_layers else None,
            "freeze_mode": fmode,
        },
        "totals": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_pct": (100.0 * trainable_params / total_params) if total_params else 0.0,
        },
        "layers": layers,
        "ts": time.time(),
    }


def architecture_for_run(run_dir: Path) -> dict:
    cfg = get_run_cfg(run_dir)
    if cfg is None:
        return {"error": "no run config available (no sidecar config.json and "
                         "no live train_mla.py process for this run)"}
    model_dir_s = cfg.get("model_dir", "")
    if not model_dir_s:
        return {"error": "run cfg has no model_dir"}
    model_dir = _resolve(model_dir_s)
    model_cfg = _load_model_config(model_dir_s)
    if model_cfg is None:
        return {"error": f"could not load {model_dir}/config.json"}
    patch_dir_s = cfg.get("patch_dir", "")
    patch_dir = _resolve(patch_dir_s) if patch_dir_s else None
    patch_manifest = _load_patch_manifest(patch_dir_s) if patch_dir_s else None
    return compute_architecture(cfg, model_cfg, model_dir, patch_manifest, patch_dir)
