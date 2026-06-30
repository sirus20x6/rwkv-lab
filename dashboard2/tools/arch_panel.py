#!/usr/bin/env python
"""Render the architecture panel HTML for one run (trainboard v2), matching the
v1 look: config pills + modification tags + a trainable bar, then an expandable
per-layer list (type pill, params, trainable badge, and an attn/mlp/norm detail
on expand — native <details>, no JS).

Reuses the v1 analyzer (legacy/dashboard_v1/architecture.py — safetensors
metadata-only, sidecar-config trainable inference). Usage: arch_panel.py <run_dir>
"""
import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "legacy" / "dashboard_v1"))
from architecture import architecture_for_run  # noqa: E402


def fmt_params(n):
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def tr_badge(state):
    if state in (True, "trainable"):
        return '<span class="tr-badge tr-on">trainable</span>'
    if state == "partial":
        return '<span class="tr-badge tr-partial">partial</span>'
    return '<span class="tr-badge tr-off">frozen</span>'


def type_class(kind):
    return re.sub(r"[^a-z0-9]", "", (kind or "").lower())


def render(d):
    if d.get("error"):
        return f'<div id="arch-body" class="arch-body"><div class="empty">{html.escape(str(d["error"]))}</div></div>'
    t = d["totals"]
    cfg = d.get("config", {})
    m = d.get("modifications", {})
    o = ['<div id="arch-body" class="arch-body">', '<div class="arch-summary">']

    # config pills
    spans = [f'<span><b>{html.escape(d.get("model_name", "?"))}</b> <i>{html.escape(d.get("model_type", ""))}</i></span>']
    for lbl, k in [("hidden", "hidden_size"), ("layers", "num_hidden_layers")]:
        if cfg.get(k) is not None:
            spans.append(f"<span>{lbl}=<b>{cfg[k]}</b></span>")
    if cfg.get("num_attention_heads") is not None:
        spans.append(f'<span>heads=<b>{cfg["num_attention_heads"]}/{cfg.get("num_key_value_heads", "—")}</b> q/kv</span>')
    for lbl, k in [("head_dim", "head_dim"), ("inter", "intermediate_size")]:
        if cfg.get(k) is not None:
            spans.append(f"<span>{lbl}=<b>{cfg[k]}</b></span>")
    if cfg.get("vocab_size") is not None:
        spans.append(f'<span>vocab=<b>{cfg["vocab_size"]:,}</b></span>')
    if cfg.get("attn_output_gate"):
        spans.append('<span class="arch-tag minor">output_gate</span>')
    if cfg.get("tie_word_embeddings"):
        spans.append('<span class="arch-tag minor">tied embed</span>')
    o.append('<div class="arch-cfg">' + "".join(spans) + "</div>")

    # modification tags
    mods = []
    fm = m.get("freeze_mode")
    if fm:
        mods.append(f'<span class="arch-tag freeze-{html.escape(str(fm))}">freeze={html.escape(str(fm))}</span>')
    if m.get("rwkv8_layer_indices"):
        mods.append(f'<span class="arch-tag rwkv8">RWKV-8 L={m["rwkv8_layer_indices"]}</span>')
    if m.get("mla_layer_indices"):
        mods.append(f'<span class="arch-tag mla">MLA L={m["mla_layer_indices"]}</span>')
    if m.get("mtp_installed"):
        mods.append('<span class="arch-tag mtp">MTP installed</span>')
    if m.get("engram_layer_indices"):
        mods.append(f'<span class="arch-tag engram">Engram L={m["engram_layer_indices"]}</span>')
    if mods:
        o.append('<div class="arch-mods">' + "".join(mods) + "</div>")

    # trainable bar
    pct = t.get("trainable_pct", 0.0)
    o.append(f'<div class="arch-totals">'
             f'<span class="arch-bar"><span class="arch-bar-fill" style="width:{pct:.2f}%"></span></span>'
             f'<span><b>{fmt_params(t.get("total_params"))}</b> total</span>'
             f'<span class="ok"><b>{fmt_params(t.get("trainable_params"))}</b> trainable ({pct:.2f}%)</span>'
             f'<span class="muted"><b>{fmt_params(t.get("frozen_params"))}</b> frozen</span></div>')
    o.append("</div>")  # arch-summary

    # per-layer list
    o.append('<div class="arch-layers">')
    for L in d.get("layers", []):
        kind = L.get("kind")
        if kind == "decoder_layer":
            a = L.get("attention") or {}
            mlp = L.get("mlp") or {}
            akind = a.get("kind", "?")
            tcls = type_class(akind)
            rowcls = ("row-rwkv8" if L.get("is_rwkv8") else "row-mla" if L.get("is_mla")
                      else "row-linear" if L.get("layer_type") == "linear_attention" else "row-fullattn")
            head = (f'<span class="lr-idx">L{L["index"]:02d}</span>'
                    f'<span class="lr-name">{html.escape(L.get("name", ""))}</span>'
                    f'<span class="lr-type {tcls}">{html.escape(akind)}</span>'
                    f'<span class="lr-params">{fmt_params(L.get("params"))}</span>'
                    f'{tr_badge(L.get("trainable_state"))}'
                    f'<span class="lr-caret">▸</span>')
            det = [f'<div class="lr-d-row"><span class="lbl">attn · {html.escape(akind)}</span>'
                   f'<span>{fmt_params(a.get("params"))}</span>{tr_badge(a.get("trainable"))}</div>'
                   f'<div class="lr-d-sub">heads={a.get("n_q_heads", "—")}/{a.get("n_kv_heads", "—")} · head_dim={a.get("head_dim", "—")}</div>']
            sub = f'intermediate={mlp.get("intermediate_size", "—")}'
            if mlp.get("n_experts"):
                sub += f' · experts={mlp["n_experts"]} top{mlp.get("n_experts_per_tok")}'
            det.append(f'<div class="lr-d-row"><span class="lbl">mlp · {html.escape(str(mlp.get("kind", "?")))}</span>'
                       f'<span>{fmt_params(mlp.get("params"))}</span>{tr_badge(mlp.get("trainable"))}</div>'
                       f'<div class="lr-d-sub">{html.escape(sub)}</div>')
            det.append(f'<div class="lr-d-row"><span class="lbl">norms</span><span>{fmt_params(L.get("norm_params"))}</span><span></span></div>')
            if L.get("other_params"):
                det.append(f'<div class="lr-d-row"><span class="lbl">other</span><span>{fmt_params(L.get("other_params"))}</span><span></span></div>')
            o.append(f'<details class="arch-item {rowcls}"><summary class="lr-head">{head}</summary>'
                     f'<div class="lr-detail">{"".join(det)}</div></details>')
        else:
            tag = {"embedding": "EMBED", "lm_head": "HEAD", "norm": "NORM"}.get(kind, str(kind).upper())
            shape = L.get("shape")
            nm = html.escape(L.get("name", "")) + (f' <span class="dim">{html.escape(str(shape))}</span>' if shape else "")
            o.append(f'<div class="arch-item row-{kind}"><div class="lr-head no-exp">'
                     f'<span class="lr-idx">·</span><span class="lr-name">{nm}</span>'
                     f'<span class="lr-type {kind}">{tag}</span>'
                     f'<span class="lr-params">{fmt_params(L.get("params"))}</span>'
                     f'{tr_badge(L.get("trainable"))}<span></span></div></div>')
    o.append("</div></div>")
    return "".join(o)


if __name__ == "__main__":
    print(render(architecture_for_run(Path(sys.argv[1]))))
