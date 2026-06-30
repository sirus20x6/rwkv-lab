# legacy/ — archived one-off / superseded scripts

Moved out of the project root so the live pipeline (convert_train + ROSA + Engram)
is uncluttered. NOT deleted — recoverable. None of these are imported by the
current pipeline (verified by import-closure analysis). Notable:
- convert_distill_progressive.py — the OLD *cumulative* GDN->RWKV trainer, replaced
  by the per-layer isolation approach (convert_train.py).
- convert.py / build_bkv_stats.py / eval_mla_patch.py — MLA-conversion one-offs
  (the MLA model was a separate experiment, set aside).
- tuned_lens / logit_lens_sweep / analyze_spectrum / embedding_probe / benchmark
  / eval_* / verify_real_layer — analysis & eval one-offs.
To run one again: `PYTHONPATH=.. python legacy/<script>.py` (root modules on path).

## dashboard_v1/ (added 2026-06-30)
The v1 dashboard — FastAPI + Chart.js (`app.py`, `architecture.py`, `static/`),
served on :4567. Superseded by the Go/SQLite/Pixi `dashboard2/` (trainboard, :9124),
which reimplemented the proven bits (liveness windows, `_finite` sanitization,
safetensors param counting) natively in Go. v1 process was stopped 2026-06-30.
Kept for reference (architecture.py param-counting logic) — recoverable.

## instrumented_convert_train.py (added 2026-06-30)
The dashboard2/instrumented/ copy of convert_train.py. Its enhancements (SIGUSR1/
SIGINT run-control hooks, GrokAutopilot, codec_rel surfacing) were reconciled INTO
the root canonical convert_train.py, so this copy is redundant. The dashboard
launches root's copy (control.go: RepoRoot+basename) and now recognizes
convert_train.py as signal-capable by name (procs.go VerifyTrainingPID), so
checkpoint-now/stop work on root runs without this copy. Kept for reference.
