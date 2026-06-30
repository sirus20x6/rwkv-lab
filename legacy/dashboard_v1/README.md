# moe-mla dashboard

Lightweight FastAPI dashboard for watching MTP/MLA training runs on Qwen3.6-35B-A3B.

## Launch

```bash
/thearray/git/moe-mla/.venv/bin/python /thearray/git/moe-mla/dashboard/app.py
```

Then open http://127.0.0.1:4567 in a browser.

Stop with Ctrl-C.

## What it shows

- **System header** (live, 2 s SSE): per-GPU util / mem / temp / power, CPU %, RAM, disk free on `/thearray`, plus any running `train_mla.py` processes with liveness state.
- **Sidebar**: all runs under `/thearray/git/moe-mla/runs/`, sorted by last-update. Click a run to load it.
- **Main panel**:
  - KPI strip (step, train loss, eval ppl, top-1, tok/sec, event counts)
  - Loss + perplexity curve
  - Per-horizon accuracy (h=1…4 top-1 solid / top-5 dashed) — only for runs that log horizons (chained MTP training)
  - Training signal (tok/sec, gnorm, lr)
  - Checkpoint list with size + age
- **Compare panel**: check 2+ runs in the sidebar, pick a metric, overlay curves. Toggle smoothing for train metrics.

## Liveness states

| color | meaning |
|---|---|
| green | log updated < 5 min ago |
| yellow | log stale 5-15 min |
| red | process gone with log stale > 15 min |
| grey | no log / idle run directory |

Adjust `HEALTHY_WINDOW` / `STALE_WINDOW` in `app.py` if your log cadence differs — defaults assume train log every ~170 s.

## Dependencies

Installed into `/thearray/git/moe-mla/.venv/`:
- `fastapi`, `uvicorn`, `sse-starlette` — web + SSE
- `psutil` — CPU/RAM/disk/process list
- `pynvml` — NVIDIA telemetry

## Notes

- Localhost-only (binds 127.0.0.1). No auth.
- Charts re-render fully when the active run's event count grows (polled via SSE every 2 s).
- Compare data is cached client-side per run; refresh page to invalidate.
- Run-list refreshes every 2 s, so new runs appear automatically.
