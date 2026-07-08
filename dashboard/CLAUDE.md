# trainboard — moe-mla training dashboard v2.0

Project-local instructions. This folder is a **self-contained successor** to `../dashboard/` (the
FastAPI + Chart.js v1). **Do not modify anything outside `dashboard/`.** v1 stays running on its own port;
v2 coexists.

## Stack (the four pillars)

| Layer        | Choice                                  | Why |
|--------------|-----------------------------------------|-----|
| Backend      | **Go 1.26** single binary               | one process, no venv, GPU-light |
| Datastore    | **SQLite** (`modernc.org/sqlite`, WAL)  | pure-Go, no cgo; real queryable history |
| Reactive UI  | **Datastar v1** (`datastar-go` SDK)     | server-driven signals over SSE, no SPA framework |
| Charts/viz   | **Pixi.js v8** (vendored UMD)           | GPU-accelerated, incremental-append, custom viz |
| Host metrics | `gopsutil/v4` + `nvidia-smi` parsing    | CPU/RAM/disk/procs + GPU telemetry, no cgo |

See **STACK.md** for API cheat-sheets and **DATA_MODEL.md** for the schema + JSONL field catalog.

## Run / build

```bash
go -C /thearray/git/moe-mla/dashboard run ./cmd/trainboard      # dev
go -C /thearray/git/moe-mla/dashboard build -o trainboard ./cmd/trainboard
go -C /thearray/git/moe-mla/dashboard vet ./...
```

- **Port `9124`** — memorable, not the banned 8080 (global rule), adjacent to the evo board 9123. Bind
  `127.0.0.1` only, no auth (localhost dev tool, same posture as v1).
- Reads `/thearray/git/moe-mla/runs/` (the same logs v1 reads). SQLite file lives at
  `dashboard/trainboard.db` (gitignore-able; rebuilt from JSONL on demand).
- **GPU-light & safe beside live training** — only file reads + `nvidia-smi` shell-outs; never touches CUDA.

## Conventions

- **Client libs are vendored** in `web/static/vendor/` (`datastar.js`, `pixi.min.js`). **No npm/bundler step.**
  `index.html` loads them with plain `<script>` tags — mirrors v1's vendored `chart.umd.min.js`.
- Datastar owns the **reactive HTML shell + scalar live-updates**; Pixi owns **series rendering**. The two
  meet only at a hidden version element (see STACK.md "glue").
- Go layout: `cmd/trainboard` wires everything; `internal/{db,ingest,sysmon,arch,series,server}` are the
  units. Keep handlers thin; put logic in the internal packages.
- **Avoid editing files outside this folder** unless the package entrypoint contract changes. The
  dashboard launches allowlisted Python modules with `python -m rwkv_lab.<module>` and
  `PYTHONPATH=<repo>/src`; `instrumented/train_mla.py` remains only as the dashboard-specific
  trainer variant with signal handling.
- Port the proven bits of v1 rather than reinventing: liveness windows (healthy<300 s / stalling<900 s),
  `_finite` NaN→null sanitization, safetensors metadata-only param counting.

## Control actions (write paths) — safety rules

stop / checkpoint-now / launch / notes / tags are confirm-gated client-side AND validated server-side:
- only signal/spawn processes whose cmdline matches the **training-script allowlist** (see sysmon);
- never escalate privileges; never `kill -9` (graceful SIGINT/SIGUSR1 only);
- append every action to the `actions` audit table.
