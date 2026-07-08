# STACK.md — API cheat-sheet & gotchas

Pinned so future-me doesn't re-research. Verified versions (this checkout):
`datastar-go v1.2.2`, `modernc.org/sqlite v1.53.0`, `gopsutil/v4 v4.26.5`, vendored `datastar.js v1.0.0`,
`pixi.js v8` (UMD). Go 1.26.

---

## Datastar (server: Go SDK; client: data-* attributes)

**Import:** `github.com/starfederation/datastar-go/datastar` (needs Go ≥1.24). v1 renamed the old
**Merge→Patch** and **Fragments→Elements** — use the Patch names.

### Server (per request)
```go
import "github.com/starfederation/datastar-go/datastar"

sse := datastar.NewSSE(w, r)                       // sets SSE headers, flushes

// Patch DOM. Default mode = morph by element id. Element must have an id (or use WithSelector).
sse.PatchElements(`<div id="run-list">…</div>`)
sse.PatchElements(html, datastar.WithSelector("#run-list"), datastar.WithMode(datastar.ElementPatchModeAppend))
sse.RemoveElement("#temp")

// Patch client signals (RFC-7386 JSON merge-patch; set a key to null to delete it).
sse.PatchSignals([]byte(`{"cpu":42.0,"runVersions":{"iso_L0":1712}}`))
sse.MarshalAndPatchSignals(map[string]any{"cpu": 42.0})

// Read signals the browser sent up (on @get/@post):
var s struct{ SelectedRun string `json:"selectedRun"` }
datastar.ReadSignals(r, &s)

if sse.IsClosed() { return }                        // client disconnected — stop work
```
**One SSE response can carry many events** — batch several `PatchSignals` / `PatchElements` per tick.
For a long-lived stream, loop with a ticker and keep writing to the same `sse` until `IsClosed()`.

### Client (in index.html)
```html
<script type="module" src="/static/vendor/datastar.js"></script>
<body data-signals="{selectedRun:'', cpu:0, runVersions:{}}">
  <span data-text="$cpu.toFixed(1) + '%'"></span>
  <div data-on-load="@get('/api/stream')"></div>   <!-- opens the SSE stream on page load -->
  <button data-on-click="@post('/api/runs/'+$selectedRun+'/stop')">stop</button>
  <li data-class="{active: $selectedRun===name}" data-on-click="$selectedRun='iso_L0'">…</li>
  <div data-show="$compareOpen">…</div>
</body>
```
Key attrs: `data-signals` (declare), `data-bind` (two-way input), `data-text`, `data-on-<evt>`,
`data-show`, `data-class`, `data-attr`. Actions: `@get/@post/@put/@delete(url)` open an SSE request whose
response patches signals/elements. `$signal` reads a signal in expressions.

**Gotchas:** patched elements need stable `id`s (morph keys on id). Signals are a single nested object —
keep names JSON-safe. SSE must not be buffered by a proxy (we bind localhost, fine).

---

## Pixi.js v8 (vendored UMD → global `PIXI`)

```js
const app = new PIXI.Application();
await app.init({ canvas: document.getElementById('c'), antialias: true,
                 resolution: window.devicePixelRatio || 1, autoDensity: true,
                 backgroundAlpha: 0, preference: 'webgl' });   // webgpu w/ webgl fallback
const g = new PIXI.Graphics();
g.moveTo(x0,y0).lineTo(x1,y1).stroke({ width: 1.5, color: 0x6fa8ff });   // v8 chained API
g.rect(x,y,w,h).fill({ color: 0x16a766, alpha: 0.8 });
app.stage.addChild(g);
const label = new PIXI.Text({ text:'loss', style:{ fill:0xcfd5dc, fontSize:11, fontFamily:'monospace' }});
app.ticker.add((t) => { /* per-frame */ });
```
- v8 `Graphics`: **build path then `.fill()` / `.stroke()`** (NOT v7's `beginFill/lineStyle`). `stroke`/`fill`
  take an options object `{width,color,alpha}`.
- One `PIXI.Application` per chart canvas is fine; share a `Container` per series. For incremental append,
  keep the series' point array + a `Graphics` you `.clear()` and redraw only the visible (zoomed) window —
  cheap because Pixi redraw is GPU. Decimate server-side; LOD client-side.
- Interaction: `app.stage.eventMode='static'; app.stage.hitArea=app.screen;` then listen for
  `pointermove`/`wheel` on the canvas for crosshair + zoom (mirror v1's wheel-zoom-x in dashboard.js).
- Resize: `app.renderer.resize(w,h)` on container resize (ResizeObserver).

---

## SQLite (modernc.org/sqlite — pure Go, driver name `"sqlite"`)

```go
import _ "modernc.org/sqlite"
db, _ := sql.Open("sqlite", "file:trainboard.db?_pragma=busy_timeout(5000)")
db.Exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;`)
db.SetMaxOpenConns(1)   // simplest correctness for the writer; use a 2nd RO handle for reads if needed
```
- **WAL** lets the ingester write while handlers read. Keep one writer goroutine (the ingester) to avoid
  `SQLITE_BUSY`; readers use their own `*sql.DB` (or rely on busy_timeout).
- Driver is `"sqlite"` (NOT `"sqlite3"` — that's mattn/cgo). No cgo, so cross-compiles & needs no libs.
- Upserts: `INSERT … ON CONFLICT(run_id,step) DO UPDATE …` for idempotent re-ingest.

---

## Host & GPU telemetry

- **GPU:** shell `nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,enforced.power.limit --format=csv,noheader,nounits`,
  parse CSV rows. Robust, no cgo, matches v1's pynvml fields. Handle absent nvidia-smi gracefully.
- **Host:** `gopsutil/v4` — `cpu.Percent(0,false)` + `cpu.Percent(0,true)` (per-core), `mem.VirtualMemory()`,
  `disk.Usage("/thearray")`, `load.Avg()`.
- **Procs:** `process.Processes()` → filter `Cmdline()` against the training-script allowlist
  (`train_mla.py`, `convert_train.py`, `distill_consolidate.py`, `drive_isolation.py`,
  `train_mla_engram.py`), including `python -m rwkv_lab.<module>` launches; derive run name from `--out-dir`/`--out` arg, runtime from
  `CreateTime()`, RSS from `MemoryInfo()`. Liveness from train.jsonl mtime: healthy<300 s, stalling<900 s,
  else cold (same thresholds as v1).

---

## Datastar ↔ Pixi glue (the boundary)

Datastar streams scalars + a per-run **version token** (max step ingested). It `PatchElements` a hidden
`<div id="run-version" data-v="iso_L0:1712">`. `pixi-glue.js` runs a `MutationObserver` on that node; on
change for the selected run it `fetch('/api/series/<run>?since=<lastStep>&metrics=…')` and **appends only new
points** to the GPU buffers — no full redraw. Result: 1 SSE stream, Datastar never carries bulk series, Pixi
never parses SSE. Initial load / zoom-out fetches a server-decimated full series.
