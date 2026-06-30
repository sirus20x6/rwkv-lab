"""
moe-mla training + system dashboard.

Serves on http://127.0.0.1:4567 — reads /thearray/git/moe-mla/runs/*/train.jsonl
for per-run metrics, pynvml+psutil for live GPU/CPU/RAM/disk, and scans
process list for running train_mla.py jobs.

Run:   python dashboard/app.py
Stop:  Ctrl-C
"""
from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import psutil
import pynvml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from architecture import architecture_for_run

ROOT = Path("/thearray/git/moe-mla")
RUNS_DIR = ROOT / "runs"
STATIC_DIR = Path(__file__).parent / "static"
PORT = 4567
POLL_INTERVAL = 2.0

HEALTHY_WINDOW = 300   # log was updated within 5 min: definitely live
STALE_WINDOW = 900     # 5-15 min: yellow. >15 min: red/cold.

_sse_clients: list[asyncio.Queue] = []
_nvml_handles: list = []
_nvml_ok = False
_parsed_run_cache: dict[Path, tuple[tuple[int, int], dict]] = {}
_run_summary_cache: dict[Path, tuple[tuple[int, int], dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _nvml_ok
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            _nvml_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        _nvml_ok = True
    except Exception as e:
        print(f"[warn] pynvml init failed: {e}")
    psutil.cpu_percent(interval=None)  # prime

    task = asyncio.create_task(_broadcaster())
    try:
        yield
    finally:
        task.cancel()
        if _nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _gpu_stats() -> list[dict]:
    out = []
    for i, h in enumerate(_nvml_handles):
        try:
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            try:
                pwr_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pwr_w = None
            try:
                pwr_cap_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                pwr_cap_w = None
            out.append({
                "index": i,
                "name": name,
                "mem_used_gb": mem.used / (1024**3),
                "mem_total_gb": mem.total / (1024**3),
                "mem_pct": 100 * mem.used / mem.total,
                "util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "temp_c": temp,
                "power_w": pwr_w,
                "power_cap_w": pwr_cap_w,
            })
        except Exception as e:
            out.append({"index": i, "error": str(e)})
    return out


def _system_snapshot() -> dict:
    cpu = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(str(ROOT))
    load = psutil.getloadavg()
    return {
        "gpus": _gpu_stats(),
        "cpu_pct": cpu,
        "cpu_per_core": cpu_per_core,
        "cpu_count": psutil.cpu_count(),
        "ram_used_gb": mem.used / (1024**3),
        "ram_total_gb": mem.total / (1024**3),
        "ram_pct": mem.percent,
        "disk_used_gb": (disk.total - disk.free) / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_total_gb": disk.total / (1024**3),
        "disk_pct": 100 * (disk.total - disk.free) / disk.total,
        "loadavg": load,
        "ts": time.time(),
    }


def _run_name_from_cmdline(cmdline: list[str]) -> Optional[str]:
    for i, a in enumerate(cmdline):
        if a == "--out-dir" and i + 1 < len(cmdline):
            return Path(cmdline[i + 1]).name
        if a.startswith("--out-dir="):
            return Path(a.split("=", 1)[1]).name
    return None


def _parse_int_arg(cmdline: list[str], name: str) -> Optional[int]:
    eq = f"{name}="
    for i, a in enumerate(cmdline):
        if a == name and i + 1 < len(cmdline):
            try:
                return int(cmdline[i + 1])
            except ValueError:
                return None
        if a.startswith(eq):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _processes() -> list[dict]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            info = p.info
            cmdline = info.get("cmdline") or []
            if not cmdline:
                continue
            joined = " ".join(cmdline)
            if "train_mla" not in joined:
                continue
            if not any("python" in (a or "").lower() for a in cmdline[:2]):
                continue

            run_name = _run_name_from_cmdline(cmdline)
            max_steps = _parse_int_arg(cmdline, "--max-steps")
            runtime = time.time() - info["create_time"]
            try:
                cpu_pct = p.cpu_percent(interval=None)
                rss_gb = p.memory_info().rss / (1024**3)
                num_threads = p.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_pct = 0.0
                rss_gb = 0.0
                num_threads = 0

            log_age = None
            if run_name:
                log_path = RUNS_DIR / run_name / "train.jsonl"
                if log_path.exists():
                    log_age = time.time() - log_path.stat().st_mtime

            if log_age is None:
                state = "unknown"
            elif log_age < HEALTHY_WINDOW:
                state = "healthy"
            elif log_age < STALE_WINDOW:
                state = "stalling"
            else:
                state = "dead"

            # Find train_mla.py arg (script path)
            script = next((a for a in cmdline if "train_mla" in a and a.endswith(".py")), None)
            if script:
                script = Path(script).name

            out.append({
                "pid": info["pid"],
                "script": script,
                "runtime_s": runtime,
                "cpu_pct": cpu_pct,
                "rss_gb": rss_gb,
                "num_threads": num_threads,
                "run_name": run_name,
                "log_age_s": log_age,
                "state": state,
                "max_steps": max_steps,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _log_fingerprint(log: Path) -> tuple[int, int]:
    stat = log.stat()
    return stat.st_mtime_ns, stat.st_size


def _empty_run_payload() -> dict:
    return {"train": [], "eval": [], "checkpoints": []}


def _finite(obj):
    """Recursively replace NaN/Inf floats with None. json.dumps emits bare
    NaN/Infinity tokens (invalid JSON) which break the frontend's JSON.parse and
    blank the whole page; converting to null keeps every response parseable."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _parse_run(run_path: Path) -> dict:
    log = run_path / "train.jsonl"
    if not log.exists():
        payload = _empty_run_payload()
    else:
        fingerprint = _log_fingerprint(log)
        cached = _parsed_run_cache.get(log)
        if cached and cached[0] == fingerprint:
            payload = cached[1]
        else:
            train, evals, checkpoints = [], [], []
            with log.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = e.get("kind")
                    if k == "train":
                        train.append(e)
                    elif k == "eval":
                        evals.append(e)
                    elif k == "checkpoint":
                        checkpoints.append(e)
            payload = _finite({"train": train, "eval": evals, "checkpoints": checkpoints})
            _parsed_run_cache[log] = (fingerprint, payload)
    # Attach LoopedRWKV residual-weight stats (published by the poller into
    # loop_rw.json). Read fresh each request so it isn't frozen by the train.jsonl
    # cache — the artifact updates on its own cadence.
    lr_path = run_path / "loop_rw.json"
    if lr_path.exists():
        try:
            payload = {**payload, "loop_rw": _finite(json.loads(lr_path.read_text()))}
        except Exception:
            pass
    # Original untouched-model baseline (runs/_baseline.json, global — the same
    # original model for every run). The chart draws it as a fixed reference line
    # so each run's eval ppl is read against the original. Fresh per request.
    bl_path = RUNS_DIR / "_baseline.json"
    if bl_path.exists():
        try:
            payload = {**payload, "baseline": _finite(json.loads(bl_path.read_text()))}
        except Exception:
            pass
    return payload


def _summarize_log(log: Path) -> dict:
    fingerprint = _log_fingerprint(log)
    cached = _run_summary_cache.get(log)
    if cached and cached[0] == fingerprint:
        return cached[1]

    latest_train = None
    latest_eval = None
    latest_checkpoint = None
    n_train = 0
    n_eval = 0
    n_checkpoint = 0
    has_horizons = False
    with log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = e.get("kind")
            if k == "train":
                latest_train = e
                n_train += 1
            elif k == "eval":
                latest_eval = e
                n_eval += 1
                if "h4_top1" in e:
                    has_horizons = True
            elif k == "checkpoint":
                latest_checkpoint = e
                n_checkpoint += 1

    latest_step = None
    latest_loss = None
    latest_ppl = None
    if latest_train:
        latest_step = latest_train.get("step")
        latest_loss = latest_train.get("loss")
    if latest_eval:
        s = latest_eval.get("step")
        if latest_step is None or (s is not None and s > latest_step):
            latest_step = s
        latest_ppl = latest_eval.get("ppl")
        if latest_loss is None:
            latest_loss = latest_eval.get("loss")

    summary = {
        "latest_step": latest_step,
        "latest_loss": latest_loss,
        "latest_ppl": latest_ppl,
        "num_train_events": n_train,
        "num_eval_events": n_eval,
        "num_checkpoint_events": n_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "has_horizons": has_horizons,
    }
    summary = _finite(summary)
    _run_summary_cache[log] = (fingerprint, summary)
    return summary


def _run_summary(run_path: Path) -> dict:
    name = run_path.name
    log = run_path / "train.jsonl"
    if not log.exists():
        return {
            "name": name,
            "has_log": False,
            "last_update": None,
            "last_update_age_s": None,
            "latest_step": None,
            "latest_loss": None,
            "latest_ppl": None,
            "num_train_events": 0,
            "num_eval_events": 0,
            "num_checkpoint_events": 0,
            "latest_checkpoint": None,
            "has_horizons": False,
            "alive_state": "no_log",
        }
    stat = log.stat()
    last_update = stat.st_mtime
    summary = _summarize_log(log)

    age = time.time() - last_update
    if age < HEALTHY_WINDOW:
        alive_state = "healthy"
    elif age < STALE_WINDOW:
        alive_state = "stalling"
    else:
        alive_state = "cold"

    return {
        "name": name,
        "has_log": True,
        "last_update": last_update,
        "last_update_age_s": age,
        **summary,
        "alive_state": alive_state,
    }


def _list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            runs.append(_run_summary(d))
        except Exception as e:
            runs.append({"name": d.name, "error": str(e), "has_log": False,
                         "alive_state": "error"})
    runs.sort(key=lambda r: (-(r.get("last_update") or 0), r.get("name", "")))
    return runs


def _resolve_run_path(name: str) -> Path:
    run = (RUNS_DIR / name).resolve()
    runs_root = RUNS_DIR.resolve()
    if run.parent != runs_root:
        raise HTTPException(400, "invalid run name")
    return run


def _list_checkpoints(run_path: Path) -> list[dict]:
    out = []
    if not run_path.exists():
        return out
    for d in sorted(run_path.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("step_"):
            continue
        total = 0
        mtime = d.stat().st_mtime
        try:
            for p in d.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
                    mt = p.stat().st_mtime
                    if mt > mtime:
                        mtime = mt
        except PermissionError:
            continue
        out.append({
            "name": d.name,
            "size_gb": total / (1024**3),
            "mtime": mtime,
            "age_s": time.time() - mtime,
        })
    return out


# ---- endpoints ----


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/runs")
async def api_runs():
    return _list_runs()


@app.get("/api/runs/{name}")
async def api_run(name: str):
    run = _resolve_run_path(name)
    if not run.is_dir():
        raise HTTPException(404, f"no such run: {name}")
    return _parse_run(run)


@app.get("/api/runs/{name}/checkpoints")
async def api_ckpts(name: str):
    run = _resolve_run_path(name)
    if not run.is_dir():
        raise HTTPException(404, f"no such run: {name}")
    return _list_checkpoints(run)


@app.get("/api/runs/{name}/architecture")
async def api_arch(name: str):
    """Architecture compute can take seconds (safetensors metadata read on
    first hit, or torch.load on legacy runs without sidecar config.json).
    Run it in a worker thread so it doesn't block the asyncio loop — that
    blocking was freezing SSE pushes and every other route while the user
    waited for the architecture panel to load."""
    run = _resolve_run_path(name)
    if not run.is_dir():
        raise HTTPException(404, f"no such run: {name}")
    return await asyncio.to_thread(architecture_for_run, run)


@app.get("/api/system")
async def api_system():
    return _system_snapshot()


@app.get("/api/processes")
async def api_procs():
    return _processes()


@app.get("/api/stream")
async def api_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    _sse_clients.append(q)

    async def gen():
        try:
            # Immediate first tick so the client has data right away
            try:
                first = {
                    "system": _system_snapshot(),
                    "processes": _processes(),
                    "runs": _list_runs(),
                    "ts": time.time(),
                }
                yield {"event": "tick", "data": json.dumps(first)}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}

            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=30)
                    yield {"event": "tick", "data": json.dumps(data)}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

    return EventSourceResponse(gen())


async def _broadcaster():
    """Poll metrics every POLL_INTERVAL and push to all SSE clients."""
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            data = {
                "system": _system_snapshot(),
                "processes": _processes(),
                "runs": _list_runs(),
                "ts": time.time(),
            }
        except Exception as e:
            data = {"error": str(e), "ts": time.time()}
        for q in list(_sse_clients):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # Drop oldest and retry once
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except Exception:
                    pass


def main() -> None:
    import uvicorn
    print(f"moe-mla dashboard listening on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
