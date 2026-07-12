"""Experiment results registry — a SQLite store so lever A/Bs ACCUMULATE instead of scrolling past.

Persists normalized campaigns, arms, seed/rung trials, curves, profiles, paired comparisons,
lineage, and reproducibility capsules. The original aggregate table remains readable for old
runs; new comparisons use paired confidence intervals and corrected permutation p-values.

    from rwkv_lab.registry import record            # experiment.py writes here
    python -m rwkv_lab.registry compare --task recall:16          # read across all runs
"""
from __future__ import annotations
import argparse, base64, hashlib, importlib.metadata, json, os, platform, socket
import sqlite3, subprocess, sys, time, zlib

DEFAULT_DB = os.environ.get("RWKV_LAB_DB", os.path.join(os.path.dirname(__file__), "..", "..", "experiments.db"))

_SCHEMA = """CREATE TABLE IF NOT EXISTS results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, git_sha TEXT, task TEXT, config TEXT,
  seeds INTEGER, steps INTEGER, metrics_json TEXT);
CREATE TABLE IF NOT EXISTS campaigns(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts REAL NOT NULL,
  finished_ts REAL,
  name TEXT,
  task TEXT NOT NULL,
  phase TEXT NOT NULL DEFAULT 'explore',
  status TEXT NOT NULL DEFAULT 'running',
  parent_id INTEGER,
  git_sha TEXT,
  config_json TEXT NOT NULL,
  capsule_json TEXT NOT NULL,
  FOREIGN KEY(parent_id) REFERENCES campaigns(id)
);
CREATE TABLE IF NOT EXISTS arms(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  UNIQUE(campaign_id,name),
  FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS trials(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id INTEGER NOT NULL,
  arm_id INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  rung INTEGER NOT NULL DEFAULT 0,
  budget REAL NOT NULL,
  phase TEXT NOT NULL DEFAULT 'explore',
  status TEXT NOT NULL,
  started_ts REAL NOT NULL,
  finished_ts REAL,
  metrics_json TEXT,
  series_json TEXT,
  profile_json TEXT,
  rng_json TEXT,
  error TEXT,
  UNIQUE(campaign_id,arm_id,seed,rung,phase),
  FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
  FOREIGN KEY(arm_id) REFERENCES arms(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS comparisons(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id INTEGER NOT NULL,
  arm_id INTEGER NOT NULL,
  baseline_arm_id INTEGER NOT NULL,
  metric TEXT NOT NULL,
  phase TEXT NOT NULL,
  n INTEGER NOT NULL,
  delta REAL,
  ci_low REAL,
  ci_high REAL,
  p_value REAL,
  effect_size REAL,
  p_adjusted REAL,
  significant INTEGER NOT NULL DEFAULT 0,
  confirmed INTEGER NOT NULL DEFAULT 0,
  details_json TEXT,
  UNIQUE(campaign_id,arm_id,metric,phase),
  FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS artifacts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id INTEGER NOT NULL,
  trial_id INTEGER,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  size_bytes INTEGER,
  metadata_json TEXT,
  created_ts REAL NOT NULL,
  FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
  FOREIGN KEY(trial_id) REFERENCES trials(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_campaign_arm ON trials(campaign_id,arm_id,rung,phase);
CREATE INDEX IF NOT EXISTS idx_comparisons_campaign ON comparisons(campaign_id,phase,metric);
"""


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "?"


def _con(db):
    c = sqlite3.connect(db or DEFAULT_DB)
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(_SCHEMA)
    return c


def _run(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _run_bytes(args: list[str]) -> bytes:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL)
    except Exception:
        return b""


def _compressed_git_diff() -> dict:
    raw = _run_bytes(["git", "diff", "--binary", "HEAD"])
    # `git diff` omits untracked source files; include them verbatim with clear
    # boundaries so a dirty campaign really is reconstructable.
    for rel in _run(["git", "ls-files", "--others", "--exclude-standard"]).splitlines():
        try:
            data = open(rel, "rb").read()
            raw += b"\nRWKV_LAB_UNTRACKED_BEGIN " + rel.encode() + b"\n" + data + \
                   b"\nRWKV_LAB_UNTRACKED_END " + rel.encode() + b"\n"
        except OSError:
            pass
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode(),
        "bytes": len(raw),
    }


def file_fingerprint(path: str, full: bool = False) -> dict:
    """Stable data identity; huge files use first/middle/last 1 MiB unless full=True."""
    p = os.path.abspath(path)
    st = os.stat(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if full or st.st_size <= 3 * 1024 * 1024:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
            mode = "full"
        else:
            for off in (0, max(0, st.st_size // 2 - 512 * 1024), max(0, st.st_size - 1024 * 1024)):
                f.seek(off); h.update(f.read(1024 * 1024))
            mode = "sampled"
    return {"path": p, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": h.hexdigest(), "hash_mode": mode}


def capture_capsule(extra: dict | None = None) -> dict:
    """Capture enough code/environment state to reproduce or audit a campaign."""
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda,
                      "cudnn": torch.backends.cudnn.version(),
                      "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                      "device_count": torch.cuda.device_count()}
    except Exception as e:
        torch_info = {"error": repr(e)}
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                      if d.metadata.get("Name"))
    cap = {
        "created_ts": time.time(), "git_sha": _sha(),
        "git_status": _run(["git", "status", "--porcelain"]),
        "git_diff": _compressed_git_diff(),
        "command": sys.argv, "cwd": os.getcwd(), "hostname": socket.gethostname(),
        "python": sys.version, "platform": platform.platform(), "torch": torch_info,
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.sm,clocks.mem",
                            "--format=csv,noheader,nounits"]),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
        "packages": packages,
    }
    if extra:
        cap["inputs"] = extra
    return cap


def create_campaign(task: str, config: dict, *, name: str = "", phase: str = "explore",
                    parent_id: int | None = None, capsule: dict | None = None,
                    db: str | None = None) -> int:
    c = _con(db)
    cur = c.execute("""INSERT INTO campaigns(created_ts,name,task,phase,status,parent_id,git_sha,config_json,capsule_json)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (time.time(), name, task, phase, "running", parent_id, _sha(),
         json.dumps(config, sort_keys=True), json.dumps(capsule or capture_capsule())))
    cid = int(cur.lastrowid); c.commit(); c.close(); return cid


def ensure_arm(campaign_id: int, name: str, config: dict, db: str | None = None) -> int:
    c = _con(db)
    c.execute("INSERT OR IGNORE INTO arms(campaign_id,name,config_json) VALUES(?,?,?)",
              (campaign_id, name, json.dumps(config, sort_keys=True)))
    aid = c.execute("SELECT id FROM arms WHERE campaign_id=? AND name=?", (campaign_id, name)).fetchone()[0]
    c.commit(); c.close(); return int(aid)


def set_arm_status(arm_id: int, status: str, db: str | None = None):
    c = _con(db); c.execute("UPDATE arms SET status=? WHERE id=?", (status, arm_id)); c.commit(); c.close()


def record_trial(campaign_id: int, arm_id: int, seed: int, rung: int, budget: float,
                 metrics: dict | None, *, series: list | None = None, profile: dict | None = None,
                 rng: dict | None = None, phase: str = "explore", status: str = "complete",
                 started_ts: float | None = None, error: str = "", db: str | None = None) -> int:
    c = _con(db)
    cur = c.execute("""INSERT INTO trials(campaign_id,arm_id,seed,rung,budget,phase,status,started_ts,
        finished_ts,metrics_json,series_json,profile_json,rng_json,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(campaign_id,arm_id,seed,rung,phase) DO UPDATE SET status=excluded.status,
        finished_ts=excluded.finished_ts,metrics_json=excluded.metrics_json,series_json=excluded.series_json,
        profile_json=excluded.profile_json,rng_json=excluded.rng_json,error=excluded.error
        RETURNING id""",
        (campaign_id, arm_id, seed, rung, float(budget), phase, status, started_ts or time.time(), time.time(),
         json.dumps(metrics) if metrics is not None else None, json.dumps(series or []),
         json.dumps(profile or {}), json.dumps(rng or {}), error or None))
    tid = int(cur.fetchone()[0]); c.commit(); c.close(); return tid


def record_comparison(campaign_id: int, arm_id: int, baseline_arm_id: int, metric: str,
                      phase: str, stats: dict, *, confirmed: bool = False, db: str | None = None):
    c = _con(db)
    c.execute("""INSERT INTO comparisons(campaign_id,arm_id,baseline_arm_id,metric,phase,n,delta,
        ci_low,ci_high,p_value,effect_size,p_adjusted,significant,confirmed,details_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(campaign_id,arm_id,metric,phase) DO UPDATE SET
        baseline_arm_id=excluded.baseline_arm_id,n=excluded.n,delta=excluded.delta,
        ci_low=excluded.ci_low,ci_high=excluded.ci_high,p_value=excluded.p_value,
        effect_size=excluded.effect_size,p_adjusted=excluded.p_adjusted,
        significant=excluded.significant,confirmed=excluded.confirmed,details_json=excluded.details_json""",
        (campaign_id, arm_id, baseline_arm_id, metric, phase, int(stats["n"]), stats["delta"],
         stats["ci_low"], stats["ci_high"], stats["p_value"], stats["effect_size"],
         stats.get("p_adjusted", stats["p_value"]), int(bool(stats.get("significant"))),
         int(confirmed), json.dumps(stats)))
    c.commit(); c.close()


def record_artifact(campaign_id: int, path: str, kind: str, *, trial_id: int | None = None,
                    metadata: dict | None = None, db: str | None = None) -> int:
    fp = file_fingerprint(path, full=True)
    c = _con(db)
    cur = c.execute("""INSERT INTO artifacts(campaign_id,trial_id,kind,path,sha256,size_bytes,
        metadata_json,created_ts) VALUES(?,?,?,?,?,?,?,?)""",
        (campaign_id, trial_id, kind, fp["path"], fp["sha256"], fp["size"],
         json.dumps(metadata or {}), time.time()))
    aid = int(cur.lastrowid); c.commit(); c.close(); return aid


def finish_campaign(campaign_id: int, status: str = "complete", db: str | None = None):
    c = _con(db); c.execute("UPDATE campaigns SET status=?,finished_ts=? WHERE id=?",
                            (status, time.time(), campaign_id)); c.commit(); c.close()


def campaign_rows(campaign_id: int, db: str | None = None) -> list[dict]:
    c = _con(db)
    rows = c.execute("""SELECT a.name,t.seed,t.rung,t.phase,t.status,t.metrics_json,t.profile_json
        FROM trials t JOIN arms a ON a.id=t.arm_id WHERE t.campaign_id=? ORDER BY t.rung,a.name,t.seed""",
        (campaign_id,)).fetchall(); c.close()
    return [{"arm": a, "seed": s, "rung": r, "phase": p, "status": st,
             "metrics": json.loads(m or "{}"), "profile": json.loads(pr or "{}")} for a,s,r,p,st,m,pr in rows]


def record(task: str, config: str, seeds: int, steps: int, metrics: dict, db: str | None = None):
    """metrics: {name: [mean, std], ...} as produced by experiment.py's aggregation."""
    c = _con(db)
    c.execute("INSERT INTO results(ts,git_sha,task,config,seeds,steps,metrics_json) VALUES(?,?,?,?,?,?,?)",
              (time.time(), _sha(), task, config, int(seeds), int(steps), json.dumps(metrics)))
    c.commit(); c.close()


def latest_by_config(task: str, metric: str = "acc", db: str | None = None) -> dict:
    """Most-recent row per config for a task -> {config: (mean, std, git_sha, ts, seeds)}."""
    c = _con(db)
    rows = c.execute("SELECT ts,git_sha,config,seeds,metrics_json FROM results WHERE task=? ORDER BY ts",
                     (task,)).fetchall()
    c.close()
    latest = {}
    for row in rows:                                        # later rows overwrite -> keep latest
        latest[row[2]] = row
    out = {}
    for config, (ts, sha, _cfg, seeds, mj) in latest.items():
        m = json.loads(mj)
        if metric in m:                                     # newest row lacks metric -> absent
            out[config] = (m[metric][0], m[metric][1], sha, ts, seeds)
    return out


def _compare(args):
    c = _con(args.db)
    latest = c.execute("SELECT id,phase FROM campaigns WHERE task=? ORDER BY created_ts DESC LIMIT 1",
                       (args.task,)).fetchone()
    if latest:
        rows = c.execute("""SELECT a.name,p.n,p.delta,p.ci_low,p.ci_high,p.p_adjusted,
            p.effect_size,p.significant,p.confirmed FROM comparisons p
            JOIN arms a ON a.id=p.arm_id WHERE p.campaign_id=? AND p.metric=? ORDER BY p.delta DESC""",
            (latest[0], args.metric)).fetchall()
        if rows:
            print(f"=== campaign #{latest[0]} {args.task}: paired {args.metric} ({latest[1]}) ===")
            for name,n,d,lo,hi,p,e,sig,confirmed in rows:
                print(f"  {name:18} Δ{d:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                      f"p_holm={p:.4g}  dz={e:.2f}  n={n}  "
                      f"{'CONFIRMED' if confirmed else 'evidence' if sig else 'inconclusive'}")
            c.close(); return
    c.close()
    data = latest_by_config(args.task, args.metric, args.db)
    if not data:
        print(f"no results for task={args.task} metric={args.metric}"); return
    base = data.get(args.baseline)
    print(f"=== {args.task}: {args.metric} across runs (latest per config) ===")
    ranked = sorted(data.items(), key=lambda kv: -kv[1][0])   # best mean first
    for config, (m, s, sha, ts, seeds) in ranked:
        line = f"  {config:18} {m:.3f}±{s:.3f}  n={seeds}  @{sha}"
        if base and config != args.baseline:
            d = m - base[0]; sig = abs(d) > (s + base[1])
            line += f"   Δ{d:+.3f} {'SIGNIFICANT' if sig else 'ns'}"
        print(line)


def _campaigns(args):
    c = _con(args.db)
    rows = c.execute("""SELECT c.id,c.task,c.phase,c.status,COALESCE(c.parent_id,0),c.git_sha,
        count(DISTINCT a.id),count(DISTINCT t.id) FROM campaigns c LEFT JOIN arms a ON a.campaign_id=c.id
        LEFT JOIN trials t ON t.campaign_id=c.id GROUP BY c.id ORDER BY c.created_ts DESC LIMIT ?""",
        (args.limit,)).fetchall(); c.close()
    for cid,task,phase,status,parent,sha,arms,trials in rows:
        print(f"#{cid:<5} {task:18} {phase:8} {status:10} arms={arms:<3} trials={trials:<4} "
              f"@{sha}" + (f" parent=#{parent}" if parent else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--task", required=True)
    cmp.add_argument("--metric", default="acc")
    cmp.add_argument("--baseline", default="baseline")
    cmp.add_argument("--db", default=None)
    cmp.set_defaults(fn=_compare)
    ls = sub.add_parser("campaigns")
    ls.add_argument("--limit", type=int, default=20); ls.add_argument("--db", default=None)
    ls.set_defaults(fn=_campaigns)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
