"""Experiment results registry — a SQLite store so lever A/Bs ACCUMULATE instead of scrolling past.

Every session this run, results were hand-assembled into markdown tables that vanished. This
persists each aggregated (config → mean±std metrics) row with its git sha + timestamp, and
`compare` queries across ALL historical runs to print a ranked table with significance vs a
baseline. Knowledge compounds across sessions instead of being re-derived.

    from rwkv_lab.registry import record            # experiment.py writes here
    python -m rwkv_lab.registry compare --task recall:16          # read across all runs
"""
from __future__ import annotations
import argparse, json, os, sqlite3, subprocess, time

DEFAULT_DB = os.environ.get("RWKV_LAB_DB", os.path.join(os.path.dirname(__file__), "..", "..", "experiments.db"))

_SCHEMA = """CREATE TABLE IF NOT EXISTS results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, git_sha TEXT, task TEXT, config TEXT,
  seeds INTEGER, steps INTEGER, metrics_json TEXT)"""


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "?"


def _con(db):
    c = sqlite3.connect(db or DEFAULT_DB)
    c.execute(_SCHEMA)
    return c


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
    out = {}
    for ts, sha, config, seeds, mj in rows:                 # later rows overwrite -> keep latest
        m = json.loads(mj)
        if metric in m:
            mean, std = m[metric][0], m[metric][1]
            out[config] = (mean, std, sha, ts, seeds)
    return out


def _compare(args):
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--task", required=True)
    cmp.add_argument("--metric", default="acc")
    cmp.add_argument("--baseline", default="baseline")
    cmp.add_argument("--db", default=None)
    cmp.set_defaults(fn=_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
