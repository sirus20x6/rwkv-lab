"""live_controls.py — trainer-side consumer for the trainboard v2 live-tuning panel.

The dashboard's POST /api/runs/<name>/control writes desired hyperparameter
overrides into the `run_controls` table of its SQLite DB (dashboard2/trainboard.db),
marking each pending (applied_ts IS NULL). This polls that table for one run,
caches the current overrides, and acks each applied one (applied_step/applied_ts)
so the panel flips it from "pending" to "applied".

Fully OPTIONAL and FAIL-SAFE: if the DB is missing, locked, or errors, poll()
keeps the last-known overrides (or {}) and never raises — live-tuning is a
convenience, never a training dependency. The dashboard opens the DB in WAL mode,
so our reads (and the small ack write) coexist with the Go writer.

Usage:
    ctl = LiveControls(db_path, run_name, whitelist={"lr_scale", "w_block", ...})
    for step in ...:
        if step % poll_every == 0:
            ctl.poll(step)
        w_block = ctl.get("w_block", args.w_block)   # override or default
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional


class LiveControls:
    def __init__(self, db_path: str, run_name: str, whitelist: Optional[Iterable[str]] = None):
        self.db_path = str(db_path) if db_path else ""
        self.run_name = run_name
        self.whitelist = set(whitelist) if whitelist is not None else None
        self.values: dict[str, float] = {}
        self.enabled = bool(self.db_path) and Path(self.db_path).exists()
        self._con: Optional[sqlite3.Connection] = None

    def _connection(self) -> sqlite3.Connection:
        """One persistent connection, reused across poll() calls instead of reopening
        the DB file every poll_every steps. Reconnects lazily if a prior poll closed it."""
        if self._con is None:
            self._con = sqlite3.connect(self.db_path, timeout=0.5)
            self._con.execute("PRAGMA busy_timeout=500")
        return self._con

    def poll(self, step: int) -> dict:
        """Read pending overrides for this run, cache them, and ack. Returns the
        current override dict. Never raises."""
        if not self.enabled:
            return self.values
        try:
            con = self._connection()
            rows = con.execute(
                "SELECT key, value, generation FROM run_controls "
                "WHERE run_name=? AND applied_ts IS NULL", (self.run_name,)
            ).fetchall()
            applied = []
            for key, value, gen in rows:
                if self.whitelist is not None and key not in self.whitelist:
                    continue
                try:
                    self.values[key] = float(value)
                except (TypeError, ValueError):
                    continue
                applied.append((key, gen))
            if applied:
                ts = time.time()
                for key, gen in applied:
                    con.execute(
                        "UPDATE run_controls SET applied_step=?, applied_ts=? "
                        "WHERE run_name=? AND key=? AND generation=?",
                        (int(step), ts, self.run_name, key, gen))
                con.commit()
                print("[live-tune] step %d: applied " % step
                      + ", ".join("%s=%g" % (k, self.values[k]) for k, _ in applied),
                      flush=True)
        except Exception:
            # diagnostics/convenience only — never break the training loop. Drop the
            # connection so the next poll() reconnects instead of reusing a broken one.
            try:
                if self._con is not None:
                    self._con.close()
            except Exception:
                pass
            self._con = None
        return self.values

    def get(self, key: str, default):
        """Effective value for `key`: the live override if set, else `default`."""
        v = self.values.get(key)
        return v if v is not None else default
