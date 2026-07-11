"""Guarded offline consolidation of daily state/replay into disposable adapters.

The day-state + SDFT idea came from the RWKV community:
https://discord.com/channels/992359628979568762/992359629419991142/1503014794171514964.
This controller adds the safety mechanism: immutable input snapshots, bounded
training, held-out regression gates, and explicit human promotion. It does not
autonomously alter a serving checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping


SCHEMA = "rwkv-lab.guarded-adapter-consolidation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GuardedAdapterConsolidator:
    def __init__(self, root: str | Path, *, max_train_tokens: int = 100_000,
                 maximum_regression: float = 0.0, minimum_improvement: float = 0.0):
        if max_train_tokens < 1 or maximum_regression < 0:
            raise ValueError("invalid consolidation bounds")
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.max_train_tokens = max_train_tokens
        self.maximum_regression = maximum_regression
        self.minimum_improvement = minimum_improvement

    def snapshot(self, files: Mapping[str, str | Path]) -> dict:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        directory = self.root / f"snapshot-{stamp}-{time.time_ns() % 1_000_000:06d}"
        directory.mkdir()
        manifest = {"schema": SCHEMA, "created_ts": time.time(), "files": {}}
        for name, source in sorted(files.items()):
            source = Path(source)
            if not source.is_file() or Path(name).name != name:
                raise ValueError("snapshot inputs must be named files")
            destination = directory / name; shutil.copy2(source, destination)
            destination.chmod(0o444)
            manifest["files"][name] = {"path": str(destination), "sha256": _sha256(destination)}
        path = directory / "snapshot.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); path.chmod(0o444)
        return {**manifest, "manifest": str(path)}

    def run(self, snapshot: Mapping, *, train: Callable[[Mapping, Path, int], str | Path],
            evaluate: Callable[[str | Path], Mapping[str, float]],
            baseline_metrics: Mapping[str, float], primary_metric: str,
            regression_metrics: tuple[str, ...] = ()) -> dict:
        run = self.root / f"candidate-{time.time_ns()}"; run.mkdir()
        candidate = Path(train(snapshot, run, self.max_train_tokens))
        if not candidate.is_file() and not candidate.is_dir():
            raise RuntimeError("consolidation trainer did not produce an adapter")
        metrics = {k: float(v) for k, v in evaluate(candidate).items()}
        if primary_metric not in metrics or primary_metric not in baseline_metrics:
            raise ValueError("primary metric missing from consolidation evaluation")
        improvement = metrics[primary_metric] - float(baseline_metrics[primary_metric])
        regressions = {key: float(baseline_metrics[key]) - metrics[key]
                       for key in regression_metrics if key in metrics and key in baseline_metrics}
        gates = {"minimum_improvement": improvement >= self.minimum_improvement,
                 "maximum_regression": all(v <= self.maximum_regression for v in regressions.values())}
        receipt = {"schema": SCHEMA, "status": "awaiting_human_promotion",
                   "snapshot_manifest": snapshot["manifest"], "candidate": str(candidate.resolve()),
                   "metrics": metrics, "baseline_metrics": dict(baseline_metrics),
                   "improvement": improvement, "regressions": regressions, "gates": gates,
                   "eligible": all(gates.values()), "promoted": False}
        path = run / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return {**receipt, "receipt": str(path)}

    def promote(self, receipt_path: str | Path, destination: str | Path, *, approved: bool) -> dict:
        if not approved:
            raise PermissionError("adapter promotion requires explicit human approval")
        path = Path(receipt_path); receipt = json.loads(path.read_text())
        if receipt.get("schema") != SCHEMA or not receipt.get("eligible"):
            raise ValueError("consolidation candidate is not eligible")
        source, destination = Path(receipt["candidate"]), Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        if source.is_dir(): shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        receipt.update({"status": "promoted", "promoted": True,
                        "promoted_path": str(destination.resolve()), "promoted_ts": time.time()})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return receipt
