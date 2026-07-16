#!/usr/bin/env python3
"""Stop MoonViT at 40k train images and build the matching fusion shard."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rwkv_lab.moonvit import checkpoint_fingerprint, feature_cache_key


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "curated_vision/vision_eight_hour.jsonl"
EVAL = ROOT / "curated_vision/vision_eight_hour_eval.jsonl"
SHARD_TRAIN = ROOT / "curated_vision/vision_next_shard_000_train.jsonl"
RECEIPT = ROOT / "curated_vision/vision_next_so400m_128_shard_000.cache.json"
MOON_CHECKPOINT = ROOT / "models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors"
CACHE_ROOT = Path("/thearray/downloads/cache/moe-mla")
ACTIVE_MOON = CACHE_ROOT / "moonvit_next_128"
SHARD_MOON = CACHE_ROOT / "moonvit_next_128_shard_000"
OVERFLOW_MOON = CACHE_ROOT / "moonvit_next_128_shard_001_partial"
SHARD_FUSION = CACHE_ROOT / "fusion_so400m_next_128_shard_000"
SIGLIP = ROOT / "models/vision/siglip2-so400m-patch16-512"
TARGET_TRAIN = 40_000
PREFIX_TOKENS = 128
TAPS = (8, 17, 26)
VIEW_MODE = "full-quadrants"


def cache_count(path: Path) -> int:
    try:
        return sum(entry.name.endswith(".pt") for entry in os.scandir(path))
    except FileNotFoundError:
        return 0


def active_cache_pids() -> list[int]:
    result = []
    needle = str(ACTIVE_MOON).encode()
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            command = (candidate / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if b"rwkv_lab.vision_cache" in command and needle in command:
            result.append(int(candidate.name))
    return result


def stop_active_cache() -> None:
    pids = active_cache_pids()
    if len(pids) != 1:
        raise RuntimeError(f"expected one active MoonViT cache process, found {pids}")
    pid = pids[0]
    print({"kind": "handoff", "action": "interrupt_moonvit", "pid": pid}, flush=True)
    os.kill(pid, signal.SIGINT)
    for _ in range(300):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    # Cache files use temporary-write + atomic rename, so SIGTERM cannot expose
    # a partially written public .pt entry if SIGINT somehow fails to unwind.
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f"MoonViT cache process {pid} did not stop")


def moon_key(image_value: str, fingerprint: str) -> str:
    image = Path(image_value)
    image = image if image.is_absolute() else ROOT / image
    return feature_cache_key(
        image.resolve(), max_input_patches=1024, prefix_tokens=PREFIX_TOKENS,
        vision_fingerprint=fingerprint, tap_layers=TAPS, view_mode=VIEW_MODE)


def select_cached_rows() -> tuple[list[str], set[str]]:
    fingerprint = checkpoint_fingerprint(MOON_CHECKPOINT)
    rows: list[str] = []
    keys: set[str] = set()
    with TRAIN.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = moon_key(row["image"], fingerprint)
            if (ACTIVE_MOON / key).is_file():
                rows.append(line if line.endswith("\n") else line + "\n")
                keys.add(key)
                if len(rows) == TARGET_TRAIN:
                    break
    if len(rows) != TARGET_TRAIN:
        raise RuntimeError(
            f"only mapped {len(rows)} cached train rows at the handoff boundary")
    return rows, keys


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def freeze_moon_shard(rows: list[str], selected: set[str]) -> None:
    if SHARD_MOON.exists():
        raise RuntimeError(f"refusing to replace existing shard cache {SHARD_MOON}")
    os.replace(ACTIVE_MOON, SHARD_MOON)
    OVERFLOW_MOON.mkdir(parents=True, exist_ok=True)
    overflow = 0
    for entry in list(SHARD_MOON.iterdir()):
        if entry.suffix == ".pt" and entry.name not in selected:
            os.replace(entry, OVERFLOW_MOON / entry.name)
            overflow += 1
        elif entry.name.endswith(".tmp"):
            entry.unlink(missing_ok=True)
    atomic_text(SHARD_TRAIN, "".join(rows))
    print({"kind": "handoff", "moon_train": len(selected),
           "overflow_preserved": overflow, "manifest": str(SHARD_TRAIN)}, flush=True)


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    print({"kind": "handoff", "command": command}, flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def expected_images() -> int:
    images: set[str] = set()
    for manifest in (SHARD_TRAIN, EVAL):
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                image = Path(json.loads(line)["image"])
                image = image if image.is_absolute() else ROOT / image
                images.add(str(image.resolve()))
    return len(images)


def write_receipt() -> None:
    expected = expected_images()
    counts = {"moon": cache_count(SHARD_MOON),
              "fusion": cache_count(SHARD_FUSION)}
    if counts != {"moon": expected, "fusion": expected}:
        raise RuntimeError(f"shard cache count mismatch: expected={expected} actual={counts}")
    receipt = {
        "schema": 1,
        "train_sha256": hashlib.sha256(SHARD_TRAIN.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(EVAL.read_bytes()).hexdigest(),
        "prefix_tokens": PREFIX_TOKENS,
        "moonvit_taps": ",".join(map(str, TAPS)),
        "view_mode": VIEW_MODE,
        "siglip2_model": str(SIGLIP.resolve()),
        "siglip2_width": 1152,
        "expected_entries": expected,
        "moon_cache": str(SHARD_MOON.resolve()),
        "fusion_cache": str(SHARD_FUSION.resolve()),
        "moon_cache_mtime_ns": SHARD_MOON.stat().st_mtime_ns,
        "fusion_cache_mtime_ns": SHARD_FUSION.stat().st_mtime_ns,
    }
    atomic_text(RECEIPT, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print({"kind": "handoff", "state": "ready", "receipt": str(RECEIPT),
           "entries": expected}, flush=True)


def main() -> None:
    if SHARD_MOON.exists() or SHARD_FUSION.exists():
        raise SystemExit("shard 000 cache already exists; refusing an ambiguous handoff")
    while True:
        completed = cache_count(ACTIVE_MOON)
        print({"kind": "handoff_wait", "completed": completed,
               "target": TARGET_TRAIN}, flush=True)
        if completed >= TARGET_TRAIN:
            break
        time.sleep(15)

    stop_active_cache()
    rows, selected = select_cached_rows()
    freeze_moon_shard(rows, selected)

    # Ensure the stable, non-adult eval set is represented in this first shard.
    run([sys.executable, "-m", "rwkv_lab.vision_cache",
         "--data", str(EVAL), "--cache", str(SHARD_MOON),
         "--prefix-tokens", str(PREFIX_TOKENS),
         "--tap-layers", ",".join(map(str, TAPS)),
         "--view-mode", VIEW_MODE, "--batch", "2", "--workers", "16",
         "--sort-window", "64"])

    fusion_command = [
        sys.executable, "-m", "rwkv_lab.vision_fusion_cache",
        "--data", str(SHARD_TRAIN), str(EVAL),
        "--cache", str(SHARD_FUSION), "--prefix-tokens", str(PREFIX_TOKENS),
        "--siglip2", str(SIGLIP), "--siglip2-width", "1152", "--batch", "16",
    ]
    try:
        run(fusion_command)
    except subprocess.CalledProcessError:
        print({"kind": "handoff", "action": "fusion_retry", "batch": 8}, flush=True)
        fusion_command[-1] = "8"
        run(fusion_command)
    write_receipt()


if __name__ == "__main__":
    main()
