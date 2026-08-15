#!/usr/bin/env python3
"""Keep teacher caching, deduplication, and compressor training moving unattended."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "curated_vision"
CACHE = Path("/workspace/downloads/cache/moe-mla")
STATE_PATH = ROOT / "runs/overnight_teacher_student_supervisor/state.json"
OCR_TRAIN = CURATED / "vision_next_ocr10_train.jsonl"
EVAL = CURATED / "vision_next_shard_000_ocr10_eval.jsonl"
DEDUP_DB = CACHE / "local_private_web_image_dedup.sqlite"
DEDUP_MANIFEST = CURATED / "local_private_web_unlabeled_dedup.jsonl"
DEDUP_POLICY = CURATED / "local_private_web_dedupe_policy.json"
POLL_SECONDS = 20
COMPRESSOR_FROZEN = ROOT / "runs/overnight_teacher_student_supervisor/COMPRESSOR_FROZEN"


def emit(kind: str, **values: object) -> None:
    print(json.dumps({"kind": kind, "time": time.time(), **values}, sort_keys=True),
          flush=True)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=check)


def active(unit: str) -> bool:
    return run(["systemctl", "--user", "is-active", unit], check=False).returncode == 0


def any_process(pattern: str) -> bool:
    result = run(["pgrep", "-f", pattern], check=False)
    return result.returncode == 0


def start_unit(unit: str, command: list[str], env: dict[str, str] | None = None) -> None:
    args = ["systemd-run", "--user", f"--unit={unit}", "--collect",
            "--property=Restart=on-failure", "--property=RestartSec=20",
            f"--working-directory={ROOT}"]
    for key, value in (env or {}).items():
        args.append(f"--setenv={key}={value}")
    args.extend(command)
    result = run(args)
    emit("unit_started", unit=unit, output=result.stdout.strip())


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {
        "schema": 1, "training_stage": 0,
        "training_unit": "rwkv-vision-teacher-compressor",
        "training_run": "runs/vision_teacher_compressor_so400m_ocr10",
        "training_target": 20000,
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE_PATH)


def cache_unit_running() -> bool:
    result = run(["systemctl", "--user", "list-units", "--state=running",
                  "--plain", "--no-legend", "rwkv-vision-cache-*"], check=False)
    return any(line.strip() for line in result.stdout.splitlines())


def maintain_dedupe() -> None:
    if DEDUP_MANIFEST.is_file() or any_process("build_unlabeled_image_manifest.py"):
        return
    if not DEDUP_DB.is_file():
        emit("dedupe_waiting", reason="database_missing")
        return
    with sqlite3.connect(f"file:{DEDUP_DB}?mode=ro", uri=True, timeout=10) as db:
        pending = db.execute(
            "SELECT count(*) FROM files WHERE excluded_reason IS NULL "
            "AND duplicate_of IS NULL AND content_hash IS NOT NULL "
            "AND phash IS NULL AND decode_error IS NULL").fetchone()[0]
    if pending:
        unit = "rwkv-local-private_web-dedup-hash"
        if not active(unit):
            start_unit(unit, [str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/build_unlabeled_image_manifest.py"),
                "--root", "/workspace/datasets/private_web", "--db", str(DEDUP_DB),
                "--manifest", str(DEDUP_MANIFEST), "--phase", "hash"])
            emit("dedupe_resumed", pending=pending)
    else:
        if not DEDUP_POLICY.is_file():
            emit("dedupe_waiting", reason="human_cutoff_policy_required",
                 review_url="http://127.0.0.1:9126")
            return
        unit = "rwkv-local-private_web-dedup-finalize"
        if not active(unit):
            start_unit(unit, [str(ROOT / "scripts/finish_local_private_web_dedupe.sh")])
            emit("dedupe_finalize_started")


def maintain_cache() -> None:
    if cache_unit_running():
        return
    local_receipt = CURATED / "local_private_web_teacher_shard_000.cache.json"
    if DEDUP_MANIFEST.is_file() and not local_receipt.is_file():
        start_unit("rwkv-vision-cache-local-private_web-000",
                   [str(ROOT / "scripts/cache_local_private_web_teacher_shard.sh")])
        return
    for index in range(1, 10):
        tag = f"{index:03d}"
        receipt = CURATED / f"vision_next_so400m_128_shard_{tag}.cache.json"
        if not receipt.is_file():
            start_unit(f"rwkv-vision-cache-shard{tag}",
                       [str(ROOT / "scripts/cache_vision_teacher_shard.sh"), str(index)])
            return
    emit("cache_corpus_complete", shards=10)


def latest_step(metrics: Path) -> int:
    result = 0
    if metrics.is_file():
        with metrics.open() as handle:
            for line in handle:
                try:
                    result = max(result, int(json.loads(line).get("step", 0)))
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
    return result


def prepare_stage(index: int) -> tuple[Path, Path, Path, Path]:
    tag = f"{index:03d}"
    shard = CURATED / f"vision_next_shard_{tag}_train.jsonl"
    train = CURATED / f"vision_next_shard_{tag}_ocr10_train.jsonl"
    moon = CACHE / f"moonvit_next_128_shard_{tag}_ocr10"
    fusion = CACHE / f"fusion_so400m_next_128_shard_{tag}_ocr10"
    selected_receipt = CURATED / f"vision_next_so400m_128_shard_{tag}_ocr10.cache.json"
    if not train.is_file():
        result = run([str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/combine_vision_manifests.py"),
            "--input", str(shard), "--input", str(OCR_TRAIN),
            "--output", str(train)])
        emit("training_manifest_ready", stage=index, output=result.stdout.strip())
    if not selected_receipt.is_file():
        result = run([str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/assemble_selected_vision_cache.py"),
            "--train", str(train), "--eval", str(EVAL),
            "--moon-source", str(CACHE / f"moonvit_next_128_shard_{tag}"),
            "--moon-source", str(CACHE / "moonvit_next_128_ocr10"),
            "--moon-source", str(CACHE / "moonvit_next_128_shard_000"),
            "--fusion-source", str(CACHE / f"fusion_so400m_next_128_shard_{tag}"),
            "--fusion-source", str(CACHE / "fusion_so400m_next_128_ocr10"),
            "--fusion-source", str(CACHE / "fusion_so400m_next_128_shard_000"),
            "--moon-output", str(moon), "--fusion-output", str(fusion),
            "--receipt", str(selected_receipt)])
        emit("training_cache_view_ready", stage=index, output=result.stdout.strip())
    return train, EVAL, moon, fusion


def launch_training(state: dict, stage: int, *, transfer: bool) -> None:
    train, evaluation, moon, fusion = prepare_stage(stage) if stage else (
        CURATED / "vision_next_shard_000_ocr10_train.jsonl", EVAL,
        CACHE / "moonvit_next_128_shard_000_ocr10",
        CACHE / "fusion_so400m_next_128_shard_000_ocr10")
    old_run = ROOT / state["training_run"]
    source_step = latest_step(old_run / "train.jsonl")
    target = max(source_step, int(state.get("training_target", 0))) + 5000
    tag = f"{stage:03d}"
    run_path = ROOT / f"runs/vision_teacher_compressor_so400m_ocr10_shard_{tag}"
    if stage == 0:
        run_path = old_run
    unit = f"rwkv-vision-compressor-stage{tag}"
    env = {
        "VISION_COMPRESSOR_RUN": str(run_path),
        "VISION_COMPRESSOR_TRAIN": str(train),
        "VISION_COMPRESSOR_EVAL": str(evaluation),
        "VISION_COMPRESSOR_MOON_CACHE": str(moon),
        "VISION_COMPRESSOR_FUSION_CACHE": str(fusion),
        "VISION_COMPRESSOR_STEPS": str(target),
        "VISION_COMPRESSOR_WORKERS": "8",
    }
    if transfer:
        checkpoint = old_run / "best.pt"
        if not checkpoint.is_file():
            checkpoint = old_run / "last.pt"
        env["VISION_COMPRESSOR_INIT_FROM"] = str(checkpoint)
    start_unit(unit, [str(ROOT / "scripts/run_vision_teacher_compressor.sh")], env)
    state.update(training_stage=stage, training_unit=unit,
                 training_run=str(run_path.relative_to(ROOT)), training_target=target)
    save_state(state)
    emit("training_started", stage=stage, target=target, transfer=transfer)


def maintain_training(state: dict) -> None:
    if COMPRESSOR_FROZEN.is_file():
        return
    stage = int(state["training_stage"])
    next_stage = stage + 1
    next_ready = next_stage < 10 and (
        CURATED / f"vision_next_so400m_128_shard_{next_stage:03d}.cache.json").is_file()
    unit = str(state["training_unit"])
    trainer_running = any_process("rwkv_lab.vision_teacher_compressor")
    if next_ready:
        if trainer_running:
            result = run(["systemctl", "--user", "stop", unit], check=False)
            emit("training_handoff_stop", stage=stage, unit=unit,
                 returncode=result.returncode)
            trainer_running = any_process("rwkv_lab.vision_teacher_compressor")
        if not trainer_running:
            launch_training(state, next_stage, transfer=True)
        return
    if not trainer_running:
        launch_training(state, stage, transfer=False)


def main() -> None:
    state = load_state()
    save_state(state)
    emit("overnight_supervisor_started", state=state)
    while True:
        try:
            maintain_dedupe()
            maintain_cache()
            maintain_training(state)
            state.update(
                heartbeat=time.time(),
                trainer_running=any_process("rwkv_lab.vision_teacher_compressor"),
                cache_running=cache_unit_running(),
                dedupe_running=(DEDUP_MANIFEST.is_file() or
                                any_process("build_unlabeled_image_manifest.py")),
                observed_training_step=latest_step(ROOT / state["training_run"] / "train.jsonl"),
                last_error=None,
            )
            save_state(state)
        except Exception as error:
            state.update(heartbeat=time.time(), last_error=repr(error))
            save_state(state)
            emit("supervisor_error", error=repr(error))
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
