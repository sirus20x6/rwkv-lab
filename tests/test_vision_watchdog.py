from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

import pytest


def load_watchdog():
    path = Path(__file__).resolve().parents[1] / "scripts/overnight_vision_watchdog.py"
    spec = importlib.util.spec_from_file_location("test_overnight_vision_watchdog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloads_only_never_touches_training_or_inference(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    data = tmp_path / "data"
    run.mkdir()
    data.mkdir()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "DATA", data)
    monkeypatch.setattr(
        watchdog,
        "training_state",
        lambda: ({"state": "failed", "step": 17617}, 17617),
    )
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    monkeypatch.setattr(watchdog, "acquisition_complete", lambda *args: True)
    monkeypatch.setattr(watchdog, "acquisition_summary", lambda *args: {})
    monkeypatch.setattr(
        watchdog, "acquisition_progress_signature", lambda *args, **kwargs: ((), ())
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("download-only supervision touched training or inference")

    monkeypatch.setattr(watchdog, "start_training", forbidden)
    monkeypatch.setattr(watchdog, "finish_or_start_smoke", forbidden)
    monkeypatch.setattr(watchdog, "start_hf_download", forbidden)
    monkeypatch.setattr(watchdog, "start_archive_download", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["overnight_vision_watchdog.py", "--downloads-only", "--once"],
    )

    assert watchdog.main() == 0
    status = json.loads((run / "overnight_status.json").read_text())
    assert status["training"]["supervision"] == "paused"
    assert status["training"]["trainer_pids"] == []
    assert status["caption_smoke"] == "disabled_while_training_paused"


def test_missing_exact_checkpoint_blocks_watchdog_restart(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    data = tmp_path / "data"
    run.mkdir()
    data.mkdir()
    (run / "train.jsonl").write_text('{"kind":"train","step":12}\n')
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "DATA", data)
    monkeypatch.setattr(
        watchdog,
        "training_state",
        lambda: ({"state": "failed", "step": 12}, 12),
    )
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    monkeypatch.setattr(watchdog, "acquisition_complete", lambda *args: True)
    monkeypatch.setattr(watchdog, "acquisition_summary", lambda *args: {})
    monkeypatch.setattr(
        watchdog, "acquisition_progress_signature", lambda *args, **kwargs: ((), ())
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("watchdog discarded an existing run")

    monkeypatch.setattr(watchdog, "start_training", forbidden)
    monkeypatch.setattr(
        watchdog, "finish_or_start_smoke", lambda *args, **kwargs: "waiting")
    monkeypatch.setattr(sys, "argv", ["overnight_vision_watchdog.py", "--once"])

    assert watchdog.main() == 0
    heartbeat = json.loads((run / "overnight_status.json").read_text())
    assert heartbeat["training"]["supervision"] == \
        "blocked_missing_exact_checkpoint"


def test_finish_target_uses_finish_environment_and_complete_state_is_terminal(monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setenv("VISION_TARGET_STEPS", "1000")
    monkeypatch.setenv("VISION_FINISH_TARGET_STEPS", "250")
    assert watchdog.resolve_target_steps(
        None, "scripts/run_vision_finish_grounded.sh") == 250
    assert watchdog.resolve_target_steps(
        None, "scripts/run_vision_i1_next.sh") == 1000
    assert watchdog.resolve_target_steps(75, "anything.sh") == 75
    assert not watchdog.training_is_done({"state": "complete"}, 50, 1000)
    assert watchdog.training_is_done({"state": "complete"}, 1000, 1000)
    assert not watchdog.training_is_done(
        {"state": "complete", "checkpoint_step": 999}, 1000, 1000)
    assert not watchdog.training_is_done({"state": "training"}, 1000, 1000)
    assert watchdog.training_is_done({"state": "paused"}, 1000, 1000)
    assert not watchdog.training_is_done(
        {"state": "paused", "checkpoint_step": 999}, 1001, 1000)
    assert watchdog.training_is_done(
        {"state": "failed", "checkpoint_step": 1000}, 1001, 1000)
    assert not watchdog.training_is_done({"state": "paused"}, 50, 1000)


def test_known_launcher_contracts_bind_run_and_eval_paths(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    launcher = tmp_path / "scripts/run_vision_finish_grounded.sh"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    run, evaluation = watchdog.launcher_contract(
        str(launcher))
    assert run == tmp_path / "runs/moonvit_rwkv_finish_grounded"
    assert evaluation == tmp_path / "curated_vision/vision_finish_grounded_eval.jsonl"
    assert watchdog.launcher_contract(
        "/somewhere/run_vision_finish_grounded.sh") is None
    assert watchdog.launcher_contract("custom_launcher.sh") is None


def test_finish_contract_adopts_environment_overrides(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    launcher = tmp_path / "scripts/run_vision_finish_grounded.sh"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VISION_FINISH_RUN", "runs/custom_finish")
    monkeypatch.setenv(
        "VISION_FINISH_EVAL", str(tmp_path / "eval/custom_eval.jsonl"))
    run, evaluation = watchdog.launcher_contract(str(launcher))
    assert run == tmp_path / "runs/custom_finish"
    assert evaluation == tmp_path / "eval/custom_eval.jsonl"

    monkeypatch.delenv("VISION_FINISH_RUN")
    monkeypatch.delenv("VISION_FINISH_EVAL")
    run, evaluation = watchdog.launcher_contract(str(launcher))
    assert run == tmp_path / "runs/moonvit_rwkv_finish_grounded"
    assert evaluation == tmp_path / "curated_vision/vision_finish_grounded_eval.jsonl"

    # Other launchers have no environment contract to adopt.
    other = tmp_path / "scripts/run_vision_i1_next.sh"
    other.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VISION_FINISH_RUN", "runs/custom_finish")
    run, _ = watchdog.launcher_contract(str(other))
    assert run == tmp_path / "runs/moonvit_rwkv_i1_civitai_phase2"


def test_launcher_needle_matches_all_operator_spellings(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    needle = "./scripts/run_vision_i1_next.sh"
    assert watchdog._needle_matches(needle, "./scripts/run_vision_i1_next.sh")
    assert watchdog._needle_matches(needle, "scripts/run_vision_i1_next.sh")
    assert watchdog._needle_matches(
        needle, str(tmp_path / "scripts/run_vision_i1_next.sh"))
    assert not watchdog._needle_matches(
        needle, "otherscripts/run_vision_i1_next.sh")
    assert not watchdog._needle_matches(needle, "run_vision_i1_next.sh")
    absolute = str(tmp_path / "scripts/run_vision_i1_next.sh")
    assert watchdog._needle_matches(absolute, "scripts/run_vision_i1_next.sh")
    assert watchdog._needle_matches(absolute, "./scripts/run_vision_i1_next.sh")
    # Non-script needles keep substring semantics.
    assert watchdog._needle_matches("rwkv_lab.vision_train", "rwkv_lab.vision_train")
    assert not watchdog._needle_matches("rwkv_lab.vision_train", "--out")

    launcher = tmp_path / "scripts/run_vision_i1_next.sh"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    assert watchdog._is_expected_invocation(
        ["bash", "scripts/run_vision_i1_next.sh"], tmp_path, (needle,))
    assert watchdog._is_expected_invocation(
        ["bash", "./scripts/run_vision_i1_next.sh"], tmp_path, (needle,))


def _loop_harness(watchdog, tmp_path, monkeypatch, *, max_sleeps,
                  trainer_pids, launcher_pids, status, step):
    """Drive main() through several polls on a fake monotonic clock."""
    run = tmp_path / "run"
    data = tmp_path / "data"
    run.mkdir()
    data.mkdir()
    (run / "last.pt").write_bytes(b"exact")
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "DATA", data)
    monkeypatch.setattr(watchdog, "training_state", lambda: (status, step))

    def fake_processes(*needles):
        if "rwkv_lab.vision_train" in needles:
            return list(trainer_pids)
        if watchdog.TRAIN_SCRIPT in needles:
            return list(launcher_pids)
        return []

    monkeypatch.setattr(watchdog, "processes", fake_processes)
    monkeypatch.setattr(watchdog, "acquisition_complete", lambda *args: True)
    monkeypatch.setattr(watchdog, "acquisition_summary", lambda *args: {})
    monkeypatch.setattr(
        watchdog, "acquisition_progress_signature", lambda *args, **kwargs: ((), ()))
    monkeypatch.setattr(
        watchdog, "finish_or_start_smoke", lambda *args, **kwargs: "waiting")
    monkeypatch.setattr(
        watchdog, "start_training",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected restart")))
    io_ticks = iter(range(1_000_000))
    monkeypatch.setattr(
        watchdog, "launcher_progress_signature",
        lambda pids=None: (next(io_ticks),))
    clock = {"now": 0.0}
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: clock["now"])

    class LoopDone(Exception):
        pass

    sleeps = {"count": 0}

    def fake_sleep(seconds):
        clock["now"] += seconds
        sleeps["count"] += 1
        if sleeps["count"] >= max_sleeps:
            raise LoopDone

    monkeypatch.setattr(watchdog.time, "sleep", fake_sleep)
    return run, LoopDone


def test_trainer_hang_is_detected_despite_launcher_tree_io(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run, LoopDone = _loop_harness(
        watchdog, tmp_path, monkeypatch, max_sleeps=6,
        trainer_pids=[111], launcher_pids=[222],
        status={"state": "training", "step": 5, "updated": "t"}, step=5)
    signals = []

    class Signaled(Exception):
        pass

    def fake_kill(pid, sig):
        signals.append((pid, sig))
        raise Signaled

    monkeypatch.setattr(watchdog.os, "kill", fake_kill)
    monkeypatch.setattr(
        sys, "argv", ["overnight_vision_watchdog.py", "--poll", "1800"])
    with pytest.raises(Signaled):
        watchdog.main()
    # Launcher-tree I/O ticked every poll, yet the deadlocked trainer with no
    # step/status/log progress still accumulated staleness and was signaled.
    assert signals == [(111, signal.SIGINT)]


def test_cache_phase_launcher_io_still_counts_as_progress(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run, LoopDone = _loop_harness(
        watchdog, tmp_path, monkeypatch, max_sleeps=6,
        trainer_pids=[], launcher_pids=[222],
        status={"state": "loading_data"}, step=0)
    signals = []
    monkeypatch.setattr(
        watchdog.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        watchdog.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(
        sys, "argv", ["overnight_vision_watchdog.py", "--poll", "1800"])
    with pytest.raises(LoopDone):
        watchdog.main()
    assert signals == []


def test_completed_trainer_shutdown_escalates_once_without_rearming(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run, LoopDone = _loop_harness(
        watchdog, tmp_path, monkeypatch, max_sleeps=6,
        trainer_pids=[111], launcher_pids=[],
        status={"state": "complete", "step": 100}, step=100)
    signals = []
    monkeypatch.setattr(
        watchdog.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        sys, "argv",
        ["overnight_vision_watchdog.py", "--poll", "300",
         "--target-steps", "100"])
    with pytest.raises(LoopDone):
        watchdog.main()
    # One SIGTERM after the 300s completion grace, one SIGKILL escalation
    # after a further timeout, and nothing else: no 300s SIGTERM loop.
    assert signals == [(111, signal.SIGTERM), (111, signal.SIGKILL)]


def test_download_restart_uses_backoff_and_terminal_blocked_state(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    data = tmp_path / "data"
    run.mkdir()
    data.mkdir()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "DATA", data)
    monkeypatch.setattr(
        watchdog, "training_state", lambda: ({"state": "failed"}, 0))
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    monkeypatch.setattr(
        watchdog, "acquisition_complete",
        lambda path, expected: expected == watchdog.ARCHIVE_SOURCES)
    monkeypatch.setattr(watchdog, "acquisition_summary", lambda *args: {})
    monkeypatch.setattr(
        watchdog, "acquisition_progress_signature", lambda *args, **kwargs: ((), ()))
    monkeypatch.setattr(
        watchdog, "start_archive_download",
        lambda: (_ for _ in ()).throw(AssertionError("archive restart")))
    clock = {"now": 0.0}
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: clock["now"])
    starts = []
    monkeypatch.setattr(
        watchdog, "start_hf_download",
        lambda: starts.append(clock["now"]) or 999)

    class LoopDone(Exception):
        pass

    sleeps = {"count": 0}

    def fake_sleep(seconds):
        clock["now"] += seconds
        sleeps["count"] += 1
        if sleeps["count"] >= 130:
            raise LoopDone

    monkeypatch.setattr(watchdog.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        sys, "argv",
        ["overnight_vision_watchdog.py", "--downloads-only", "--poll", "30"])
    with pytest.raises(LoopDone):
        watchdog.main()
    # Exponentially spaced launches, then a terminal blocked state instead of
    # relaunching a deterministically failing fetcher every poll forever.
    assert starts == [0, 30, 90, 210, 450, 930]
    heartbeat = json.loads((run / "overnight_status.json").read_text())
    assert heartbeat["downloads"]["hf_restart_attempts"] == 6
    assert heartbeat["downloads"]["hf_supervision"] == "blocked_repeated_failures"
    assert heartbeat["downloads"]["archive_supervision"] == "enabled"
    events = [json.loads(line)["event"]
              for line in (run / "overnight_watchdog.jsonl").read_text().splitlines()]
    assert events.count("hf_download_blocked") == 1


def test_caption_smoke_failures_back_off_then_block_and_prune(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    checkpoint = best / "ckpt_step_00000200.pt"
    checkpoint.write_bytes(b"checkpoint")
    (best / "best.json").write_text(json.dumps({
        "step": 200, "loss": 1.0, "checkpoint": checkpoint.name,
    }))
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    events = []
    monkeypatch.setattr(
        watchdog, "event", lambda kind, **fields: events.append(kind))
    image = tmp_path / "held_out.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(watchdog, "held_out_image", lambda: image)
    spawned = []

    class FakeProcess:
        pid = 999

    monkeypatch.setattr(
        watchdog.subprocess, "Popen",
        lambda *args, **kwargs: spawned.append(args) or FakeProcess())
    clock = {"now": 0.0}
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: clock["now"])
    ticks = iter(range(1_000_000, 2_000_000))
    monkeypatch.setattr(watchdog.time, "time", lambda: next(ticks))

    # Two pre-existing failure receipts from an earlier night.
    for stamp in (100, 101):
        old = run / f"overnight_caption_smoke.failed-{stamp}.txt"
        old.write_text("old failure")
        os.utime(old, ns=(stamp * 10**9, stamp * 10**9))

    temporary = run / "overnight_caption_smoke.json.tmp"
    temporary.write_text("not json")

    # Failure 1: backoff instead of an immediate GPU respawn.
    assert watchdog.finish_or_start_smoke(True, []) == "smoke_retry_backoff"
    assert spawned == []
    clock["now"] = 59.0
    assert watchdog.finish_or_start_smoke(True, []) == "smoke_retry_backoff"
    assert spawned == []

    # Deterministic failure: every respawn leaves an invalid temporary.
    expected_backoffs = [300.0, 900.0, 3600.0]
    for expected_delay in expected_backoffs:
        clock["now"] += 3601.0
        assert watchdog.finish_or_start_smoke(True, []) == "running"
        assert temporary.exists()
        assert watchdog.finish_or_start_smoke(True, []) == "smoke_retry_backoff"
    clock["now"] += 3601.0
    assert watchdog.finish_or_start_smoke(True, []) == "running"
    assert watchdog.finish_or_start_smoke(True, []) == "smoke_blocked"
    assert len(spawned) == 4
    assert "caption_smoke_blocked" in events

    # Blocked is terminal for this checkpoint: no further respawns.
    clock["now"] += 10 * 3600.0
    assert watchdog.finish_or_start_smoke(True, []) == "smoke_blocked"
    assert len(spawned) == 4

    # Receipts are capped: five failures kept, the pre-existing ones pruned.
    receipts = sorted(run.glob("overnight_caption_smoke.failed-*.txt"))
    assert len(receipts) == 5
    assert run / "overnight_caption_smoke.failed-100.txt" not in receipts

    # A new best checkpoint is a fresh deterministic input: budget resets.
    replacement = best / "ckpt_step_00000300.pt"
    replacement.write_bytes(b"new checkpoint")
    (best / "best.json").write_text(json.dumps({
        "step": 300, "loss": 0.9, "checkpoint": replacement.name,
    }))
    assert watchdog.finish_or_start_smoke(True, []) == "running"
    assert len(spawned) == 5


def test_stall_escalation_reaches_sigkill_only_after_grace_periods():
    watchdog = load_watchdog()
    assert watchdog.stall_escalation_signal(0, None) == signal.SIGINT
    assert watchdog.stall_escalation_signal(1, 119) is None
    assert watchdog.stall_escalation_signal(1, 120) == signal.SIGTERM
    assert watchdog.stall_escalation_signal(2, 119) is None
    assert watchdog.stall_escalation_signal(2, 120) == signal.SIGKILL


def test_training_restart_backoff_is_exponential_and_bounded():
    watchdog = load_watchdog()
    assert watchdog.restart_backoff_seconds(1, 10) == 30
    assert watchdog.restart_backoff_seconds(2, 30) == 60
    assert watchdog.restart_backoff_seconds(7, 30) == 1800
    assert watchdog.restart_backoff_seconds(20, 30) == 1800


def test_fresh_process_gets_new_stall_grace_and_escalation_sequence(monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: 321.5)
    last_progress, signaled_at, stage = watchdog.fresh_stall_supervision()
    assert last_progress == 321.5
    assert signaled_at is None
    assert stage == 0
    assert watchdog.stall_escalation_signal(stage, signaled_at) == signal.SIGINT


def test_process_identity_rejects_searches_and_editors(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    run_name = "moonvit-run"
    monkeypatch.setattr(watchdog, "RUN", tmp_path / "runs" / run_name)
    assert watchdog._is_expected_invocation(
        ["python", "-m", "rwkv_lab.vision_train", "--out", f"runs/{run_name}"],
        tmp_path,
        ("rwkv_lab.vision_train", run_name),
    )
    assert not watchdog._is_expected_invocation(
        ["rg", "rwkv_lab.vision_train", run_name],
        tmp_path,
        ("rwkv_lab.vision_train", run_name),
    )
    assert not watchdog._is_expected_invocation(
        ["bash", "-m", "rwkv_lab.vision_train", "--out", f"runs/{run_name}"],
        tmp_path,
        ("rwkv_lab.vision_train", run_name),
    )
    assert not watchdog._is_expected_invocation(
        [
            "python", "-m", "rwkv_lab.vision_train",
            "--out", f"runs/{run_name}", "--out", "runs/other",
        ],
        tmp_path,
        ("rwkv_lab.vision_train", run_name),
    )
    assert not watchdog._is_expected_invocation(
        ["python", "-m", "rwkv_lab.vision_train", "--out", "runs/other"],
        tmp_path,
        ("rwkv_lab.vision_train", "other"),
    )

    launcher = tmp_path / "scripts/run_vision_i1_next.sh"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    assert watchdog._is_expected_invocation(
        ["bash", "./scripts/run_vision_i1_next.sh"],
        tmp_path,
        ("./scripts/run_vision_i1_next.sh",),
    )
    assert not watchdog._is_expected_invocation(
        ["vim", "./scripts/run_vision_i1_next.sh"],
        tmp_path,
        ("./scripts/run_vision_i1_next.sh",),
    )
    assert not watchdog._is_expected_invocation(
        ["python", "-c", "print('x')", "scripts/fetch_i1_sources.py"],
        tmp_path,
        ("scripts/fetch_i1_sources.py",),
    )


def test_launcher_progress_tracks_cache_and_log_without_walking_files(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    run.mkdir()
    cache.mkdir()
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "FEATURE_CACHE", cache)
    monkeypatch.setattr(
        watchdog, "_process_tree_io_signature", lambda pids: ((pids[0], 10),))
    before = watchdog.launcher_progress_signature([123])

    (cache / "feature.pt").write_bytes(b"feature")
    os.utime(cache, ns=(2_000_000_000, 2_000_000_000))
    after_cache = watchdog.launcher_progress_signature([123])
    assert after_cache != before

    (run / "overnight_trainer.log").write_text("prefill batch complete\n")
    after_log = watchdog.launcher_progress_signature([123])
    assert after_log != after_cache

    monkeypatch.setattr(
        watchdog, "_process_tree_io_signature", lambda pids: ((pids[0], 20),))
    assert watchdog.launcher_progress_signature([123]) != after_log


def test_existing_run_without_last_checkpoint_blocks_automatic_restart(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(watchdog, "RUN", run)
    assert not watchdog.exact_resume_is_missing()
    (run / "train.jsonl").write_text('{"kind":"train","step":1}\n')
    assert watchdog.exact_resume_is_missing()
    (run / "last.pt").write_bytes(b"exact")
    assert not watchdog.exact_resume_is_missing()


def test_startup_only_receipts_allow_retry_but_positive_steps_fail_closed(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()

    def classify(name, files):
        run = tmp_path / name
        run.mkdir()
        for relative, payload in files.items():
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        monkeypatch.setattr(watchdog, "RUN", run)
        return watchdog.classify_run(run)

    startup = classify("startup-status", {
        "status.json": json.dumps({
            "state": "failed", "previous_state": "loading_data",
        }),
        "config.json": "{}",
        ".trainer.lock": "",
    })
    assert startup.state == "startup"
    assert not watchdog.exact_resume_is_missing()

    startup_log = classify("startup-log", {
        "train.jsonl": json.dumps({"kind": "startup", "step": 0}) + "\n",
        "loop_rw.json": json.dumps({"step": 0}),
    })
    assert startup_log.state == "startup"
    assert not watchdog.exact_resume_is_missing()

    for state in ("training", "evaluating", "checkpointing", "failed"):
        evidence = classify(f"positive-{state}", {
            "status.json": json.dumps({"state": state, "step": 1}),
        })
        assert evidence.state == "committed"
        assert watchdog.exact_resume_is_missing()

    assert classify("positive-log", {
        "train.jsonl": json.dumps({"kind": "train", "step": 1}) + "\n",
    }).state == "committed"
    assert classify("positive-loop", {
        "loop_rw.json": json.dumps({"step": 1}),
    }).state == "committed"


def test_missing_step_statuses_fail_closed_except_known_startup_states(tmp_path):
    watchdog = load_watchdog()
    for index, status in enumerate((
        {},
        {"state": "training"},
        {"state": "evaluating"},
        {"state": "checkpointing"},
        {"state": "unknown"},
        {"state": "failed"},
    )):
        run = tmp_path / f"ambiguous-{index}"
        run.mkdir()
        (run / "status.json").write_text(json.dumps(status))
        assert watchdog.classify_run(run).state == "committed"

    for index, status in enumerate((
        {"state": "loading_data"},
        {"state": "loading_rwkv"},
        {"state": "loading_moonvit"},
        {"state": "failed", "previous_state": "loading_data"},
        {"state": "training", "step": 0},
        {"state": "failed", "previous_state": "training", "step": 0},
    )):
        run = tmp_path / f"startup-{index}"
        run.mkdir()
        (run / "status.json").write_text(json.dumps(status))
        assert watchdog.classify_run(run).state == "startup"


def test_preloop_checkpoint_requires_known_step_zero_contract(tmp_path):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    (run / "pre_loop.pt").write_bytes(b"opaque checkpoint")
    assert watchdog.classify_run(run).state == "committed"
    assert watchdog.classify_run(
        run, allow_step_zero_preloop=True).state == "startup"


def test_unknown_or_checkpoint_receipts_fail_closed_without_last(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    cases = {
        "malformed-status": ("status.json", "{"),
        "malformed-log": ("train.jsonl", "not json\n"),
        "checkpoint-temp": ("last.tmp", "partial"),
        "best-receipt": ("best/best.json", "{}"),
        "eval-receipt": ("eval_samples/step_00000001.json", "{}"),
        "eval-contract-reset": ("eval_contract_reset.json", "{}"),
        "eval-contract-reset-temp": ("eval_contract_reset.json.tmp", "{}"),
    }
    for name, (relative, payload) in cases.items():
        run = tmp_path / name
        path = run / relative
        path.parent.mkdir(parents=True)
        path.write_text(payload)
        monkeypatch.setattr(watchdog, "RUN", run)
        assert watchdog.classify_run(run).state == "committed"
        assert watchdog.exact_resume_is_missing()

    symlink_run = tmp_path / "symlink-last"
    symlink_run.mkdir()
    outside = tmp_path / "outside-last.pt"
    outside.write_bytes(b"checkpoint")
    (symlink_run / "last.pt").symlink_to(outside)
    assert watchdog.classify_run(symlink_run).state == "committed"


def test_fresh_eval_contract_receipt_remains_startup_only_until_step_one(tmp_path):
    watchdog = load_watchdog()
    fresh = {
        "schema": 1, "reset": True, "step": 0, "reasons": ["fresh"],
    }
    run = tmp_path / "fresh"
    run.mkdir()
    (run / "eval_contract_reset.json").write_text(json.dumps(fresh))
    (run / "status.json").write_text(json.dumps({"state": "loading_data"}))
    (run / "train.jsonl").write_text(
        json.dumps({"kind": "startup", "step": 0}) + "\n")
    assert watchdog.classify_run(run).state == "startup"

    (run / "train.jsonl").write_text(
        json.dumps({"kind": "train", "step": 1}) + "\n")
    assert watchdog.classify_run(run).state == "committed"

    temporary = tmp_path / "fresh-temporary"
    temporary.mkdir()
    (temporary / "eval_contract_reset.json.tmp").write_text(json.dumps(fresh))
    (temporary / "status.json").write_text(json.dumps({
        "state": "failed", "previous_state": "loading_data",
    }))
    assert watchdog.classify_run(temporary).state == "startup"

    invalid_receipts = (
        {**fresh, "schema": True},
        {**fresh, "schema": 2},
        {**fresh, "reset": False},
        {**fresh, "step": 1},
        {**fresh, "reasons": ["loop_reset"]},
        {**fresh, "reasons": ["fresh", "loop_reset"]},
        {**fresh, "reasons": ["loop_reset", "fresh"]},
        {**fresh, "unexpected": "field"},
    )
    for index, receipt in enumerate(invalid_receipts):
        other = tmp_path / f"invalid-reset-{index}"
        other.mkdir()
        (other / "eval_contract_reset.json").write_text(json.dumps(receipt))
        assert watchdog.classify_run(other).state == "committed"

    malformed = tmp_path / "malformed-reset"
    malformed.mkdir()
    (malformed / "eval_contract_reset.json").write_text("{")
    assert watchdog.classify_run(malformed).state == "committed"

    malformed_temporary = tmp_path / "malformed-reset-temporary"
    malformed_temporary.mkdir()
    (malformed_temporary / "eval_contract_reset.json.tmp").write_bytes(b"\xff")
    assert watchdog.classify_run(malformed_temporary).state == "committed"

    positive_temporary = tmp_path / "positive-reset-temporary"
    positive_temporary.mkdir()
    (positive_temporary / "eval_contract_reset.json.tmp").write_text(
        json.dumps({**fresh, "step": 1}))
    assert watchdog.classify_run(positive_temporary).state == "committed"


def test_terminal_receipt_without_last_checkpoint_is_not_restartable(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(watchdog, "RUN", run)
    assert watchdog.exact_resume_is_missing({"state": "complete", "step": 100})
    assert watchdog.exact_resume_is_missing({
        "state": "failed", "step": 100, "exact_checkpoint_saved": True,
    })
    assert not watchdog.exact_resume_is_missing({"state": "loading_data"})


def test_first_training_launch_prefills_cache_but_exact_resume_skips_walk(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    captured = []
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "event", lambda *args, **kwargs: None)
    monkeypatch.delenv("SKIP_CACHE_VERIFY", raising=False)

    def launch(command, log_path, *, env=None):
        captured.append(env)
        return 123

    monkeypatch.setattr(watchdog, "launch", launch)
    assert watchdog.start_training() == 123
    assert "SKIP_CACHE_VERIFY" not in captured[-1]

    (run / "last.pt").write_bytes(b"exact checkpoint")
    assert watchdog.start_training() == 123
    assert captured[-1]["SKIP_CACHE_VERIFY"] == "1"


def test_download_escalation_uses_the_same_bounded_signal_sequence(monkeypatch):
    watchdog = load_watchdog()
    sent = []
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    stage, at, sig = watchdog.escalate_stalled_processes(
        [12], stage=0, signaled_at=None)
    assert (stage, sig) == (1, signal.SIGINT) and at is not None
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: at + 120)
    stage, at, sig = watchdog.escalate_stalled_processes(
        [12], stage=stage, signaled_at=at)
    assert (stage, sig) == (2, signal.SIGTERM)
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: at + 120)
    stage, _, sig = watchdog.escalate_stalled_processes(
        [12], stage=stage, signaled_at=at)
    assert (stage, sig) == (3, signal.SIGKILL)
    assert [item[1] for item in sent] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_launcher_escalation_signals_only_verified_session_leaders(monkeypatch):
    watchdog = load_watchdog()
    sent = []
    monkeypatch.setattr(watchdog.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        watchdog.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    stage, _, sig = watchdog.escalate_stalled_processes(
        [42], stage=0, signaled_at=None, process_groups=True)
    assert (stage, sig) == (1, signal.SIGINT)
    assert sent == [(42, signal.SIGINT)]

    monkeypatch.setattr(watchdog.os, "getpgid", lambda pid: pid + 1)
    stage, _, sig = watchdog.escalate_stalled_processes(
        [42], stage=0, signaled_at=None, process_groups=True)
    assert (stage, sig) == (1, signal.SIGINT)
    assert sent == [(42, signal.SIGINT)]


def test_best_checkpoint_prefers_atomic_manifest_target(tmp_path):
    watchdog = load_watchdog()
    best = tmp_path / "best"
    best.mkdir()
    (best / "ckpt.pt").write_bytes(b"old")
    immutable = best / "ckpt_step_00000200.pt"
    immutable.write_bytes(b"new")
    (best / "best.json").write_text(json.dumps({
        "step": 200, "loss": 1.0, "checkpoint": immutable.name,
    }))
    assert watchdog.best_checkpoint(tmp_path) == immutable


def test_best_checkpoint_rejects_manifest_path_escape(tmp_path):
    watchdog = load_watchdog()
    best = tmp_path / "best"
    best.mkdir()
    (best / "ckpt.pt").write_bytes(b"legacy")
    (best / "best.json").write_text(json.dumps({
        "step": 1, "loss": 1.0, "checkpoint": "../other.pt",
    }))
    assert watchdog.best_checkpoint(tmp_path) is None


def test_best_checkpoint_rejects_manifest_and_temporary_targets(tmp_path):
    watchdog = load_watchdog()
    best = tmp_path / "best"
    best.mkdir()
    for name in ("", "best.json", "candidate.bin", ".candidate.pt.tmp"):
        if name and name != "best.json":
            (best / name).write_bytes(b"not a checkpoint")
        (best / "best.json").write_text(json.dumps({
            "step": 1, "loss": 1.0, "checkpoint": name,
        }))
        resolved = watchdog.resolve_best_checkpoint(tmp_path)
        assert resolved.state == "invalid"
        assert resolved.checkpoint is None


def test_best_checkpoint_legacy_and_malformed_manifest_behavior(tmp_path):
    watchdog = load_watchdog()
    best = tmp_path / "best"
    best.mkdir()
    legacy = best / "ckpt.pt"
    legacy.write_bytes(b"legacy")

    # A missing manifest is the original legacy layout.
    assert watchdog.best_checkpoint(tmp_path) == legacy
    assert watchdog.resolve_best_checkpoint(tmp_path).state == "valid"
    (best / "best.json").write_text(json.dumps({
        "step": 12, "loss": 1.5, "ppl": 4.48,
    }))
    assert watchdog.best_checkpoint(tmp_path) == legacy

    for malformed in ("{", "{}", "[]"):
        (best / "best.json").write_text(malformed)
        assert watchdog.best_checkpoint(tmp_path) is None
        assert watchdog.resolve_best_checkpoint(tmp_path).state == "invalid"

    (best / "best.json").write_text(json.dumps({
        "step": 12, "loss": 1.0, "checkpoint": "missing.pt",
    }))
    assert watchdog.best_checkpoint(tmp_path) is None

    (best / "best.json").write_text(json.dumps({
        "checkpoint": legacy.name,
    }))
    assert watchdog.best_checkpoint(tmp_path) is None

    (best / "best.json").unlink()
    legacy.unlink()
    assert watchdog.resolve_best_checkpoint(tmp_path).state == "invalid"
    best.rmdir()
    assert watchdog.resolve_best_checkpoint(tmp_path).state == "absent"

    best.mkdir()
    legacy.write_bytes(b"legacy")
    (best / "best.json.tmp").write_text("{}")
    assert watchdog.resolve_best_checkpoint(tmp_path).state == "invalid"


def test_best_checkpoint_rejects_declared_symlink_escape(tmp_path):
    watchdog = load_watchdog()
    best = tmp_path / "best"
    best.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (best / "linked.pt").symlink_to(outside)
    (best / "best.json").write_text(json.dumps({
        "step": 12, "loss": 1.0, "checkpoint": "linked.pt",
    }))
    assert watchdog.best_checkpoint(tmp_path) is None

    (best / "best.json").unlink()
    outside_manifest = tmp_path / "outside.json"
    outside_manifest.write_text(json.dumps({"step": 12, "loss": 1.0}))
    (best / "ckpt.pt").write_bytes(b"legacy")
    (best / "best.json").symlink_to(outside_manifest)
    assert watchdog.resolve_best_checkpoint(tmp_path).state == "invalid"

    escaped_run = tmp_path / "escaped-run"
    escaped_run.mkdir()
    (escaped_run / "best").symlink_to(best, target_is_directory=True)
    assert watchdog.resolve_best_checkpoint(escaped_run).state == "invalid"


def test_best_checkpoint_default_follows_runtime_run_override(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "custom-run"
    best = run / "best"
    best.mkdir(parents=True)
    checkpoint = best / "ckpt_step_00000200.pt"
    checkpoint.write_bytes(b"new")
    (best / "best.json").write_text(json.dumps({
        "step": 200, "loss": 1.0, "checkpoint": checkpoint.name,
    }))
    monkeypatch.setattr(watchdog, "RUN", run)
    assert watchdog.best_checkpoint() == checkpoint


def test_smoke_must_match_current_checkpoint_and_step(tmp_path):
    watchdog = load_watchdog()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({
        "checkpoint": str(checkpoint), "step": 200, "caption": "A caption.",
    }))
    assert watchdog.valid_smoke(
        smoke, checkpoint=checkpoint, expected_step=200)
    assert not watchdog.valid_smoke(
        smoke, checkpoint=checkpoint, expected_step=201)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    assert not watchdog.valid_smoke(
        smoke, checkpoint=other, expected_step=200)


def test_smoke_is_not_complete_while_training_is_incomplete(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    (run / "overnight_caption_smoke.json").write_text(json.dumps({
        "checkpoint": str(run / "old.pt"), "step": 100,
        "caption": "An old caption.",
    }))
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [77])
    stopped = []
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, sig: stopped.append((pid, sig)))
    monkeypatch.setattr(watchdog, "event", lambda *args, **kwargs: None)
    assert watchdog.finish_or_start_smoke(False, []) == "waiting_for_training"
    assert stopped == [(77, signal.SIGTERM)]


def test_smoke_waits_for_launcher_to_release_training_session(tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    assert watchdog.finish_or_start_smoke(
        True, [], launcher_pids=[88]) == "waiting_for_training"


def test_smoke_does_not_mask_invalid_best_publication_with_last(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    (best / "best.json").write_text("{}")
    (run / "last.pt").write_bytes(b"last")
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [])
    assert watchdog.finish_or_start_smoke(
        True, [], launcher_pids=[]) == "blocked_invalid_best_checkpoint"


def test_stalled_caption_smoke_escalates_to_verified_process_group(
        tmp_path, monkeypatch):
    watchdog = load_watchdog()
    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    checkpoint = best / "ckpt_step_00000200.pt"
    checkpoint.write_bytes(b"checkpoint")
    (best / "best.json").write_text(json.dumps({
        "step": 200, "loss": 1.0, "checkpoint": checkpoint.name,
    }))
    monkeypatch.setattr(watchdog, "RUN", run)
    monkeypatch.setattr(watchdog, "processes", lambda *needles: [55])
    monkeypatch.setattr(
        watchdog, "process_age_seconds",
        lambda pid: watchdog.SMOKE_STALE_SECONDS + watchdog.SMOKE_KILL_GRACE_SECONDS,
    )
    monkeypatch.setattr(watchdog.os, "getpgid", lambda pid: pid)
    sent = []
    monkeypatch.setattr(
        watchdog.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    monkeypatch.setattr(watchdog, "event", lambda *args, **kwargs: None)

    assert watchdog.finish_or_start_smoke(True, []) == "stopping_stalled"
    assert sent == [(55, signal.SIGKILL)]


def test_hf_progress_signature_tracks_nested_incomplete_files(tmp_path):
    watchdog = load_watchdog()
    state = tmp_path / "acquisition_state.json"
    state.write_text(json.dumps({
        "sources": {"source": {"downloaded_bytes": 10, "status": "downloading"}},
    }))
    partial = (tmp_path / "source" / ".cache" / "huggingface" / "download"
               / "nested" / "shard.incomplete")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"first")

    before = watchdog.acquisition_progress_signature(state, tmp_path)
    partial.write_bytes(b"a larger partial payload")
    after = watchdog.acquisition_progress_signature(state, tmp_path)

    assert before != after
    assert after[1][0][0].endswith("nested/shard.incomplete")


def test_corrupt_acquisition_receipts_do_not_crash_supervision(tmp_path):
    watchdog = load_watchdog()
    state = tmp_path / "acquisition_state.json"
    state.write_bytes(b"\xff")
    assert watchdog.acquisition_summary(state)["sources"] == {}
    assert not watchdog.acquisition_complete(state, {"source"})
    assert not watchdog.acquisition_waiting_for_space(state)
    assert watchdog.acquisition_progress_signature(state, tmp_path) == ((), ())

    state.write_text(json.dumps({"sources": []}))
    assert watchdog.acquisition_summary(state)["sources"] == {}
    assert not watchdog.acquisition_complete(state, {"source"})
    assert not watchdog.acquisition_waiting_for_space(state)

    state.write_text(json.dumps({
        "sources": {
            "source": {
                "expected_bytes": "invalid",
                "downloaded_bytes": {"invalid": True},
                "status": "downloading",
            },
        },
    }))
    assert watchdog.acquisition_summary(state)["sources"]["source"] == {
        "status": "downloading",
        "downloaded_bytes": 0,
        "expected_bytes": 0,
        "percent": None,
    }
    assert watchdog.acquisition_progress_signature(state, tmp_path)[0] == (
        ("source", 0, "downloading"),
    )
