#!/usr/bin/env python3
"""Keep phase-2 vision training and i1 acquisition alive unattended.

The watchdog is deliberately independent of the trainer. It resumes from the
trainer's atomic ``last.pt``, restarts both resumable download queues, and
records a small heartbeat for the dashboard/operator. Training is continuous;
the large finite bound exists only because the trainer requires one.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from vision_run_evidence import classify_run, resolve_best_checkpoint  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/moonvit_rwkv_i1_civitai_phase2"
DATA = ROOT / "datasets/i1_full_sources"
TRAIN_SCRIPT = "./scripts/run_vision_i1_next.sh"
EVAL_MANIFEST = ROOT / "curated_vision/vision_next_i1_civitai_eval.jsonl"
TARGET_STEPS = int(os.environ.get("VISION_TARGET_STEPS", "1000000000"))
TRAIN_STALE_SECONDS = 30 * 60
DOWNLOAD_STALE_SECONDS = 30 * 60
SMOKE_STALE_SECONDS = 30 * 60
SMOKE_KILL_GRACE_SECONDS = 2 * 60
HF_SOURCES = {
    "pexels", "midjourneyv6", "fluxreason", "imagenet22k",
    "megalith10m", "gptedit", "textatlas", "rendered_text",
}
ARCHIVE_SOURCES = {"yfcc_metadata", "inaturalist", "places365"}
FEATURE_CACHE = ROOT / "caches/moonvit_features_stage1_v3"
LAUNCHER_CONTRACTS = {
    "run_vision_i1_next.sh": (
        "runs/moonvit_rwkv_i1_civitai_phase2",
        "curated_vision/vision_next_i1_civitai_eval.jsonl",
    ),
    "run_vision_finish_grounded.sh": (
        "runs/moonvit_rwkv_finish_grounded",
        "curated_vision/vision_finish_grounded_eval.jsonl",
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event(kind: str, **fields: Any) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    record = {"time": now(), "event": kind, **fields}
    with (RUN / "overnight_watchdog.jsonl").open("a", buffering=1) as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def _is_expected_invocation(
    argv: list[str], cwd: Path, needles: tuple[str, ...],
) -> bool:
    """Reject argv text matches that are not the process type we launch.

    A direct ``rg rwkv_lab.vision_train`` or an editor opened on a launcher used
    to satisfy the substring-only matcher.  That can suppress recovery forever
    or, worse, make stall escalation signal an operator process.  All supervised
    commands have one of two explicit shapes: ``python -m module`` or an
    interpreter followed by a script path.
    """
    modules = [needle for needle in needles if needle.startswith("rwkv_lab.")]
    if modules:
        executable = Path(argv[0]).name if argv else ""
        module_positions = [
            index for index in range(len(argv) - 1)
            if argv[index] == "-m" and argv[index + 1] == modules[0]
        ]
        if (len(modules) != 1 or not executable.startswith("python")
                or len(module_positions) != 1):
            return False
        module = modules[0]
        if module == "rwkv_lab.vision_train":
            if argv.count("--out") != 1:
                return False
            try:
                output = Path(argv[argv.index("--out") + 1])
            except (ValueError, IndexError):
                return False
            if not output.is_absolute():
                output = cwd / output
            try:
                if output.resolve() != RUN.resolve():
                    return False
            except OSError:
                return False
        elif module == "rwkv_lab.vision_caption":
            if argv.count("--checkpoint") != 1:
                return False
            try:
                checkpoint = Path(argv[argv.index("--checkpoint") + 1])
            except (ValueError, IndexError):
                return False
            if not checkpoint.is_absolute():
                checkpoint = cwd / checkpoint
            try:
                checkpoint.resolve().relative_to(RUN.resolve())
            except (OSError, ValueError):
                return False

    scripts = [needle for needle in needles
               if Path(needle).suffix in {".py", ".sh"}]
    for expected in scripts:
        suffix = Path(expected).suffix
        executable = Path(argv[0]).name
        if suffix == ".py" and "python" not in executable:
            return False
        if suffix == ".sh" and executable not in {"bash", "dash", "sh", "zsh"}:
            return False
        if len(argv) < 2:
            return False
        expected_path = Path(expected)
        if not expected_path.is_absolute():
            expected_path = ROOT / expected_path
        actual_path = Path(argv[1])
        if not actual_path.is_absolute():
            actual_path = cwd / actual_path
        try:
            if actual_path.resolve() != expected_path.resolve():
                return False
        except OSError:
            return False
    return True


def processes(*needles: str) -> list[int]:
    """Return PIDs whose argv tokens contain every requested identifier.

    Do not search shell ``-c`` program text: an operator command such as
    ``ps | rg rwkv_lab.vision_train`` must never impersonate the trainer and
    suppress recovery.
    """
    result: list[int] = []
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            argv = [token.decode("utf-8", errors="replace") for token in
                    (entry / "cmdline").read_bytes().split(b"\0") if token]
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        searchable = [token for token in argv if not any(char.isspace() for char in token)]
        if (searchable and _is_expected_invocation(argv, cwd, needles)
                and all(any(needle in token for token in searchable)
                        for needle in needles)):
            result.append(pid)
    return sorted(result)


def launch(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def low_priority(command: list[str]) -> list[str]:
    prefix: list[str] = []
    nice = shutil.which("nice")
    ionice = shutil.which("ionice")
    if nice:
        prefix += [nice, "-n", "15"]
    if ionice:
        prefix += [ionice, "-c", "3"]
    return prefix + command


def training_state() -> tuple[dict[str, Any], int]:
    status = read_json(RUN / "status.json")
    try:
        return status, int(status.get("step", 0))
    except (TypeError, ValueError):
        return status, 0


def resolve_target_steps(explicit: int | None, train_script: str) -> int:
    """Use the same environment contract as the selected launcher."""
    if explicit is not None:
        target = explicit
    elif Path(train_script).name == "run_vision_finish_grounded.sh":
        target = int(os.environ.get(
            "VISION_FINISH_TARGET_STEPS",
            os.environ.get("VISION_TARGET_STEPS", "1000000000"),
        ))
    else:
        target = int(os.environ.get("VISION_TARGET_STEPS", "1000000000"))
    if target < 1:
        raise ValueError("target steps must be positive")
    return target


def launcher_contract(train_script: str) -> tuple[Path, Path] | None:
    """Return the run and eval paths hard-coded by a known launcher."""
    candidate = Path(train_script)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    name = candidate.name
    contract = LAUNCHER_CONTRACTS.get(name)
    if contract is None:
        return None
    try:
        if candidate.resolve() != (ROOT / "scripts" / name).resolve():
            return None
    except OSError:
        return None
    return ROOT / contract[0], ROOT / contract[1]


def training_is_done(status: dict[str, Any], step: int, target_steps: int) -> bool:
    try:
        checkpoint_step = int(status["checkpoint_step"])
    except (KeyError, TypeError, ValueError):
        checkpoint_step = None
    if checkpoint_step is not None:
        return checkpoint_step >= target_steps
    # The trainer writes complete/paused only after its exact terminal or
    # interrupt checkpoint, and operator-authored stopped states likewise
    # attest that the displayed step is preserved. Still require the configured
    # target: a stale ``complete`` receipt from an earlier, shorter run must not
    # prevent an operator from extending training to a higher target.
    # A stale/nonterminal "training" step is not a completion receipt: restart
    # it so final checkpoint/status publication can finish.
    return status.get("state") in {"complete", "paused", "stopped"} and step >= target_steps


def stall_escalation_signal(stage: int, seconds_since_signal: float | None) -> int | None:
    if stage == 0:
        return signal.SIGINT
    if seconds_since_signal is None or seconds_since_signal < 120:
        return None
    if stage == 1:
        return signal.SIGTERM
    return signal.SIGKILL


def restart_backoff_seconds(attempt: int, poll_seconds: int) -> int:
    """Bound deterministic launcher failures without giving up recovery."""
    return min(30 * 60, max(30, int(poll_seconds)) * 2 ** max(0, attempt - 1))


def fresh_stall_supervision() -> tuple[float, None, int]:
    """Give a newly launched process a fresh progress/escalation window."""
    return time.monotonic(), None, 0


def _process_tree_io_signature(roots: list[int]) -> tuple[tuple[int, int, ...], ...]:
    """Read cumulative I/O for a small launched process tree from procfs."""
    supervised_groups: set[int] = set()
    for pid in roots:
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            continue
        # launch(..., start_new_session=True) makes the returned PID the group
        # leader. Refuse to follow a PID which has already been reused or moved.
        if pgid == pid:
            supervised_groups.add(pgid)
    pending = list(roots)
    seen: set[int] = set()
    signature: list[tuple[int, int, ...]] = []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            if os.getpgid(pid) not in supervised_groups:
                continue
        except (ProcessLookupError, PermissionError):
            continue
        proc = Path("/proc") / str(pid)
        try:
            children = (proc / "task" / str(pid) / "children").read_text()
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError,
                ValueError):
            pass
        try:
            counters = {
                key.rstrip(":"): int(value)
                for line in (proc / "io").read_text().splitlines()
                for key, value in [line.split(maxsplit=1)]
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError,
                ValueError):
            continue
        signature.append((
            pid,
            counters.get("rchar", 0), counters.get("wchar", 0),
            counters.get("read_bytes", 0), counters.get("write_bytes", 0),
        ))
    return tuple(sorted(signature))


def launcher_progress_signature(
    launcher_pids: list[int] | None = None,
) -> tuple[Any, ...]:
    """Observe launcher I/O/cache output without walking the cache corpus."""
    paths: list[tuple[str, int, int]] = []
    for path in (FEATURE_CACHE, RUN / "overnight_trainer.log"):
        try:
            stat = path.stat()
        except OSError:
            continue
        paths.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(paths), _process_tree_io_signature(launcher_pids or [])


def exact_resume_is_missing(
    status: dict[str, Any] | None = None,
) -> bool:
    """Return true when restarting would discard evidence of an existing run."""
    return classify_run(
        RUN,
        status_override=status,
        allow_step_zero_preloop=launcher_contract(TRAIN_SCRIPT) is not None,
    ).state == "committed"


def start_training() -> int:
    env = os.environ.copy()
    evidence = classify_run(
        RUN, allow_step_zero_preloop=launcher_contract(TRAIN_SCRIPT) is not None)
    # Exact resumes validate entries as they are consumed and should not spend
    # hours walking the full cache after a crash. A first launch has no durable
    # run checkpoint yet, so leave cache verification enabled: the launcher
    # must fill every missing feature before step 1.
    if (RUN / "last.pt").is_file():
        env.setdefault("SKIP_CACHE_VERIFY", "1")
    pid = launch(
        ["bash", TRAIN_SCRIPT],
        RUN / "overnight_trainer.log",
        env=env,
    )
    event("training_restarted", pid=pid, checkpoint=str(RUN / "last.pt"),
          run_evidence=evidence.state)
    return pid


def start_hf_download() -> int:
    command = low_priority([
        sys.executable,
        "scripts/fetch_i1_sources.py",
        "--chunk-gib", "256",
        "--reserve-tib", "8",
        "--max-chunks", "0",
        "--wait-for-space",
    ])
    pid = launch(command, DATA / "overnight_hf_download.log")
    event("hf_download_restarted", pid=pid)
    return pid


def start_archive_download() -> int:
    command = low_priority([
        sys.executable,
        "scripts/fetch_i1_archives.py",
        "--sources", "yfcc_metadata", "inaturalist", "places365",
        "--chunk-gib", "4",
        "--reserve-tib", "8",
        "--max-chunks", "0",
        "--wait-for-space",
    ])
    pid = launch(command, DATA / "overnight_archive_download.log")
    event("archive_download_restarted", pid=pid)
    return pid


def held_out_image() -> Path | None:
    manifest = EVAL_MANIFEST
    try:
        with manifest.open() as handle:
            for line in handle:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                candidate = Path(str(raw.get("image", "")))
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if candidate.is_file():
                    return candidate.resolve()
    except OSError:
        pass
    return None


def valid_smoke(path: Path, *, checkpoint: Path | None = None,
                expected_step: int | None = None) -> bool:
    value = read_json(path)
    try:
        step = int(value.get("step", -1))
    except (TypeError, ValueError):
        return False
    if not str(value.get("caption", "")).strip() or step < 0:
        return False
    if expected_step is not None and step != expected_step:
        return False
    if checkpoint is not None:
        recorded = value.get("checkpoint")
        if not isinstance(recorded, str) or not recorded:
            return False
        try:
            if Path(recorded).resolve() != checkpoint.resolve():
                return False
        except OSError:
            return False
    return True


def process_age_seconds(pid: int) -> float | None:
    """Return process age from procfs monotonic boot time, if still identifiable."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        # ``comm`` is parenthesized and may itself contain spaces.
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])  # field 22; fields starts at field 3
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        ticks_per_second = os.sysconf("SC_CLK_TCK")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError,
            ValueError, IndexError):
        return None
    return max(0.0, uptime - start_ticks / ticks_per_second)


def signal_caption_process(pid: int, escalation: int) -> None:
    """Signal the caption session when ours, otherwise only the exact PID."""
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, escalation)
        else:
            os.kill(pid, escalation)
    except ProcessLookupError:
        pass


def best_checkpoint(run: Path | None = None) -> Path | None:
    """Resolve the immutable checkpoint selected by best.json, with legacy fallback."""
    # ``main`` can retarget the global RUN with --run. A default of ``RUN`` in
    # the function signature would capture the original phase-2 directory at
    # import time and make custom-run smoke tests use the wrong checkpoint.
    if run is None:
        run = RUN
    return resolve_best_checkpoint(run).checkpoint


def finish_or_start_smoke(
    training_done: bool,
    trainer_pids: list[int],
    launcher_pids: list[int] | None = None,
) -> str:
    final = RUN / "overnight_caption_smoke.json"
    temporary = RUN / "overnight_caption_smoke.json.tmp"
    launcher_pids = launcher_pids or []
    inference_pids = processes("rwkv_lab.vision_caption", str(RUN))
    if not training_done or trainer_pids or launcher_pids:
        # Raising a completed run's target can make training active again while
        # an old post-training smoke is still decoding on the same GPU. Never
        # allow that stale inference process to contend with the trainer.
        for pid in inference_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if inference_pids:
            event("caption_smoke_stopped_for_training", pids=inference_pids)
        return "waiting_for_training"

    best = resolve_best_checkpoint(RUN)
    if best.state == "invalid":
        return "blocked_invalid_best_checkpoint"
    checkpoint = best.checkpoint
    expected_step: int | None = None
    if checkpoint is not None:
        try:
            expected_step = int(read_json(RUN / "best" / "best.json")["step"])
        except (KeyError, TypeError, ValueError):
            pass
    else:
        checkpoint = RUN / "last.pt"
        status, status_step = training_state()
        try:
            expected_step = int(status.get("checkpoint_step", status_step))
        except (TypeError, ValueError):
            pass

    if not checkpoint.is_file():
        return "waiting_for_inputs"
    if valid_smoke(final, checkpoint=checkpoint, expected_step=expected_step):
        return "complete"
    if inference_pids:
        stale = [
            (pid, age) for pid in inference_pids
            if (age := process_age_seconds(pid)) is not None
            and age >= SMOKE_STALE_SECONDS
        ]
        if stale:
            actions = []
            for pid, age in stale:
                escalation = (
                    signal.SIGKILL
                    if age >= SMOKE_STALE_SECONDS + SMOKE_KILL_GRACE_SECONDS
                    else signal.SIGTERM
                )
                signal_caption_process(pid, escalation)
                actions.append({
                    "pid": pid, "signal": signal.Signals(escalation).name,
                })
            event("caption_smoke_stall_signal", actions=actions,
                  age_seconds=round(max(age for _, age in stale), 1))
            return "stopping_stalled"
        return "running"
    if temporary.exists():
        if valid_smoke(
                temporary, checkpoint=checkpoint, expected_step=expected_step):
            temporary.replace(final)
            result = read_json(final)
            event(
                "caption_smoke_complete",
                checkpoint=result.get("checkpoint"),
                step=result.get("step"),
                image=result.get("image"),
                caption=result.get("caption"),
            )
            return "complete"
        failed = RUN / f"overnight_caption_smoke.failed-{int(time.time())}.txt"
        temporary.replace(failed)
        event("caption_smoke_failed", output=str(failed))
    image = held_out_image()
    if image is None:
        return "waiting_for_inputs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with temporary.open("wb") as output, (RUN / "overnight_inference.log").open("ab") as error:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "rwkv_lab.vision_caption",
                "--checkpoint", str(checkpoint),
                "--image", str(image),
                "--max-new", "192",
                "--temperature", "0",
                "--json",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=error,
            start_new_session=True,
        )
    event("caption_smoke_started", pid=process.pid, checkpoint=str(checkpoint), image=str(image))
    return "running"


def acquisition_summary(path: Path) -> dict[str, Any]:
    state = read_json(path)
    sources: dict[str, Any] = {}
    raw_sources = state.get("sources", {})
    if not isinstance(raw_sources, dict):
        raw_sources = {}
    for name, value in raw_sources.items():
        if not isinstance(value, dict):
            continue
        try:
            expected = int(value.get("expected_bytes", 0) or 0)
            downloaded = int(value.get("downloaded_bytes", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            expected = downloaded = 0
        if value.get("status") == "complete" and expected:
            downloaded = expected
        sources[name] = {
            "status": value.get("status"),
            "downloaded_bytes": downloaded,
            "expected_bytes": expected,
            "percent": round(downloaded * 100 / expected, 3) if expected else None,
        }
    return {"updated_at": state.get("updated_at"), "sources": sources}


def acquisition_complete(path: Path, expected_sources: set[str]) -> bool:
    state = read_json(path)
    sources = state.get("sources", {})
    if not isinstance(sources, dict):
        return False
    return all(
        name in sources and isinstance(sources[name], dict)
        and sources[name].get("status") == "complete"
        for name in expected_sources
    )


def acquisition_waiting_for_space(path: Path) -> bool:
    state = read_json(path)
    sources = state.get("sources", {})
    if not isinstance(sources, dict):
        return False
    return any(isinstance(value, dict) and value.get("status") == "waiting_for_space"
               for value in sources.values())


def acquisition_progress_signature(
    path: Path,
    root: Path,
    patterns: tuple[str, ...] = ("*/.cache/huggingface/download/**/*.incomplete",),
) -> tuple[Any, ...]:
    """Track durable completions plus growth of the currently partial file."""
    state = read_json(path)
    sources = state.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}
    def downloaded_bytes(value: dict[str, Any]) -> int:
        try:
            return int(value.get("downloaded_bytes", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
    completed = tuple(sorted(
        (name, downloaded_bytes(value), value.get("status"))
        for name, value in sources.items()
        if isinstance(value, dict)
    ))
    partials: list[tuple[str, int, int]] = []
    for pattern in patterns:
        try:
            for item in root.glob(pattern):
                try:
                    stat = item.stat()
                except OSError:
                    continue
                partials.append((str(item.relative_to(root)), stat.st_size,
                                 stat.st_mtime_ns))
        except OSError:
            pass
    return completed, tuple(sorted(partials))


def escalate_stalled_processes(
    pids: list[int], *, stage: int, signaled_at: float | None,
    process_groups: bool = False,
) -> tuple[int, float | None, int | None]:
    """Apply one bounded escalation action and return updated state."""
    since_signal = (None if signaled_at is None
                    else time.monotonic() - signaled_at)
    escalation = stall_escalation_signal(stage, since_signal)
    if escalation is None:
        return stage, signaled_at, None
    for pid in pids:
        try:
            if process_groups:
                # Every process launched here starts a new session. Only signal
                # a group when the allowlisted PID is still its leader; never
                # widen a signal to an unrelated group after PID reuse.
                if os.getpgid(pid) != pid:
                    continue
                os.killpg(pid, escalation)
            else:
                os.kill(pid, escalation)
        except ProcessLookupError:
            pass
    return min(stage + 1, 3), time.monotonic(), escalation


def main() -> int:
    global RUN, TRAIN_SCRIPT, EVAL_MANIFEST, TARGET_STEPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--downloads-only",
        action="store_true",
        help=(
            "supervise resumable data acquisition without starting, stopping, "
            "or running inference with the trainer"
        ),
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="run directory to supervise; defaults to the current phase-2 run",
    )
    parser.add_argument(
        "--train-script",
        help="trainer launcher used for recovery when training supervision is enabled",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        help="held-out manifest used by the post-training caption smoke test",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        help="completion threshold; defaults to the selected launcher's environment contract",
    )
    args = parser.parse_args()
    if args.poll < 1:
        raise SystemExit("--poll must be positive")
    if args.train_script:
        candidate = Path(args.train_script)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise SystemExit(f"training launcher does not exist: {candidate}")
        TRAIN_SCRIPT = str(candidate)
    contract = launcher_contract(TRAIN_SCRIPT)
    if args.run is not None:
        RUN = args.run if args.run.is_absolute() else ROOT / args.run
        RUN = RUN.resolve()
        if (not args.downloads_only and contract is not None
                and RUN != contract[0].resolve()):
            raise SystemExit(
                f"{Path(TRAIN_SCRIPT).name} writes {contract[0]}, not {RUN}")
    elif contract is not None and args.train_script:
        RUN = contract[0].resolve()
    elif args.train_script and not args.downloads_only:
        raise SystemExit("--run is required for an unknown training launcher")
    if args.eval_manifest is not None:
        EVAL_MANIFEST = (
            args.eval_manifest if args.eval_manifest.is_absolute()
            else ROOT / args.eval_manifest
        ).resolve()
    elif contract is not None and args.train_script:
        EVAL_MANIFEST = contract[1].resolve()
    elif args.train_script and contract is None and not args.downloads_only:
        raise SystemExit("--eval-manifest is required for an unknown training launcher")
    try:
        TARGET_STEPS = resolve_target_steps(args.target_steps, TRAIN_SCRIPT)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with (RUN / "overnight_watchdog.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another overnight watchdog already holds the lock")

        event(
            "watchdog_started",
            pid=os.getpid(),
            poll_seconds=args.poll,
            downloads_only=args.downloads_only,
        )
        last_training_signature: tuple[int, Any, int] | None = None
        last_progress = time.monotonic()
        last_launcher_signature: tuple[Any, ...] | None = None
        stall_signaled_at: float | None = None
        stall_signal_stage = 0
        hf_signature: tuple[Any, ...] | None = None
        hf_last_progress = time.monotonic()
        hf_stall_signaled_at: float | None = None
        hf_signal_stage = 0
        archive_signature: tuple[Any, ...] | None = None
        archive_last_progress = time.monotonic()
        archive_stall_signaled_at: float | None = None
        archive_signal_stage = 0
        completion_seen_at: float | None = None
        training_restart_attempts = 0
        training_restart_step: int | None = None
        next_training_start_at = 0.0

        while True:
            status, step = training_state()
            try:
                train_log_mtime = (RUN / "train.jsonl").stat().st_mtime_ns
            except OSError:
                train_log_mtime = 0
            # status is intentionally sparse during ordinary training (every
            # ten steps), while JSONL is line-buffered every logged step. Count
            # either as progress so slow but healthy steps cannot be killed.
            training_signature = (step, status.get("updated"), train_log_mtime)
            if training_signature != last_training_signature:
                last_training_signature = training_signature
                last_progress = time.monotonic()
                stall_signaled_at = None
                stall_signal_stage = 0

            trainer_pids = processes("rwkv_lab.vision_train", RUN.name)
            launcher_pids = processes(TRAIN_SCRIPT)
            if launcher_pids:
                current_launcher_signature = launcher_progress_signature(launcher_pids)
                if (last_launcher_signature is None
                        or current_launcher_signature != last_launcher_signature):
                    last_launcher_signature = current_launcher_signature
                    last_progress = time.monotonic()
                    stall_signaled_at = None
                    stall_signal_stage = 0
            else:
                last_launcher_signature = None
            completion_receipt = training_is_done(status, step, TARGET_STEPS)
            run_evidence = classify_run(
                RUN,
                status_override=status,
                allow_step_zero_preloop=launcher_contract(TRAIN_SCRIPT) is not None,
            )
            resume_missing = run_evidence.state == "committed"
            training_done = completion_receipt and not resume_missing
            stale_for = time.monotonic() - last_progress
            if (trainer_pids and training_restart_step is not None
                    and step > training_restart_step):
                # The new trainer committed visible progress; a later failure
                # starts a fresh retry sequence instead of inheriting old debt.
                training_restart_attempts = 0
                training_restart_step = None
                next_training_start_at = 0.0
            if training_done:
                completion_seen_at = completion_seen_at or time.monotonic()
            else:
                completion_seen_at = None

            if not args.downloads_only:
                if (not training_done and not resume_missing
                        and not trainer_pids and not launcher_pids):
                    if time.monotonic() >= next_training_start_at:
                        start_training()
                        training_restart_attempts += 1
                        training_restart_step = step
                        # A first launch may spend substantial time in cache
                        # verification before the trainer exists or updates
                        # status.json. Give every replacement launcher a fresh
                        # stall grace period instead of inheriting the failed
                        # predecessor's stale timer/escalation stage.
                        last_progress = time.monotonic()
                        stale_for = 0.0
                        stall_signaled_at = None
                        stall_signal_stage = 0
                        delay = restart_backoff_seconds(
                            training_restart_attempts, args.poll)
                        next_training_start_at = time.monotonic() + delay
                        event("training_restart_backoff", attempt=training_restart_attempts,
                              retry_in_seconds=delay, step=step)
                elif (not training_done and not trainer_pids and launcher_pids
                      and stale_for >= TRAIN_STALE_SECONDS):
                    # Before vision_train starts, the launcher may be walking
                    # or filling the MoonViT cache. Supervise that phase too;
                    # otherwise one wedged cache process holds .launcher.lock
                    # forever while no trainer PID exists to trigger the
                    # ordinary stall path. The launcher's INT/TERM traps clean
                    # up its cache children before the retry backoff expires.
                    stall_signal_stage, stall_signaled_at, sent = \
                        escalate_stalled_processes(
                            launcher_pids, stage=stall_signal_stage,
                            signaled_at=stall_signaled_at,
                            process_groups=True)
                    if sent is not None:
                        event("training_launcher_stall_signal", pids=launcher_pids,
                              signal=signal.Signals(sent).name,
                              stale_seconds=round(stale_for, 1))
                elif not training_done and trainer_pids and stale_for >= TRAIN_STALE_SECONDS:
                    since_signal = (None if stall_signaled_at is None
                                    else time.monotonic() - stall_signaled_at)
                    escalation = stall_escalation_signal(
                        stall_signal_stage, since_signal)
                    if escalation == signal.SIGINT:
                        for pid in trainer_pids:
                            try:
                                os.kill(pid, signal.SIGINT)
                            except ProcessLookupError:
                                pass
                        stall_signaled_at = time.monotonic()
                        stall_signal_stage = 1
                        event("training_stall_sigint", pids=trainer_pids, step=step,
                              stale_seconds=round(stale_for, 1))
                    elif escalation == signal.SIGTERM:
                        for pid in trainer_pids:
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        event("training_stall_sigterm", pids=trainer_pids, step=step)
                        stall_signaled_at = time.monotonic()
                        stall_signal_stage = 2
                    elif escalation == signal.SIGKILL:
                        # SIGKILL is the final recovery action, only after four
                        # minutes of failed graceful escalation. Re-resolve the
                        # allowlisted argv matches to guard against PID reuse.
                        current_trainers = set(processes(
                            "rwkv_lab.vision_train", RUN.name))
                        current_launchers = set(processes(TRAIN_SCRIPT))
                        killed_trainers = sorted(set(trainer_pids) & current_trainers)
                        killed_launchers = sorted(set(launcher_pids) & current_launchers)
                        for pid in killed_trainers:
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        for pid in killed_launchers:
                            try:
                                if os.getpgid(pid) == pid:
                                    os.killpg(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        event("training_stall_sigkill", pids=killed_trainers,
                              launcher_pids=killed_launchers, step=step)
                        stall_signaled_at = time.monotonic()
                        stall_signal_stage = 3
                elif (training_done and trainer_pids and completion_seen_at is not None
                      and time.monotonic() - completion_seen_at >= 300):
                    # The final checkpoint and complete status are already atomic.
                    # Do not let Python's background cache threads hold the GPU and
                    # prevent the promised post-training caption smoke test.
                    for pid in trainer_pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    event("completed_trainer_sigterm", pids=trainer_pids, step=step)
                    completion_seen_at = time.monotonic()

            hf_pids = processes("scripts/fetch_i1_sources.py")
            current_hf_signature = acquisition_progress_signature(
                DATA / "acquisition_state.json", DATA)
            if hf_signature is None or current_hf_signature != hf_signature:
                hf_signature = current_hf_signature
                hf_last_progress = time.monotonic()
                hf_stall_signaled_at = None
                hf_signal_stage = 0
            hf_stale_for = time.monotonic() - hf_last_progress
            if not hf_pids and not acquisition_complete(
                    DATA / "acquisition_state.json", HF_SOURCES):
                start_hf_download()
                # The old child may have reached TERM/KILL. A replacement must
                # not inherit that stale age or escalation stage and be killed
                # before it has had a chance to open and grow its next shard.
                hf_last_progress, hf_stall_signaled_at, hf_signal_stage = \
                    fresh_stall_supervision()
                hf_stale_for = 0.0
            elif (hf_pids and hf_stale_for >= DOWNLOAD_STALE_SECONDS
                  and not acquisition_waiting_for_space(DATA / "acquisition_state.json")):
                hf_signal_stage, hf_stall_signaled_at, sent = \
                    escalate_stalled_processes(
                        hf_pids, stage=hf_signal_stage,
                        signaled_at=hf_stall_signaled_at,
                        process_groups=True)
                if sent is not None:
                    event("hf_download_stall_signal", pids=hf_pids,
                          signal=signal.Signals(sent).name,
                          stale_seconds=round(hf_stale_for, 1))
            archive_pids = processes("scripts/fetch_i1_archives.py")
            current_archive_signature = acquisition_progress_signature(
                DATA / "archive_acquisition_state.json", DATA,
                # Known archive destinations only; a recursive glob across the
                # multi-hundred-GB corpus would itself become watchdog load.
                patterns=("inaturalist/*.part", "places365/*.part", "yfcc/*.part"))
            if (archive_signature is None
                    or current_archive_signature != archive_signature):
                archive_signature = current_archive_signature
                archive_last_progress = time.monotonic()
                archive_stall_signaled_at = None
                archive_signal_stage = 0
            archive_stale_for = time.monotonic() - archive_last_progress
            if not archive_pids and not acquisition_complete(
                    DATA / "archive_acquisition_state.json", ARCHIVE_SOURCES):
                start_archive_download()
                archive_last_progress, archive_stall_signaled_at, \
                    archive_signal_stage = fresh_stall_supervision()
                archive_stale_for = 0.0
            elif (archive_pids and archive_stale_for >= DOWNLOAD_STALE_SECONDS
                  and not acquisition_waiting_for_space(
                      DATA / "archive_acquisition_state.json")):
                archive_signal_stage, archive_stall_signaled_at, sent = \
                    escalate_stalled_processes(
                        archive_pids, stage=archive_signal_stage,
                        signaled_at=archive_stall_signaled_at,
                        process_groups=True)
                if sent is not None:
                    event("archive_download_stall_signal", pids=archive_pids,
                          signal=signal.Signals(sent).name,
                          stale_seconds=round(archive_stale_for, 1))

            # Refresh process lists after any launches and expose one compact,
            # atomically replaced heartbeat for status checks and the dashboard.
            trainer_pids = processes("rwkv_lab.vision_train", RUN.name)
            launcher_pids = processes(TRAIN_SCRIPT)
            hf_pids = processes("scripts/fetch_i1_sources.py")
            archive_pids = processes("scripts/fetch_i1_archives.py")
            smoke = (
                "disabled_while_training_paused"
                if args.downloads_only
                else finish_or_start_smoke(
                    training_done, trainer_pids, launcher_pids)
            )
            heartbeat = {
                "updated_at": now(),
                "watchdog_pid": os.getpid(),
                "training": {
                    "supervision": (
                        "paused" if args.downloads_only
                        else "blocked_missing_exact_checkpoint" if resume_missing
                        else "enabled"
                    ),
                    "launcher": TRAIN_SCRIPT,
                    "state": status.get("state"),
                    "step": step,
                    "target_steps": TARGET_STEPS,
                    "trainer_pids": trainer_pids,
                    "launcher_pids": launcher_pids,
                    "restart_attempts": training_restart_attempts,
                    "next_restart_in_seconds": round(max(
                        0.0, next_training_start_at - time.monotonic()), 1),
                    "seconds_since_progress": round(stale_for, 1),
                    "last_checkpoint_exists": run_evidence.state == "exact",
                    "resume_evidence": run_evidence.state,
                    "resume_evidence_reasons": run_evidence.reasons,
                    "best_checkpoint_exists": best_checkpoint() is not None,
                    "best_checkpoint_state": resolve_best_checkpoint(RUN).state,
                },
                "caption_smoke": smoke,
                "downloads": {
                    "hf_pids": hf_pids,
                    "hf_seconds_since_progress": round(hf_stale_for, 1),
                    "archive_pids": archive_pids,
                    "archive_seconds_since_progress": round(archive_stale_for, 1),
                    "hf": acquisition_summary(DATA / "acquisition_state.json"),
                    "archives": acquisition_summary(DATA / "archive_acquisition_state.json"),
                },
                "disk_free_bytes": shutil.disk_usage(ROOT).free,
            }
            atomic_json(RUN / "overnight_status.json", heartbeat)
            if args.once:
                return 0
            time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
