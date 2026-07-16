#!/usr/bin/env python3
"""Classify whether a vision run can safely take a fresh first launch."""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


EvidenceState = Literal["pristine", "startup", "committed", "exact"]
BestState = Literal["valid", "absent", "invalid"]


@dataclass(frozen=True)
class RunEvidence:
    state: EvidenceState
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BestCheckpoint:
    state: BestState
    checkpoint: Path | None = None
    reason: str | None = None


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _step_state(value: Any) -> Literal["zero", "positive", "invalid"]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "invalid"
    return "positive" if value > 0 else "zero"


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _status_evidence(value: dict[str, Any]) -> RunEvidence:
    has_step = "step" in value
    step = _step_state(value["step"]) if has_step else "zero"
    if step == "positive":
        return RunEvidence("committed", ("status_step_positive",))
    if step == "invalid":
        return RunEvidence("committed", ("status_step_invalid",))
    state = value.get("state")
    if not isinstance(state, str) or not state:
        return RunEvidence("committed", ("status_state_missing",))
    if state in {"complete", "paused", "stopped"}:
        return RunEvidence("committed", ("terminal_status_without_checkpoint",))
    if bool(value.get("exact_checkpoint_saved")):
        return RunEvidence("committed", ("checkpoint_receipt_without_checkpoint",))

    loading_states = {"loading_data", "loading_rwkv", "loading_moonvit"}
    if state in loading_states:
        return RunEvidence("startup", ("startup_status",))
    if state == "preloading_features" and has_step:
        return RunEvidence("startup", ("startup_status",))
    if state == "training" and has_step:
        return RunEvidence("startup", ("startup_status",))
    if state == "failed":
        previous = value.get("previous_state")
        if previous in loading_states:
            return RunEvidence("startup", ("startup_failure",))
        if has_step and previous in {"preloading_features", "training"}:
            return RunEvidence("startup", ("startup_failure",))
    return RunEvidence("committed", ("ambiguous_status_without_checkpoint",))


def _json_step_receipt(path: Path, label: str) -> RunEvidence:
    value = _read_object(path)
    if value is None:
        return RunEvidence("committed", (f"malformed_{label}",))
    step = _step_state(value.get("step"))
    if step == "positive":
        return RunEvidence("committed", (f"{label}_step_positive",))
    if step == "invalid":
        return RunEvidence("committed", (f"{label}_step_invalid",))
    return RunEvidence("startup", (f"{label}_step_zero",))


def _train_log_evidence(path: Path) -> RunEvidence:
    reasons: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError):
        return RunEvidence("committed", ("unreadable_train_log",))
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return RunEvidence("committed", ("malformed_train_log",))
        if not isinstance(record, dict):
            return RunEvidence("committed", ("invalid_train_record",))
        step = _step_state(record.get("step"))
        if step == "positive":
            return RunEvidence("committed", ("train_log_step_positive",))
        if step == "invalid":
            return RunEvidence("committed", ("train_log_step_invalid",))
        reasons.append("train_log_step_zero")
    return RunEvidence("startup", tuple(dict.fromkeys(reasons)) or ("empty_train_log",))


def _eval_contract_evidence(path: Path) -> RunEvidence:
    value = _read_object(path)
    if value is None:
        return RunEvidence("committed", ("malformed_eval_contract_reset",))
    fresh_step_zero = (
        set(value) == {"schema", "reset", "step", "reasons"}
        and isinstance(value["schema"], int)
        and not isinstance(value["schema"], bool)
        and value["schema"] == 1
        and value.get("reset") is True
        and _step_state(value.get("step")) == "zero"
        and value.get("reasons") == ["fresh"]
    )
    if fresh_step_zero:
        return RunEvidence("startup", ("fresh_eval_contract",))
    return RunEvidence("committed", ("mutated_eval_contract_without_last",))


def classify_run(
    run: Path,
    *,
    status_override: dict[str, Any] | None = None,
    allow_step_zero_preloop: bool = False,
) -> RunEvidence:
    """Classify exact state, committed work, or safely retryable startup debris."""
    run = Path(run)
    last = run / "last.pt"
    if last.is_file() and not last.is_symlink():
        return RunEvidence("exact", ("last_checkpoint",))
    if _lexists(last):
        return RunEvidence("committed", ("invalid_last_checkpoint",))

    # These are only published while checkpointing/evaluating committed model
    # state. Their presence is ambiguous without last.pt, so recovery must stop.
    for name in (
        "last.tmp", "best", "eval_samples",
    ):
        if _lexists(run / name):
            return RunEvidence("committed", (f"{name}_without_last",))
    if _lexists(run / "pre_loop.pt") and not allow_step_zero_preloop:
        return RunEvidence("committed", ("unverified_pre_loop_without_last",))

    status_path = run / "status.json"
    startup_reasons: list[str] = []
    if _lexists(status_path):
        status = _read_object(status_path)
        if status is None:
            return RunEvidence("committed", ("malformed_status",))
        evidence = _status_evidence(status)
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(evidence.reasons)
    status_temporary = run / "status.json.tmp"
    if _lexists(status_temporary):
        status = _read_object(status_temporary)
        if status is None:
            return RunEvidence("committed", ("malformed_status_temporary",))
        evidence = _status_evidence(status)
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(("startup_status_temporary",))
    if status_override:
        evidence = _status_evidence(status_override)
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(evidence.reasons)

    for reset_receipt in (
        run / "eval_contract_reset.json",
        run / "eval_contract_reset.json.tmp",
    ):
        if not _lexists(reset_receipt):
            continue
        evidence = _eval_contract_evidence(reset_receipt)
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(evidence.reasons)

    train_log = run / "train.jsonl"
    if _lexists(train_log):
        evidence = _train_log_evidence(train_log)
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(evidence.reasons)

    for loop_receipt in (run / "loop_rw.json", run / "loop_rw.tmp"):
        if not _lexists(loop_receipt):
            continue
        evidence = _json_step_receipt(loop_receipt, "loop_receipt")
        if evidence.state == "committed":
            return evidence
        startup_reasons.extend(evidence.reasons)

    # These files are created before the first optimizer update and are not, by
    # themselves, evidence that learned state would be discarded.
    for name in ("config.json", "config.json.tmp", ".trainer.lock"):
        if _lexists(run / name):
            startup_reasons.append(f"startup_artifact:{name}")
    if allow_step_zero_preloop and _lexists(run / "pre_loop.pt"):
        startup_reasons.append("known_step_zero_pre_loop")

    if startup_reasons:
        return RunEvidence("startup", tuple(dict.fromkeys(startup_reasons)))
    return RunEvidence("pristine")


def _valid_best_metadata(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    step = value.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return False
    found_metric = False
    for name in ("loss", "ppl"):
        if name not in value:
            continue
        found_metric = True
        metric = value[name]
        if (isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))):
            return False
        if name == "loss" and float(metric) < 0:
            return False
        if name == "ppl" and float(metric) <= 0:
            return False
    return found_metric


def _contained_regular_file(parent: Path, candidate: Path) -> bool:
    try:
        if candidate.is_symlink() or not stat.S_ISREG(candidate.stat().st_mode):
            return False
        return candidate.resolve(strict=True).parent == parent.resolve(strict=True)
    except OSError:
        return False


def resolve_best_checkpoint(run: Path) -> BestCheckpoint:
    """Resolve an atomic best publication without masking invalid metadata."""
    best = Path(run) / "best"
    manifest = best / "best.json"
    manifest_temporary = best / "best.json.tmp"
    legacy = best / "ckpt.pt"
    if _lexists(best) and (best.is_symlink() or not best.is_dir()):
        return BestCheckpoint("invalid", reason="best_directory_not_regular")
    if not _lexists(manifest) and _lexists(manifest_temporary):
        return BestCheckpoint("invalid", reason="incomplete_best_manifest_publication")
    if not _lexists(manifest):
        if _contained_regular_file(best, legacy):
            return BestCheckpoint("valid", legacy, "unmanifested_legacy")
        if _lexists(best):
            return BestCheckpoint("invalid", reason="incomplete_best_directory")
        return BestCheckpoint("absent", reason="no_best_manifest")
    if manifest.is_symlink() or not manifest.is_file():
        return BestCheckpoint("invalid", reason="best_manifest_not_regular")
    payload = _read_object(manifest)
    if not _valid_best_metadata(payload):
        return BestCheckpoint("invalid", reason="invalid_best_metadata")
    assert payload is not None
    if "checkpoint" in payload:
        name = payload["checkpoint"]
        if (not isinstance(name, str) or not name
                or Path(name).name != name or Path(name).suffix != ".pt"):
            return BestCheckpoint("invalid", reason="invalid_best_target")
        checkpoint = best / name
    else:
        checkpoint = legacy
    if not _contained_regular_file(best, checkpoint):
        return BestCheckpoint("invalid", reason="best_target_missing_or_escaped")
    return BestCheckpoint("valid", checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-step-zero-preloop", action="store_true")
    parser.add_argument("--resolve-best", action="store_true")
    args = parser.parse_args()
    if args.resolve_best:
        best = resolve_best_checkpoint(args.run)
        if args.json:
            print(json.dumps({
                "state": best.state,
                "checkpoint": str(best.checkpoint) if best.checkpoint else None,
                "reason": best.reason,
            }))
        elif best.checkpoint is not None:
            print(best.checkpoint)
        return {"valid": 0, "absent": 2, "invalid": 3}[best.state]
    evidence = classify_run(
        args.run, allow_step_zero_preloop=args.allow_step_zero_preloop)
    if args.json:
        print(json.dumps({"state": evidence.state, "reasons": evidence.reasons}))
    else:
        print(evidence.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
