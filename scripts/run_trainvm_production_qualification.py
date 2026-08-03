#!/usr/bin/env python3
"""Run the three real trainer-family qualification experiments through TrainVM.

The runner submits existing declarative experiment documents through the
loopback dashboard, drives a checkpoint-first resource-releasing pause and
resume in each family, waits for terminal success, then invokes the immutable
live evidence capture.  It never constructs a trainer command line itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILIES = ("mageflow", "rwkv", "transformer")
MAXIMUM_DOCUMENT_BYTES = 2 * 1024 * 1024
TERMINAL_FAILURES = frozenset({"cancelled", "failed", "blocked"})


class QualificationRunError(RuntimeError):
    """The live qualification could not safely reach its required state."""


def _load_capture_module() -> Any:
    path = Path(__file__).with_name("capture_trainvm_production_evidence.py")
    spec = importlib.util.spec_from_file_location("trainvm_production_capture", path)
    if spec is None or spec.loader is None:
        raise QualificationRunError("cannot load production evidence capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_module()


def _experiment_argument(value: str) -> tuple[str, Path]:
    family, separator, raw_path = value.partition("=")
    if not separator or family not in FAMILIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "experiment must be mageflow|rwkv|transformer=DOCUMENT.json"
        )
    return family, Path(raw_path)


def _step_argument(value: str) -> tuple[str, int]:
    family, separator, raw_step = value.partition("=")
    if not separator or family not in FAMILIES:
        raise argparse.ArgumentTypeError(
            "pause step must be mageflow|rwkv|transformer=POSITIVE_INTEGER"
        )
    try:
        step = int(raw_step)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pause step must be an integer") from error
    if step < 1 or step > 1_000_000_000:
        raise argparse.ArgumentTypeError("pause step is outside the supported bound")
    return family, step


def _eval_metric(name: Any) -> bool:
    return isinstance(name, str) and (
        name.startswith(("eval.", "eval/", "val."))
        or "validation" in name.lower()
    )


def _family_adapter(family: str, adapter: str) -> bool:
    if family == "mageflow":
        return adapter.startswith("rwkv-lab.mageflow-")
    if family == "rwkv":
        return adapter.startswith("rwkv-lab.rwkv-")
    return adapter.startswith("rwkv-lab.transformer-") or adapter == "rwkv-lab.qwen-ao3"


def load_qualification_document(family: str, path: Path) -> tuple[dict[str, Any], str]:
    try:
        if path.is_symlink():
            raise QualificationRunError(f"experiment document is a symlink: {path}")
        path = path.resolve(strict=True)
        raw = path.read_bytes()
    except OSError as error:
        raise QualificationRunError(f"cannot read {family} experiment: {error}") from error
    if len(raw) > MAXIMUM_DOCUMENT_BYTES:
        raise QualificationRunError(f"{family} experiment exceeds the document bound")
    try:
        document = CAPTURE._parse_json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationRunError(f"{family} experiment is not valid JSON: {error}") from error
    if not isinstance(document, dict) or document.get("kind") != "Experiment":
        raise QualificationRunError(f"{family} qualification is not an Experiment")
    spec = document.get("spec")
    components = spec.get("components") if isinstance(spec, dict) else None
    workflow = spec.get("workflow") if isinstance(spec, dict) else None
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(components, dict) or not isinstance(nodes, dict):
        raise QualificationRunError(f"{family} qualification has no declarative workflow")
    family_components = {
        name
        for name, component in components.items()
        if isinstance(component, dict)
        and isinstance(component.get("adapter"), str)
        and _family_adapter(family, component["adapter"])
        and component.get("runtime") == "python_worker"
    }
    if not family_components:
        raise QualificationRunError(
            f"{family} qualification invokes no registered trainer-family adapter"
        )
    training_nodes = []
    checkpoint_outputs: set[str] = set()
    gallery_outputs: set[str] = set()
    for name, node in nodes.items():
        if not isinstance(node, dict):
            continue
        invoke = node.get("invoke")
        if not isinstance(invoke, dict) or invoke.get("component") not in family_components:
            continue
        training = invoke.get("training")
        if node.get("effect") != "process" or not isinstance(training, dict):
            continue
        if training.get("model_family") != family:
            raise QualificationRunError(
                f"{family} training node {name} declares the wrong model family"
            )
        training_nodes.append(name)
        publishes = node.get("publishes")
        if isinstance(publishes, dict):
            if isinstance(publishes.get("checkpoint"), str):
                checkpoint_outputs.add(publishes["checkpoint"])
            if isinstance(publishes.get("eval_gallery"), str):
                gallery_outputs.add(publishes["eval_gallery"])
    if not training_nodes or not checkpoint_outputs:
        raise QualificationRunError(
            f"{family} qualification has no checkpoint-publishing training node"
        )

    execution = spec.get("execution")
    gpu_trace = execution.get("gpu_trace") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or execution.get("component") not in family_components
        or not isinstance(gpu_trace, dict)
        or gpu_trace.get("enabled") is not True
        or gpu_trace.get("backend") != "torch"
        or isinstance(gpu_trace.get("capture_steps"), bool)
        or not isinstance(gpu_trace.get("capture_steps"), int)
        or gpu_trace["capture_steps"] < 1
        or "accelerator" not in gpu_trace.get("activities", [])
    ):
        raise QualificationRunError(
            f"{family} qualification has no bounded in-process GPU trace"
        )
    artifacts = spec.get("artifacts")
    output_artifact = gpu_trace.get("output_artifact")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(output_artifact, str)
        or not isinstance(artifacts.get(output_artifact), dict)
        or artifacts[output_artifact].get("schema") != "trainvm.gpu-trace.v1"
        or not checkpoint_outputs.issubset(artifacts)
    ):
        raise QualificationRunError(f"{family} qualification artifact contract is incomplete")
    observability = spec.get("observability")
    metrics = observability.get("metrics") if isinstance(observability, dict) else None
    if not isinstance(metrics, list) or not any(
        isinstance(metric, dict) and _eval_metric(metric.get("name"))
        for metric in metrics
    ):
        raise QualificationRunError(f"{family} qualification declares no eval metric")
    if family == "mageflow":
        gallery_name = observability.get("eval_gallery_artifact")
        if (
            not gallery_outputs
            or gallery_name not in gallery_outputs
            or not isinstance(artifacts.get(gallery_name), dict)
            or artifacts[gallery_name].get("schema") != "rwkv-lab.eval-gallery.v2"
        ):
            raise QualificationRunError(
                "MageFlow qualification does not publish its side-by-side eval gallery"
            )
    recovery = spec.get("recovery")
    if (
        not isinstance(recovery, dict)
        or recovery.get("checkpoint_artifact") not in checkpoint_outputs
        or recovery.get("reconcile") != "restart_from_checkpoint"
    ):
        raise QualificationRunError(
            f"{family} qualification does not select checkpoint recovery"
        )
    return document, json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class SubmittedRun:
    family: str
    run_id: str
    plan_hash: str


class AuthorityClient(CAPTURE.Dashboard):
    def post(self, path: str, value: Any, expected: set[int]) -> tuple[int, Any]:
        encoded = json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        request = urllib.request.Request(
            self.origin + path,
            data=encoded,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, urllib.error.URLError) as error:
            raise QualificationRunError(f"POST {path} failed: {error}") from error
        with response:
            raw = response.read(CAPTURE.MAXIMUM_RESPONSE_BYTES + 1)
            status = response.status
            final_url = response.geturl()
        if final_url != self.origin + path or status not in expected:
            detail = raw[:2048].decode("utf-8", "replace")
            raise QualificationRunError(f"POST {path} returned HTTP {status}: {detail}")
        if len(raw) > CAPTURE.MAXIMUM_RESPONSE_BYTES:
            raise QualificationRunError(f"POST {path} exceeded the response bound")
        try:
            return status, CAPTURE._parse_json(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QualificationRunError(f"POST {path} returned invalid JSON") from error


def _run_view(client: AuthorityClient, run_id: str) -> dict[str, Any]:
    value = client.get("/api/trainvm/runs/" + urllib.parse.quote(run_id, safe=""))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise QualificationRunError(f"authority returned a malformed run view for {run_id}")
    return value


def _wait(
    client: AuthorityClient,
    run: SubmittedRun,
    predicate: Any,
    description: str,
    deadline: float,
    poll_seconds: float,
) -> dict[str, Any]:
    while True:
        view = _run_view(client, run.run_id)
        observed = view.get("observed_state")
        if predicate(view):
            return view
        if observed in TERMINAL_FAILURES:
            raise QualificationRunError(
                f"{run.family} entered {observed}: {view.get('failure_summary', '')}"
            )
        if observed == "completed":
            raise QualificationRunError(
                f"{run.family} completed before it could {description}"
            )
        if time.monotonic() >= deadline:
            raise QualificationRunError(
                f"timed out waiting for {run.family} to {description}"
            )
        time.sleep(poll_seconds)


def _submit(
    client: AuthorityClient, family: str, source: str, journal_id: str
) -> SubmittedRun:
    _, preview = client.post("/api/trainvm/compile", CAPTURE._parse_json(source), {200})
    if not isinstance(preview, dict) or preview.get("valid") is not True:
        raise QualificationRunError(f"native compiler rejected {family}: {preview}")
    plan_hash = preview.get("plan_hash")
    adapter_lock = preview.get("adapter_lock_digest")
    component_lock = preview.get("training_component_lock_digest", "")
    if (
        not isinstance(plan_hash, str)
        or not plan_hash
        or not isinstance(adapter_lock, str)
        or not adapter_lock
        or not isinstance(component_lock, str)
        or not component_lock
    ):
        raise QualificationRunError(f"native compiler omitted {family} lock identity")
    _, created = client.post(
        "/api/trainvm/experiments",
        {
            "source_document": source,
            "source_format": "json",
            "create_run": True,
            "idempotency_key": f"production-qualification-{family}-{plan_hash[:24]}",
            "expected_journal_id": journal_id,
            "expected_plan_hash": plan_hash,
            "expected_adapter_lock_digest": adapter_lock,
            "expected_training_component_lock_digest": component_lock,
            "reason": f"production parity qualification for {family}",
        },
        {200, 202},
    )
    identity = created.get("run") if isinstance(created, dict) else None
    run_id = identity.get("run_id") if isinstance(identity, dict) else None
    if not isinstance(run_id, str) or not run_id or identity.get("plan_hash") != plan_hash:
        raise QualificationRunError(f"authority returned no bound {family} run identity")
    return SubmittedRun(family, run_id, plan_hash)


def _action(
    client: AuthorityClient,
    run: SubmittedRun,
    view: dict[str, Any],
    action: str,
    ordinal: int,
) -> None:
    revision = view.get("run_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise QualificationRunError(f"{run.family} has no lifecycle revision")
    body = {
        "expected_run_revision": revision,
        "idempotency_key": f"production-qualification-{run.family}-{action}-{ordinal}",
        "reason": f"production qualification {action} cycle",
        "action": action,
        "checkpoint_first": action == "pause",
        "release_resources": action == "pause",
        "graceful_timeout_seconds": 0,
    }
    _, result = client.post(
        f"/api/trainvm/runs/{urllib.parse.quote(run.run_id, safe='')}/actions",
        body,
        {200, 202},
    )
    if not isinstance(result, dict) or result.get("disposition") not in {
        "ACCEPTED",
        "ALREADY_APPLIED",
    }:
        raise QualificationRunError(f"{run.family} {action} was not accepted")


def _prove_resume_progress(
    client: AuthorityClient,
    run: SubmittedRun,
    paused: dict[str, Any],
    completed: dict[str, Any],
) -> None:
    paused_step = paused.get("optimizer_step")
    final_step = completed.get("optimizer_step")
    through = completed.get("last_event_sequence")
    if (
        isinstance(paused_step, bool)
        or not isinstance(paused_step, int)
        or paused_step < 1
        or isinstance(final_step, bool)
        or not isinstance(final_step, int)
        or final_step <= paused_step
        or isinstance(through, bool)
        or not isinstance(through, int)
        or through < 1
    ):
        raise QualificationRunError(
            f"{run.family} made no authoritative optimizer progress after resume"
        )
    base = "/api/trainvm/runs/" + urllib.parse.quote(run.run_id, safe="")
    timeline = client.paged(base + "/timeline", through)
    resumed_checkpoint_ids = {
        event.get("payload", {}).get("checkpoint_artifact_id")
        for event in timeline
        if isinstance(event, dict)
        and event.get("run_id") == run.run_id
        and event.get("event_type") == "node.attempt_restarted"
        and event.get("optimizer_step") == paused_step
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("checkpoint_artifact_id"), str)
        and event["payload"]["checkpoint_artifact_id"]
    }
    if not resumed_checkpoint_ids:
        raise QualificationRunError(
            f"{run.family} has no checkpoint-bound restart at its paused step"
        )
    metrics = client.paged(base + "/metrics", through)
    if not any(
        isinstance(metric, dict)
        and metric.get("run_id") == run.run_id
        and isinstance(metric.get("name"), str)
        and metric["name"]
        and isinstance(metric.get("optimizer_step"), int)
        and not isinstance(metric.get("optimizer_step"), bool)
        and metric["optimizer_step"] > paused_step
        and isinstance(metric.get("value"), (int, float))
        and not isinstance(metric.get("value"), bool)
        and math.isfinite(float(metric["value"]))
        for metric in metrics
    ):
        raise QualificationRunError(
            f"{run.family} emitted no finite live metric after resume"
        )


def _latest_checkpoint_restart_step(
    client: AuthorityClient,
    run: SubmittedRun,
    view: dict[str, Any],
) -> int | None:
    through = view.get("last_event_sequence")
    if isinstance(through, bool) or not isinstance(through, int) or through < 1:
        return None
    base = "/api/trainvm/runs/" + urllib.parse.quote(run.run_id, safe="")
    timeline = client.paged(base + "/timeline", through)
    steps = [
        event["optimizer_step"]
        for event in timeline
        if isinstance(event, dict)
        and event.get("run_id") == run.run_id
        and event.get("event_type") == "node.attempt_restarted"
        and isinstance(event.get("optimizer_step"), int)
        and not isinstance(event.get("optimizer_step"), bool)
        and event["optimizer_step"] > 0
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("checkpoint_artifact_id"), str)
        and event["payload"]["checkpoint_artifact_id"]
    ]
    return max(steps) if steps else None


def _released_host_authority(client: AuthorityClient, family: str) -> None:
    authority = client.get("/api/trainvm/host-authority")
    if (
        not isinstance(authority, dict)
        or authority.get("active_process_count") != 0
        or authority.get("active_fence_count") != 0
    ):
        raise QualificationRunError(
            f"{family} resource-releasing pause left host authority active"
        )


def _drive_family(
    client: AuthorityClient,
    run: SubmittedRun,
    pause_step: int,
    deadline: float,
    poll_seconds: float,
) -> None:
    """Reconcile one idempotently submitted run through pause/resume/completion."""
    view = _run_view(client, run.run_id)
    while True:
        observed = view.get("observed_state")
        if observed in TERMINAL_FAILURES:
            raise QualificationRunError(
                f"{run.family} entered {observed}: {view.get('failure_summary', '')}"
            )
        restart_step = _latest_checkpoint_restart_step(client, run, view)
        if observed == "completed":
            if restart_step is None:
                raise QualificationRunError(
                    f"{run.family} completed without its qualification resume cycle"
                )
            _prove_resume_progress(
                client,
                run,
                {"optimizer_step": restart_step},
                view,
            )
            return
        if restart_step is not None:
            completed = _wait(
                client,
                run,
                lambda candidate: candidate.get("observed_state") == "completed",
                "complete after checkpoint resume",
                deadline,
                poll_seconds,
            )
            _prove_resume_progress(
                client,
                run,
                {"optimizer_step": restart_step},
                completed,
            )
            return
        optimizer_step = view.get("optimizer_step")
        if observed == "paused":
            if (
                isinstance(optimizer_step, bool)
                or not isinstance(optimizer_step, int)
                or optimizer_step < pause_step
            ):
                raise QualificationRunError(
                    f"{run.family} paused before qualification step {pause_step}"
                )
            _released_host_authority(client, run.family)
            _action(client, run, view, "resume", 1)
        elif (
            observed == "running"
            and isinstance(optimizer_step, int)
            and not isinstance(optimizer_step, bool)
            and optimizer_step >= pause_step
        ):
            _action(client, run, view, "pause", 1)
        view = _wait(
            client,
            run,
            lambda candidate: candidate.get("observed_state")
            in {"paused", "completed"}
            or (
                candidate.get("observed_state") == "running"
                and isinstance(candidate.get("optimizer_step"), int)
                and not isinstance(candidate.get("optimizer_step"), bool)
                and candidate["optimizer_step"] >= pause_step
            ),
            f"reach pause step {pause_step} or reconcile lifecycle progress",
            deadline,
            poll_seconds,
        )


def qualify(
    origin: str,
    documents: dict[str, str],
    pause_steps: dict[str, int],
    output: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> Path:
    client = AuthorityClient(origin)
    client.health()
    listing = client.get("/api/trainvm/runs")
    journal_id = listing.get("journal_id") if isinstance(listing, dict) else None
    if (
        not isinstance(journal_id, str)
        or not journal_id
        or listing.get("commands_enabled") is not True
    ):
        raise QualificationRunError("dashboard is not bound to TrainVM command authority")
    runs: dict[str, str] = {}
    for family in FAMILIES:
        run = _submit(client, family, documents[family], journal_id)
        deadline = time.monotonic() + timeout_seconds
        _drive_family(
            client,
            run,
            pause_steps[family],
            deadline,
            poll_seconds,
        )
        runs[family] = run.run_id
    return CAPTURE.live_capture(origin, runs, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-url", type=CAPTURE._local_dashboard_url, required=True)
    parser.add_argument("--experiment", action="append", type=_experiment_argument, required=True)
    parser.add_argument("--pause-after-step", action="append", type=_step_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    arguments = parser.parse_args()
    experiments = dict(arguments.experiment)
    pause_steps = dict(arguments.pause_after_step)
    if (
        len(experiments) != len(arguments.experiment)
        or set(experiments) != set(FAMILIES)
        or len(pause_steps) != len(arguments.pause_after_step)
        or set(pause_steps) != set(FAMILIES)
        or not math.isfinite(arguments.timeout_seconds)
        or arguments.timeout_seconds <= 0
        or not math.isfinite(arguments.poll_seconds)
        or not 0.05 <= arguments.poll_seconds <= 60
    ):
        parser.error("provide one experiment and pause step per family plus finite bounds")
    try:
        sources = {
            family: load_qualification_document(family, experiments[family])[1]
            for family in FAMILIES
        }
        result = qualify(
            arguments.dashboard_url,
            sources,
            pause_steps,
            arguments.output,
            arguments.timeout_seconds,
            arguments.poll_seconds,
        )
    except (QualificationRunError, CAPTURE.CaptureError, ValueError) as error:
        print(f"production qualification failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
