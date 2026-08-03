from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_trainvm_production_qualification.py"
)
SPEC = importlib.util.spec_from_file_location("trainvm_production_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _document(family: str) -> dict[str, Any]:
    adapter = {
        "mageflow": "rwkv-lab.mageflow-terminal-expert",
        "rwkv": "rwkv-lab.rwkv-optimizer-finetune",
        "transformer": "rwkv-lab.transformer-mla",
    }[family]
    artifacts: dict[str, Any] = {
        "checkpoint": {
            "type": "checkpoint",
            "schema": f"qualification.{family}-checkpoint.v1",
        },
        "gpu_trace": {
            "type": "opaque",
            "schema": "trainvm.gpu-trace.v1",
            "immutability": "mutable_until_publish",
            "fingerprint": "adapter",
        },
    }
    publishes = {"checkpoint": "checkpoint"}
    observability: dict[str, Any] = {
        "metrics": [{"name": "eval.loss"}],
    }
    if family == "mageflow":
        artifacts["eval_gallery"] = {
            "type": "image_gallery",
            "schema": "rwkv-lab.eval-gallery.v2",
        }
        publishes["eval_gallery"] = "eval_gallery"
        observability["eval_gallery_artifact"] = "eval_gallery"
    return {
        "api_version": "trainvm.rwkv-lab/v1alpha1",
        "kind": "Experiment",
        "metadata": {"name": f"production-{family}-qualification"},
        "spec": {
            "components": {
                "trainer": {
                    "adapter": adapter,
                    "version": "1.0.0",
                    "runtime": "python_worker",
                    "operations": {"train": {"contract": "qualification.Train"}},
                }
            },
            "workflow": {
                "entrypoint": "train",
                "nodes": {
                    "train": {
                        "invoke": {
                            "component": "trainer",
                            "operation": "train",
                            "training": {"model_family": family, "components": {}},
                        },
                        "publishes": publishes,
                        "effect": "process",
                    }
                },
            },
            "execution": {
                "component": "trainer",
                "operation": "train",
                "gpu_trace": {
                    "enabled": True,
                    "backend": "torch",
                    "warmup_steps": 1,
                    "skip_steps": 0,
                    "capture_steps": 2,
                    "output_artifact": "gpu_trace",
                    "activities": ["cpu", "accelerator"],
                },
            },
            "artifacts": artifacts,
            "observability": observability,
            "recovery": {
                "exact_resume": False,
                "checkpoint_artifact": "checkpoint",
                "reconcile": "restart_from_checkpoint",
            },
        },
    }


@pytest.mark.parametrize("family", MODULE.FAMILIES)
def test_runner_accepts_only_real_resumable_profiled_family_documents(
    tmp_path: Path, family: str
) -> None:
    path = tmp_path / f"{family}.json"
    path.write_text(json.dumps(_document(family)))
    document, source = MODULE.load_qualification_document(family, path)
    assert document["metadata"]["name"] == f"production-{family}-qualification"
    assert json.loads(source) == document


def test_runner_rejects_coverage_adapter_and_missing_mageflow_gallery(
    tmp_path: Path,
) -> None:
    coverage = _document("transformer")
    coverage["spec"]["components"]["trainer"]["adapter"] = "coverage.transformer-mla"
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))
    with pytest.raises(MODULE.QualificationRunError, match="no registered"):
        MODULE.load_qualification_document("transformer", path)

    no_gallery = _document("mageflow")
    del no_gallery["spec"]["workflow"]["nodes"]["train"]["publishes"][
        "eval_gallery"
    ]
    path = tmp_path / "no-gallery.json"
    path.write_text(json.dumps(no_gallery))
    with pytest.raises(MODULE.QualificationRunError, match="side-by-side"):
        MODULE.load_qualification_document("mageflow", path)


class FakeAuthorityClient:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.states: dict[str, str] = {}
        self.resumed: set[str] = set()
        self.actions: list[tuple[str, str]] = []
        self.health_checked = False

    def health(self) -> None:
        self.health_checked = True

    def get(self, path: str) -> Any:
        if path == "/api/trainvm/runs":
            return {"journal_id": "journal-1", "commands_enabled": True}
        if path == "/api/trainvm/host-authority":
            return {"active_process_count": 0, "active_fence_count": 0}
        run_id = path.rsplit("/", 1)[-1]
        state = self.states[run_id]
        optimizer_step = 4 if state == "completed" else 3
        return {
            "run_id": run_id,
            "plan_hash": run_id.replace("-run", "-plan"),
            "run_revision": len(self.actions) + 1,
            "observed_state": state,
            "desired_state": state,
            "optimizer_step": optimizer_step,
            "last_event_sequence": 10,
            "failure_summary": "",
        }

    def paged(self, path: str, through: int) -> list[dict[str, Any]]:
        assert through == 10
        run_id = path.split("/runs/", 1)[1].split("/", 1)[0]
        if path.endswith("/timeline"):
            if run_id not in self.resumed:
                return []
            return [
                {
                    "sequence": 8,
                    "run_id": run_id,
                    "event_type": "node.attempt_restarted",
                    "optimizer_step": 3,
                    "payload": {"checkpoint_artifact_id": f"{run_id}-checkpoint"},
                }
            ]
        if path.endswith("/metrics"):
            return [
                {
                    "sequence": 9,
                    "run_id": run_id,
                    "name": "train.loss",
                    "value": 1.0,
                    "optimizer_step": 4,
                }
            ]
        raise AssertionError(path)

    def post(self, path: str, value: Any, expected: set[int]) -> tuple[int, Any]:
        del expected
        if path == "/api/trainvm/compile":
            family = value["metadata"]["name"].split("-")[1]
            return 200, {
                "valid": True,
                "plan_hash": f"{family}-plan",
                "adapter_lock_digest": "adapter-lock",
                "training_component_lock_digest": "component-lock",
            }
        if path == "/api/trainvm/experiments":
            family = json.loads(value["source_document"])["metadata"]["name"].split(
                "-"
            )[1]
            run_id = f"{family}-run"
            self.states[run_id] = "running"
            return 202, {
                "run": {"run_id": run_id, "plan_hash": f"{family}-plan"}
            }
        run_id = path.split("/")[-2]
        action = value["action"]
        self.actions.append((run_id, action))
        if action == "resume":
            self.resumed.add(run_id)
        self.states[run_id] = "paused" if action == "pause" else "completed"
        return 202, {"disposition": "ACCEPTED", "action": action}


def test_runner_drives_checkpoint_release_resume_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeAuthorityClient("http://127.0.0.1:9124")
    monkeypatch.setattr(MODULE, "AuthorityClient", lambda origin: client)
    captured: dict[str, Any] = {}

    def capture(origin: str, runs: dict[str, str], output: Path) -> Path:
        captured.update(origin=origin, runs=dict(runs), output=output)
        return output / "live-capture.json"

    monkeypatch.setattr(MODULE.CAPTURE, "live_capture", capture)
    sources = {
        family: json.dumps(_document(family), separators=(",", ":"))
        for family in MODULE.FAMILIES
    }
    output = tmp_path / "live"
    result = MODULE.qualify(
        client.origin,
        sources,
        {family: 2 for family in MODULE.FAMILIES},
        output,
        timeout_seconds=30,
        poll_seconds=0.05,
    )
    assert result == output / "live-capture.json"
    assert client.health_checked
    assert client.actions == [
        ("mageflow-run", "pause"),
        ("mageflow-run", "resume"),
        ("rwkv-run", "pause"),
        ("rwkv-run", "resume"),
        ("transformer-run", "pause"),
        ("transformer-run", "resume"),
    ]
    assert captured["runs"] == {
        family: f"{family}-run" for family in MODULE.FAMILIES
    }


@pytest.mark.parametrize(
    ("initial_state", "expected_actions"),
    [
        ("paused", [("rwkv-run", "resume")]),
        ("completed", []),
    ],
)
def test_family_driver_reconciles_an_interrupted_controller(
    initial_state, expected_actions
) -> None:
    client = FakeAuthorityClient("http://127.0.0.1:9124")
    client.states["rwkv-run"] = initial_state
    if initial_state == "completed":
        client.resumed.add("rwkv-run")

    MODULE._drive_family(
        client,
        MODULE.SubmittedRun("rwkv", "rwkv-run", "rwkv-plan"),
        pause_step=2,
        deadline=MODULE.time.monotonic() + 1,
        poll_seconds=0.05,
    )

    assert client.actions == expected_actions


def test_wait_rejects_run_that_finishes_before_pause() -> None:
    client = FakeAuthorityClient("http://127.0.0.1:9124")
    client.states["rwkv-run"] = "completed"
    run = MODULE.SubmittedRun("rwkv", "rwkv-run", "rwkv-plan")
    with pytest.raises(MODULE.QualificationRunError, match="completed before"):
        MODULE._wait(
            client,
            run,
            lambda view: view["observed_state"] == "paused",
            "checkpoint and pause",
            deadline=MODULE.time.monotonic() + 1,
            poll_seconds=0.05,
        )


@pytest.mark.parametrize(
    ("final_step", "timeline", "metrics", "message"),
    [
        (3, [], [], "no authoritative optimizer progress"),
        (4, [], [], "no checkpoint-bound restart"),
        (
            4,
            [
                {
                    "sequence": 8,
                    "run_id": "rwkv-run",
                    "event_type": "node.attempt_restarted",
                    "optimizer_step": 3,
                    "payload": {"checkpoint_artifact_id": "checkpoint-3"},
                }
            ],
            [],
            "no finite live metric",
        ),
    ],
)
def test_resume_proof_fails_closed_without_post_restart_progress(
    final_step, timeline, metrics, message
) -> None:
    class EvidenceClient:
        def paged(self, path: str, through: int) -> list[dict[str, Any]]:
            assert through == 10
            return timeline if path.endswith("/timeline") else metrics

    with pytest.raises(MODULE.QualificationRunError, match=message):
        MODULE._prove_resume_progress(
            EvidenceClient(),
            MODULE.SubmittedRun("rwkv", "rwkv-run", "rwkv-plan"),
            {"optimizer_step": 3},
            {"optimizer_step": final_step, "last_event_sequence": 10},
        )
