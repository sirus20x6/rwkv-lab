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
    / "capture_trainvm_production_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("trainvm_production_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ACCEPTANCE_FIXTURE_PATH = Path(__file__).with_name(
    "test_trainvm_production_acceptance.py"
)
ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "trainvm_production_acceptance_fixture", ACCEPTANCE_FIXTURE_PATH
)
assert ACCEPTANCE_SPEC is not None and ACCEPTANCE_SPEC.loader is not None
ACCEPTANCE_FIXTURE = importlib.util.module_from_spec(ACCEPTANCE_SPEC)
sys.modules[ACCEPTANCE_SPEC.name] = ACCEPTANCE_FIXTURE
ACCEPTANCE_SPEC.loader.exec_module(ACCEPTANCE_FIXTURE)


class FakeDashboard:
    def __init__(self, origin: str, *, fail_family: str = "") -> None:
        self.origin = origin
        self.fail_family = fail_family
        self.health_checked = False

    def health(self) -> None:
        self.health_checked = True

    def _family(self, path: str) -> str:
        return next(family for family in MODULE.FAMILIES if family in path)

    def get(self, path: str) -> Any:
        if path == "/api/trainvm/host-authority":
            return {"api_version": "trainvm.hostd-authority-status/v1"}
        family = self._family(path)
        if family == self.fail_family:
            raise MODULE.CaptureError("injected capture failure")
        run_id = f"{family}-run"
        base = f"/api/trainvm/runs/{run_id}"
        if path == base:
            return {
                "run_id": run_id,
                "plan_hash": f"{family}-plan",
                "run_revision": 7,
                "desired_state": "completed",
                "observed_state": "completed",
                "last_event_sequence": 9,
            }
        if path == base + "/plan":
            return {"run_id": run_id, "plan_hash": f"{family}-plan"}
        if path.startswith(base + "/checkpoints"):
            return [{"valid": True, "artifact_id": f"{family}-checkpoint"}]
        if path == base + "/galleries":
            return {
                "galleries": [
                    {"sequence": 3, "artifact_id": "old-gallery"},
                    {"sequence": 8, "artifact_id": "new-gallery"},
                ]
                if family == "mageflow"
                else [],
                "history_truncated": False,
            }
        if path == base + "/galleries/new-gallery":
            return {
                "artifact_id": "new-gallery",
                "items": [
                    {
                        "generated_image_url": "/generated",
                        "target_image_url": "/original",
                    }
                ],
            }
        if path == base + "/profiles":
            return [{"artifact_id": f"{family}-trace"}]
        if path == base + "/capsule":
            return {"body": {"run": {"run_id": run_id}}}
        raise AssertionError(f"unexpected GET {path}")

    def paged(self, path: str, through: int) -> list[dict[str, Any]]:
        family = self._family(path)
        run_id = f"{family}-run"
        if path.endswith("/timeline"):
            return [
                {"sequence": 1, "run_id": run_id, "event_type": "run.created"},
                {
                    "sequence": through,
                    "run_id": run_id,
                    "event_type": "run.observed_state_changed",
                },
            ]
        if path.endswith("/metrics"):
            return [{"sequence": 4, "run_id": run_id, "name": "eval.loss"}]
        if path.endswith("/artifacts"):
            return [{"sequence": 5, "run_id": run_id, "kind": "checkpoint"}]
        raise AssertionError(f"unexpected paged GET {path}")


def test_live_capture_is_atomic_and_selects_latest_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients: list[FakeDashboard] = []

    def factory(origin: str) -> FakeDashboard:
        client = FakeDashboard(origin)
        clients.append(client)
        return client

    monkeypatch.setattr(MODULE, "Dashboard", factory)
    output = tmp_path / "capture"
    result = MODULE.live_capture(
        "http://127.0.0.1:9124",
        {family: f"{family}-run" for family in MODULE.FAMILIES},
        output,
    )
    assert result == output / "live-capture.json"
    assert clients[0].health_checked
    manifest = json.loads(result.read_text())
    assert set(manifest["runs"]) == set(MODULE.FAMILIES)
    gallery = json.loads((output / manifest["runs"]["mageflow"]["gallery"]["path"]).read_text())
    assert gallery["artifact_id"] == "new-gallery"
    assert gallery["items"][0]["target_image_url"] == "/original"


def test_failed_live_capture_leaves_no_claimable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "Dashboard",
        lambda origin: FakeDashboard(origin, fail_family="rwkv"),
    )
    output = tmp_path / "capture"
    with pytest.raises(MODULE.CaptureError, match="injected capture failure"):
        MODULE.live_capture(
            "http://127.0.0.1:9124",
            {family: f"{family}-run" for family in MODULE.FAMILIES},
            output,
        )
    assert not output.exists()


def test_finalize_seals_capture_and_runs_offline_gate(tmp_path: Path) -> None:
    root = tmp_path / "live"
    root.mkdir()
    original_bundle = ACCEPTANCE_FIXTURE._bundle(root)
    manifest = json.loads(original_bundle.read_text())
    original_bundle.unlink()
    capture = {
        "api_version": MODULE.CAPTURE_VERSION,
        "dashboard_origin": "http://127.0.0.1:9124",
        "host_authority": manifest["host_authority"],
        "runs": manifest["runs"],
    }
    capture_path = root / "live-capture.json"
    capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    receipt = tmp_path / "production-receipt.json"
    result = MODULE.finalize_capture(
        capture_path,
        root / "acceptance.json",
        root / "hostd-crash.json",
        root / "journal.json",
        receipt,
    )
    assert result == root / "bundle.json"
    assert json.loads(receipt.read_text())["gate_open"] is True
    assert json.loads(result.read_text())["source_commit"] == "a" * 40


@pytest.mark.parametrize(
    "value",
    [
        "https://127.0.0.1:9124",
        "http://192.0.2.1:9124",
        "http://user@localhost:9124",
        "http://localhost:9124/path",
    ],
)
def test_dashboard_capture_refuses_nonlocal_or_ambiguous_origins(value: str) -> None:
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE._local_dashboard_url(value)
