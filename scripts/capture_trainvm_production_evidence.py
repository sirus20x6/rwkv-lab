#!/usr/bin/env python3
"""Capture and seal real TrainVM production-acceptance evidence.

Run ``live`` while the completed runs are still visible through the dashboard.
After stopping TrainVM and verifying its journal, run ``finalize`` to attach the
developer, hostd-crash, and journal receipts and invoke the offline gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CAPTURE_VERSION = "trainvm.production-live-capture/v1"
BUNDLE_VERSION = "trainvm.production-acceptance-bundle/v1"
FAMILIES = ("mageflow", "rwkv", "transformer")
MAXIMUM_RESPONSE_BYTES = 64 * 1024 * 1024
MAXIMUM_STREAM_ITEMS = 100_000


class CaptureError(RuntimeError):
    """Live evidence could not be captured coherently."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise CaptureError(f"non-finite JSON number: {value}")


def _parse_json(raw: bytes | str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_invalid_constant,
    )


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish(root: Path, relative: str, value: Any) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"path": relative, "sha256": _digest(encoded)}


def _copy_json(root: Path, relative: str, source: Path) -> dict[str, str]:
    try:
        if source.is_symlink():
            raise CaptureError(f"evidence source must not be a symlink: {source}")
        source = source.resolve(strict=True)
        if not source.is_file():
            raise CaptureError(f"evidence source is not a regular file: {source}")
        raw = source.read_bytes()
        _parse_json(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot copy JSON evidence {source}: {error}") from error
    destination = root / relative
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CaptureError(f"refusing to replace evidence: {destination}")
        existing = destination.read_bytes()
        if existing != raw:
            raise CaptureError(f"refusing to replace changed evidence: {destination}")
        return {"path": relative, "sha256": _digest(existing)}
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"path": relative, "sha256": _digest(raw)}


def _local_dashboard_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or parsed.hostname is None
    ):
        raise argparse.ArgumentTypeError("dashboard URL must be a local HTTP origin")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if not loopback:
        raise argparse.ArgumentTypeError("dashboard URL must resolve textually to loopback")
    try:
        _ = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("dashboard URL port is malformed") from error
    return value.rstrip("/")


def _run_argument(value: str) -> tuple[str, str]:
    family, separator, run_id = value.partition("=")
    if not separator or family not in FAMILIES or not run_id or len(run_id) > 256:
        raise argparse.ArgumentTypeError("run must be mageflow|rwkv|transformer=RUN_ID")
    if any(character in run_id for character in "/\\\x00\r\n?#"):
        raise argparse.ArgumentTypeError("run ID contains an unsafe URL character")
    return family, run_id


class Dashboard:
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def get(self, path: str) -> Any:
        request = urllib.request.Request(
            self.origin + path,
            headers={"Accept": "application/json", "Cache-Control": "no-cache"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200 or response.geturl() != self.origin + path:
                    raise CaptureError(
                        f"GET {path} returned an unexpected status or redirect"
                    )
                raw = response.read(MAXIMUM_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError) as error:
            raise CaptureError(f"GET {path} failed: {error}") from error
        if len(raw) > MAXIMUM_RESPONSE_BYTES:
            raise CaptureError(f"GET {path} exceeded the capture bound")
        try:
            return _parse_json(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureError(f"GET {path} returned invalid JSON: {error}") from error

    def health(self) -> None:
        request = urllib.request.Request(
            self.origin + "/healthz", headers={"Cache-Control": "no-cache"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read(3)
                if (
                    response.status != 200
                    or response.geturl() != self.origin + "/healthz"
                    or raw != b"ok"
                ):
                    raise CaptureError("dashboard health check did not pass")
        except (OSError, urllib.error.URLError) as error:
            raise CaptureError(f"dashboard health check failed: {error}") from error

    def paged(self, path: str, through: int) -> list[Any]:
        result: list[Any] = []
        after = 0
        while after < through:
            separator = "&" if "?" in path else "?"
            page = self.get(f"{path}{separator}after={after}&limit=1000")
            if not isinstance(page, list):
                raise CaptureError(f"GET {path} did not return a list")
            if not page:
                break
            previous = after
            for item in page:
                sequence = item.get("sequence") if isinstance(item, dict) else None
                if (
                    isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence <= previous
                    or sequence > through
                ):
                    raise CaptureError(f"GET {path} returned an incoherent sequence")
                previous = sequence
                result.append(item)
            after = previous
            if len(result) > MAXIMUM_STREAM_ITEMS:
                raise CaptureError(f"GET {path} exceeded the item bound")
            if len(page) < 1000:
                break
        return result


def _capture_run(client: Dashboard, family: str, run_id: str) -> dict[str, Any]:
    base = "/api/trainvm/runs/" + urllib.parse.quote(run_id, safe="")
    before = client.get(base)
    if (
        not isinstance(before, dict)
        or before.get("run_id") != run_id
        or before.get("observed_state") != "completed"
    ):
        raise CaptureError(f"{family} run is not completed")
    through = before.get("last_event_sequence")
    if isinstance(through, bool) or not isinstance(through, int) or through < 1:
        raise CaptureError(f"{family} run has no terminal journal sequence")
    evidence = {
        "run": before,
        "plan": client.get(base + "/plan"),
        "timeline": client.paged(base + "/timeline", through),
        "metrics": client.paged(base + "/metrics", through),
        "artifacts": client.paged(base + "/artifacts", through),
        "checkpoints": client.get(base + "/checkpoints?limit=1000"),
        "galleries": client.get(base + "/galleries"),
        "profiles": client.get(base + "/profiles"),
        "capsule": client.get(base + "/capsule"),
    }
    if family == "mageflow":
        galleries = evidence["galleries"]
        history = galleries.get("galleries") if isinstance(galleries, dict) else None
        if not isinstance(history, list) or not history:
            raise CaptureError("MageFlow run has no eval gallery to capture")
        latest = max(
            history,
            key=lambda item: item.get("sequence", -1) if isinstance(item, dict) else -1,
        )
        artifact_id = latest.get("artifact_id") if isinstance(latest, dict) else None
        if not isinstance(artifact_id, str) or not artifact_id:
            raise CaptureError("MageFlow gallery history is malformed")
        evidence["gallery"] = client.get(
            base + "/galleries/" + urllib.parse.quote(artifact_id, safe="")
        )
    else:
        evidence["gallery"] = {}
    after = client.get(base)
    identity_fields = (
        "run_id",
        "plan_hash",
        "run_revision",
        "observed_state",
        "desired_state",
        "last_event_sequence",
    )
    if not isinstance(after, dict) or any(
        after.get(field) != before.get(field) for field in identity_fields
    ):
        raise CaptureError(f"{family} run changed while evidence was captured")
    if not evidence["timeline"] or evidence["timeline"][-1].get("sequence") != through:
        raise CaptureError(f"{family} timeline is not captured through its terminal event")
    return evidence


def live_capture(origin: str, runs: dict[str, str], output: Path) -> Path:
    if set(runs) != set(FAMILIES):
        raise CaptureError(f"exactly one run is required for each of {FAMILIES}")
    output = output.absolute()
    if output.exists():
        raise CaptureError(f"refusing to replace capture directory: {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.capture-", dir=output.parent)
    )
    try:
        os.chmod(staging, 0o700)
        client = Dashboard(origin)
        client.health()
        capture: dict[str, Any] = {
            "api_version": CAPTURE_VERSION,
            "dashboard_origin": origin,
            "host_authority": _publish(
                staging,
                "host-authority.json",
                client.get("/api/trainvm/host-authority"),
            ),
            "runs": {},
        }
        for family in FAMILIES:
            documents = _capture_run(client, family, runs[family])
            capture["runs"][family] = {
                name: _publish(staging, f"{family}/{name}.json", document)
                for name, document in documents.items()
            }
        _publish(staging, "live-capture.json", capture)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "live-capture.json"


def _load_capture(path: Path) -> dict[str, Any]:
    try:
        value = _parse_json(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read live capture: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("api_version") != CAPTURE_VERSION
        or not isinstance(value.get("runs"), dict)
        or set(value["runs"]) != set(FAMILIES)
    ):
        raise CaptureError("live capture manifest is malformed")
    return value


def _load_verifier() -> Any:
    path = Path(__file__).with_name("verify_trainvm_production_acceptance.py")
    spec = importlib.util.spec_from_file_location("trainvm_production_acceptance", path)
    if spec is None or spec.loader is None:
        raise CaptureError("cannot load production acceptance verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finalize_capture(
    capture_path: Path,
    acceptance: Path,
    hostd_crash: Path,
    journal: Path,
    receipt_path: Path,
) -> Path:
    capture_path = capture_path.resolve(strict=True)
    root = capture_path.parent
    capture = _load_capture(capture_path)
    try:
        acceptance_document = _parse_json(acceptance.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read developer acceptance receipt: {error}") from error
    commit = acceptance_document.get("commit") if isinstance(acceptance_document, dict) else None
    if not isinstance(commit, str):
        raise CaptureError("developer acceptance receipt has no source commit")
    bundle = {
        "api_version": BUNDLE_VERSION,
        "source_commit": commit,
        "acceptance": _copy_json(root, "developer-acceptance.json", acceptance),
        "hostd_crash": _copy_json(root, "hostd-crash.json", hostd_crash),
        "host_authority": capture["host_authority"],
        "journal_verification": _copy_json(root, "journal-verification.json", journal),
        "runs": capture["runs"],
    }
    bundle_path = root / "bundle.json"
    if bundle_path.exists():
        raise CaptureError(f"refusing to replace production bundle: {bundle_path}")
    _publish(root, bundle_path.name, bundle)
    verifier = _load_verifier()
    try:
        verified = verifier.verify_bundle(bundle_path)
    except BaseException:
        bundle_path.unlink(missing_ok=True)
        raise
    receipt_path = receipt_path.absolute()
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if receipt_path.exists():
        raise CaptureError(f"refusing to replace production receipt: {receipt_path}")
    encoded = _json_bytes(verified)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", suffix=".tmp", dir=receipt_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, receipt_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return bundle_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="capture completed runs through dashboard")
    live.add_argument("--dashboard-url", type=_local_dashboard_url, required=True)
    live.add_argument("--run", action="append", type=_run_argument, required=True)
    live.add_argument("--output", type=Path, required=True)
    finalize = commands.add_parser("finalize", help="attach offline receipts and verify")
    finalize.add_argument("--capture", type=Path, required=True)
    finalize.add_argument("--acceptance", type=Path, required=True)
    finalize.add_argument("--hostd-crash", type=Path, required=True)
    finalize.add_argument("--journal-verification", type=Path, required=True)
    finalize.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "live":
            runs = dict(arguments.run)
            if len(runs) != len(arguments.run):
                raise CaptureError("duplicate family run binding")
            result = live_capture(arguments.dashboard_url, runs, arguments.output)
        else:
            result = finalize_capture(
                arguments.capture,
                arguments.acceptance,
                arguments.hostd_crash,
                arguments.journal_verification,
                arguments.receipt,
            )
    except (CaptureError, ValueError) as error:
        print(f"production evidence capture failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
