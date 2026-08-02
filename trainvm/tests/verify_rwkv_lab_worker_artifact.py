#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent import futures
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def content_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def write_runtime_drift_archive(source: Path, output: Path) -> None:
    """Keep the closure manifest self-consistent but lie about one real file."""

    with zipfile.ZipFile(source) as archive:
        members = [(info, archive.read(info)) for info in archive.infolist()]
    replaced = False
    rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in members:
        if info.filename == "TRAINVM_RUNTIME_CLOSURE.json":
            document = json.loads(data)
            for entry in document["files"]:
                if entry["kind"] == "regular":
                    entry["sha256"] = "sha256:" + "0" * 64
                    replaced = True
                    break
            body = dict(document)
            body.pop("closure_digest")
            document["closure_digest"] = content_digest(canonical(body))
            data = canonical(document) + b"\n"
        rewritten.append((info, data))
    if not replaced:
        raise SystemExit("runtime closure fixture has no regular file to drift")
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for info, data in rewritten:
            archive.writestr(info, data)


def run_completed_replay(
    archive: Path,
    source_root: Path,
    code_fingerprint: str,
    capabilities: list[str],
    directory: Path,
) -> None:
    sys.path.insert(0, str(source_root))
    import grpc

    from trainvm.v1 import trainvm_pb2 as wire
    from trainvm.v1 import trainvm_pb2_grpc as wire_grpc

    socket_path = directory / "worker-control.sock"
    target = f"unix:{socket_path}"
    invocation_body = {
        "adapter": {
            "adapter": "rwkv-lab.mageflow-appearance-expert",
            "contract": "rwkv_lab.mageflow_appearance_expert.v1.Train",
            "operation": "train",
            "runtime": "python_worker",
            "version": "1.0.0",
        },
        "api_version": "trainvm.worker-invocation/v1",
        "attempt_id": "artifact-attempt",
        "controls": {},
        "dispatch_id": "artifact-dispatch",
        "effective_control_revision": 0,
        "execution": None,
        "host_id": "sha256:" + "b" * 64,
        "inputs": {},
        "node_id": "train",
        "observability": {},
        "plan_hash": "c" * 64,
        "plan_revision": 1,
        "publishes": {},
        "resources": {},
        "run_id": "artifact-run",
        "training": None,
        "workspace": {},
    }
    invocation = canonical(
        {
            **invocation_body,
            "invocation_digest": content_digest(canonical(invocation_body)),
        }
    )
    observed: list[object] = []
    errors: list[BaseException] = []

    class Controller(wire_grpc.WorkerControlServicer):
        def Connect(self, request_iterator, context):
            try:
                first = next(request_iterator)
                if first.WhichOneof("message") != "hello":
                    raise AssertionError("packaged worker did not send Hello first")
                hello = first.hello
                observed.append(hello)
                yield wire.ControllerToWorker(
                    welcome=wire.WorkerWelcome(
                        disposition=wire.WorkerWelcome.DISPOSITION_ALREADY_COMPLETED,
                        journal_id="artifact-journal",
                        plan_hash="c" * 64,
                        plan_revision=1,
                        run_id=hello.run_id,
                        run_revision=2,
                        node_id=hello.node_id,
                        attempt_id=hello.attempt_id,
                        launch_nonce=hello.launch_nonce,
                        concurrency_key=hello.concurrency_key,
                        lease_id=hello.lease_id,
                        fencing_token=hello.fencing_token,
                        dispatch_id="artifact-dispatch",
                        component="trainer",
                        operation="train",
                        acknowledged_worker_sequence=0,
                        canonical_invocation_json=invocation,
                        invocation_digest=json.loads(invocation)[
                            "invocation_digest"
                        ],
                    )
                )
                observed.extend(request_iterator)
            except grpc.RpcError as error:
                # A completed worker closes its request stream immediately after
                # Welcome. grpcio may surface that normal peer cancellation to
                # the server iterator depending on scheduler timing.
                if not observed or error.code() != grpc.StatusCode.CANCELLED:
                    errors.append(error)
                    context.abort(grpc.StatusCode.INTERNAL, "fixture rejected worker")
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                context.abort(grpc.StatusCode.INTERNAL, "fixture rejected worker")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    wire_grpc.add_WorkerControlServicer_to_server(Controller(), server)
    if server.add_insecure_port(target) != 1:
        raise SystemExit("could not bind packaged-worker gRPC fixture")
    server.start()
    bootstrap_body = {
        "adapter": "rwkv-lab.mageflow-appearance-expert",
        "adapter_version": "1.0.0",
        "api_version": "trainvm.worker-bootstrap/v1",
        "attempt_id": "artifact-attempt",
        "capabilities": capabilities,
        "code_fingerprint": code_fingerprint,
        "concurrency_key": "gpu:0",
        "controller_target": target,
        "fencing_token": 1,
        "last_acked_controller_sequence": 0,
        "launch_nonce": "artifact-launch",
        "lease_id": "artifact-lease",
        "node_id": "train",
        "run_id": "artifact-run",
    }
    bootstrap = canonical(
        {
            **bootstrap_body,
            "bootstrap_digest": content_digest(canonical(bootstrap_body)),
        }
    )
    descriptor = os.memfd_create(
        "trainvm-artifact-bootstrap", getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    )
    output_read, output_write = os.pipe2(os.O_CLOEXEC)
    try:
        os.write(descriptor, bootstrap)
        fcntl.fcntl(
            descriptor,
            getattr(fcntl, "F_ADD_SEALS", 1033),
            getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_SEAL", 0x0001),
        )
        actions = [
            (os.POSIX_SPAWN_DUP2, descriptor, 4),
            (os.POSIX_SPAWN_DUP2, output_write, 1),
            (os.POSIX_SPAWN_DUP2, output_write, 2),
            (os.POSIX_SPAWN_CLOSE, output_read),
        ]
        child = os.posix_spawn(
            sys.executable,
            [
                sys.executable,
                "-I",
                str(archive),
                "--trainvm-bootstrap-fd=4",
            ],
            {},
            file_actions=actions,
        )
        os.close(output_write)
        output_write = -1
        deadline = time.monotonic() + 30
        status = None
        while time.monotonic() < deadline:
            waited, candidate = os.waitpid(child, os.WNOHANG)
            if waited == child:
                status = candidate
                break
            time.sleep(0.01)
        if status is None:
            os.kill(child, 9)
            os.waitpid(child, 0)
            raise SystemExit("packaged worker replay timed out")
        output = bytearray()
        while block := os.read(output_read, 64 * 1024):
            output.extend(block)
        if os.waitstatus_to_exitcode(status) != 0:
            raise SystemExit(
                "packaged worker replay failed: "
                + output.decode("utf-8", errors="replace")[-2000:]
            )
    finally:
        if output_write >= 0:
            os.close(output_write)
        os.close(output_read)
        os.close(descriptor)
        server.stop(0).wait()
    if errors or len(observed) != 1:
        raise SystemExit(f"packaged worker replay protocol drift: {errors!r}")
    hello = observed[0]
    if (
        hello.code_fingerprint != code_fingerprint
        or list(hello.capabilities) != capabilities
        or hello.launch_nonce != "artifact-launch"
    ):
        raise SystemExit("packaged worker Hello disagrees with sealed bootstrap")


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: verify_rwkv_lab_worker_artifact.py BUILDER CLOSURE_BUILDER MATERIALIZER TRAINVM SOURCE_ROOT"
        )
    builder = Path(sys.argv[1]).resolve(strict=True)
    closure_builder = Path(sys.argv[2]).resolve(strict=True)
    materializer = Path(sys.argv[3]).resolve(strict=True)
    trainvm = Path(sys.argv[4]).resolve(strict=True)
    source_root = Path(sys.argv[5]).resolve(strict=True)
    requirements = json.loads(
        subprocess.check_output(
            [str(trainvm), "inspect-rwkv-lab-runtime-requirements"], text=True
        )
    )
    if requirements["api_version"] != (
        "trainvm.rwkv-lab-worker-runtime-requirements/v1"
    ):
        raise SystemExit("native runtime requirements schema drifted")
    adapters = tuple(profile["adapter"] for profile in requirements["profiles"])
    shared_distributions = tuple(requirements["shared_root_distributions"])
    if (
        len(adapters) != 13
        or len(set(adapters)) != len(adapters)
        or shared_distributions != tuple(sorted(set(shared_distributions)))
        or any(
            not set(shared_distributions).issubset(profile["root_distributions"])
            for profile in requirements["profiles"]
        )
    ):
        raise SystemExit("native runtime requirements are not canonical")

    module_spec = importlib.util.spec_from_file_location(
        "materialize_trainvm_worker_deployment", materializer
    )
    if module_spec is None or module_spec.loader is None:
        raise SystemExit("could not load deployment materializer")
    sys.path.insert(0, str(materializer.parent))
    materializer_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(materializer_module)
    grouped = materializer_module._runtime_groups(
        adapters,
        {
            adapter: Path(
                "/runtime/mageflow"
                if "mageflow" in adapter
                else f"/runtime/{adapter}"
            )
            for adapter in adapters
        },
        {
            profile["adapter"]: tuple(profile["root_distributions"])
            for profile in requirements["profiles"]
        },
    )
    if len(grouped) != len(adapters) - 1 or not any(
        members == list(adapters[:2]) for members in grouped.values()
    ):
        raise SystemExit("per-adapter runtime grouping drifted from native requirements")
    with tempfile.TemporaryDirectory(prefix="trainvm-worker-artifact-") as raw:
        directory = Path(raw)
        first = directory / "first.pyz"
        second = directory / "second.pyz"
        runtime_closure = directory / "runtime-closure.json"
        subprocess.check_call(
            [
                sys.executable,
                "-I",
                str(closure_builder),
                "--output",
                str(runtime_closure),
            ]
        )
        receipts = []
        for output in (first, second):
            receipts.append(
                json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            str(builder),
                            "--source-root",
                            str(source_root),
                            "--output",
                            str(output),
                            "--runtime-closure",
                            str(runtime_closure),
                        ],
                        text=True,
                    )
                )
            )
        if first.read_bytes() != second.read_bytes():
            raise SystemExit("worker artifact build is not byte deterministic")
        if receipts[0]["archive_sha256"] != digest(first) or receipts[0][
            "archive_sha256"
        ] != receipts[1]["archive_sha256"]:
            raise SystemExit("worker artifact receipt does not bind exact bytes")
        closure_document = json.loads(runtime_closure.read_text())
        if (
            receipts[0]["runtime_closure_digest"]
            != closure_document["closure_digest"]
            or receipts[1]["runtime_closure_digest"]
            != closure_document["closure_digest"]
        ):
            raise SystemExit("worker artifact receipt does not bind its runtime closure")

        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            if (
                names.count("__main__.py") != 1
                or names.count("TRAINVM_WORKER_MANIFEST.json") != 1
                or any("__pycache__" in name or name.endswith(".pyc") for name in names)
            ):
                raise SystemExit("worker artifact member surface is not closed")
            manifest = json.loads(archive.read("TRAINVM_WORKER_MANIFEST.json"))
            expected = {
                item["path"]: (item["size"], item["sha256"])
                for item in manifest["members"]
            }
            for name, (size, sha256) in expected.items():
                data = archive.read(name)
                if len(data) != size or "sha256:" + hashlib.sha256(data).hexdigest() != sha256:
                    raise SystemExit(f"worker member manifest mismatch: {name}")

        loaded = subprocess.run(
            [sys.executable, "-I", str(first), "--artifact-load-test"],
            env={},
            text=True,
            capture_output=True,
            check=False,
        )
        if loaded.returncode == 0 or (
            "authority worker accepts only its fixed bootstrap descriptor"
            not in loaded.stderr
        ):
            raise SystemExit(
                "isolated worker artifact did not reach its closed argv boundary: "
                + loaded.stderr[-1000:]
            )

        drifted = directory / "runtime-drift.pyz"
        write_runtime_drift_archive(first, drifted)
        drift_result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(drifted),
                "--trainvm-bootstrap-fd=4",
            ],
            env={},
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            drift_result.returncode == 0
            or "RuntimeClosureError" not in drift_result.stderr
            or "runtime closure file content changed" not in drift_result.stderr
        ):
            raise SystemExit(
                "worker did not reject self-consistent runtime closure drift before bootstrap: "
                + drift_result.stderr[-2000:]
            )

        python_path = Path(sys.executable).resolve(strict=True)
        python_root = Path(python_path.anchor) / "usr"
        if not python_root.is_dir() or python_root not in python_path.parents:
            python_root = python_path.parent
        deployment = json.loads(
            subprocess.check_output(
                [str(trainvm), "inspect-rwkv-lab-deployment"],
                input=json.dumps(
                    {
                        "api_version": "trainvm.rwkv-lab-worker-runtimes/v1",
                        "runtimes": [
                            {
                                "adapter": adapter,
                                "code_path": str(first),
                                "code_fingerprint": digest(first),
                                "bootstrap_runtime_closure_fingerprint": closure_document[
                                    "closure_digest"
                                ],
                                "executable_path": str(python_path),
                                "executable_fingerprint": digest(python_path),
                                "working_directory": str(directory),
                            }
                            for adapter in adapters
                        ],
                        "trusted_roots": [str(directory), str(python_root)],
                    },
                    separators=(",", ":"),
                ),
                text=True,
            )
        )
        profiles = deployment["host_launch_registry"]["profiles"]
        if deployment["schema"] != "trainvm.rwkv-lab-worker-deployment/v3":
            raise SystemExit("deployment inspector emitted the wrong schema")
        if len(profiles) != 13 or any(
            profile["code_argument_index"] != 1
            or profile["public_arguments"] != ["-I", "rwkv-lab-worker.pyz"]
            or profile["code_fingerprint"] != digest(first)
            or profile["bootstrap_runtime_closure_fingerprint"]
            != closure_document["closure_digest"]
            for profile in profiles
        ):
            raise SystemExit("deployment registry drifted from sealed worker artifact")
        run_completed_replay(
            first,
            source_root,
            digest(first),
            deployment["provided_capabilities"],
            directory,
        )

        deployment_directory = directory / "deployment"
        materialize = [
            sys.executable,
            str(materializer),
            "--trainvm",
            str(trainvm),
            "--python",
            str(python_path),
            "--source-root",
            str(source_root),
            "--output-directory",
            str(deployment_directory),
            "--working-directory",
            str(directory),
            "--trusted-root",
            str(directory),
            "--trusted-root",
            str(python_root),
        ]
        first_materialization = json.loads(
            subprocess.check_output(materialize, text=True)
        )
        second_materialization = json.loads(
            subprocess.check_output(materialize, text=True)
        )
        if first_materialization != second_materialization:
            raise SystemExit("worker deployment materialization is not idempotent")
        for name in ("adapters.json", "host-launches.json", "deployment-receipt.json"):
            if not (deployment_directory / name).is_file():
                raise SystemExit(f"worker deployment omitted {name}")
        if len(first_materialization["runtime_groups"]) != 1:
            raise SystemExit("shared-interpreter fixture did not coalesce runtime groups")
        runtime_group = first_materialization["runtime_groups"][0]
        if runtime_group["adapters"] != list(adapters):
            raise SystemExit("materialized runtime group has the wrong adapter membership")
        deployed_artifact = Path(runtime_group["worker_artifact"]["output"])
        deployed_closure = Path(runtime_group["runtime_closure"]["output"])
        if not deployed_artifact.is_file() or not deployed_closure.is_file():
            raise SystemExit("worker deployment omitted grouped runtime payloads")
        deployed_closure_document = json.loads(deployed_closure.read_text())
        deployed_profiles = json.loads(
            (deployment_directory / "host-launches.json").read_text()
        )["profiles"]
        if (
            first_materialization["schema"]
            != "trainvm.rwkv-lab-worker-deployment-materialization/v3"
            or runtime_group["runtime_closure"]["closure_digest"]
            != deployed_closure_document["closure_digest"]
            or runtime_group["runtime_closure"]["manifest_sha256"]
            != digest(deployed_closure)
            or runtime_group["worker_artifact"]["archive_sha256"]
            != digest(deployed_artifact)
            or any(
                profile["bootstrap_runtime_closure_fingerprint"]
                != deployed_closure_document["closure_digest"]
                for profile in deployed_profiles
            )
        ):
            raise SystemExit(
                "deployment receipt, closure bytes, and host profiles disagree"
            )
    print("deterministic isolated rwkv_lab worker artifact passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
