#!/usr/bin/env python3
"""Real-browser contract test for the descriptor-driven TrainVM composer."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    raise SystemExit(77)


WEB_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WEB_ROOT.parents[1]
SCHEMA = json.loads(
    (REPOSITORY_ROOT / "docs/experiment-vm/experiment-v1.schema.json").read_text()
)
EXAMPLE = json.loads(
    (REPOSITORY_ROOT / "docs/experiment-vm/examples/mageflow-cache-resume.json").read_text()
)

COMPONENTS = {
    "api_version": "trainvm.training-components/v1",
    "components": [
        {
            "backend": "runtime_builtin",
            "configuration": [
                {
                    "minimum": 0.0,
                    "name": "gain",
                    "required": True,
                    "type": "number",
                }
            ],
            "implementation": "test.synthetic_activation_a",
            "key": {"category": "activation", "name": "a:b", "version": "c"},
            "model_families": ["synthetic"],
            "reference_implementation": True,
            "required_capabilities": [],
            "state": [],
            "state_grade": "stateless",
        },
        {
            "backend": "runtime_builtin",
            "configuration": [
                {
                    "minimum": 0.0,
                    "name": "gain",
                    "required": True,
                    "type": "number",
                }
            ],
            "implementation": "test.synthetic_activation_b",
            "key": {"category": "activation", "name": "a", "version": "b:c"},
            "model_families": ["synthetic"],
            "reference_implementation": True,
            "required_capabilities": [],
            "state": [],
            "state_grade": "stateless",
        }
    ],
}

def operation(adapter: str, version: str) -> dict[str, object]:
    return {
            "authoring": {
                "inputs": {
                    "config": {
                        "description": "Synthetic generic browser fixture.",
                        "required": True,
                        "type": "object",
                    }
                },
                "outputs": {
                    "checkpoint": {
                        "artifact_type": "checkpoint",
                        "required": False,
                        "type": "artifact",
                    }
                },
            },
            "code_fingerprint": "sha256:" + "1" * 64,
            "effect": "process",
            "idempotency": "receipt_required",
            "key": {
                "adapter": adapter,
                "contract": "test.synthetic.v1.Train",
                "operation": "train",
                "runtime": "python_worker",
                "version": version,
            },
            "lifecycle": {
                "checkpoint_now": False,
                "compile": False,
                "graceful_stop": False,
                "pause_keep_resources": False,
                "pause_release_resources": False,
                "profile": False,
                "qualify": False,
                "resume_grade": "none",
                "stateful": False,
                "warmup": False,
            },
            "required_capabilities": [],
            "training_composition": {
                "model_family": "synthetic",
                "slots": {"activation": "activation"},
            },
        }


OPERATIONS = {
    "api_version": "trainvm.operations/v1",
    "operations": [
        operation("test:synthetic", "trainer"),
        operation("test", "synthetic:trainer"),
    ],
}

HTML = """<!doctype html><html><body>
<details id="trainvm-authoring">
  <button id="vm-load-example"></button><button id="vm-compile"></button>
  <button id="vm-diff"></button><input id="vm-submit-reason" />
  <button id="vm-submit" disabled></button><span id="vm-editor-state"></span>
  <span id="vm-operation-registry-state"></span><div id="vm-operation-composer"></div>
  <span id="vm-component-registry-state"></span><div id="vm-component-composer"></div>
  <div id="vm-schema-form"></div><textarea id="vm-json-source"></textarea>
  <button id="vm-apply-json"></button><div id="vm-diagnostics"></div>
  <div id="vm-plan-preview"></div><div id="vm-plan-diff"></div>
</details><script src="/trainvm-editor.js"></script></body></html>"""


def compact(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tuple_identity(*parts: str) -> str:
    return json.dumps(list(parts), separators=(",", ":"))


class Handler(BaseHTTPRequestHandler):
    submitted: dict[str, object] | None = None

    def log_message(self, *_: object) -> None:
        pass

    def send_json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/trainvm-editor.js":
            body = (WEB_ROOT / "static/trainvm-editor.js").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/trainvm/schema":
            self.send_json(SCHEMA)
        elif self.path == "/api/trainvm/example":
            self.send_json(EXAMPLE)
        elif self.path == "/api/trainvm/runs":
            self.send_json({"journal_id": "journal-browser", "runs": []})
        elif self.path == "/api/trainvm/training-components":
            raw = compact(COMPONENTS)
            self.send_json({"schema": COMPONENTS, "schema_hash": "sha256:" + hashlib.sha256(raw).hexdigest()})
        elif self.path == "/api/trainvm/operations":
            raw = compact(OPERATIONS)
            self.send_json({"schema": OPERATIONS, "schema_hash": "sha256:" + hashlib.sha256(raw).hexdigest()})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/trainvm/compile":
            self.send_json(
                {
                    "adapter_lock_digest": "sha256:adapter-browser",
                    "artifact_count": len(payload.get("spec", {}).get("artifacts", {})),
                    "canonical_plan": payload,
                    "control_count": len(payload.get("spec", {}).get("controls", {}).get("catalog", {})),
                    "diagnostics": [],
                    "entrypoint": payload.get("spec", {}).get("workflow", {}).get("entrypoint", ""),
                    "experiment": payload.get("metadata", {}).get("name", ""),
                    "node_count": len(payload.get("spec", {}).get("workflow", {}).get("nodes", {})),
                    "plan_hash": "sha256:plan-browser",
                    "training_component_lock_digest": "",
                    "valid": True,
                }
            )
        elif self.path == "/api/trainvm/experiments":
            type(self).submitted = payload
            self.send_json(
                {
                    "adapter_lock_digest": payload["expected_adapter_lock_digest"],
                    "plan_hash": payload["expected_plan_hash"],
                    "run": {"plan_hash": payload["expected_plan_hash"], "run_id": "browser-run"},
                    "training_component_lock_digest": payload.get(
                        "expected_training_component_lock_digest", ""
                    ),
                }
            )
        else:
            self.send_error(404)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.evaluate(
                """const editor=document.querySelector('#trainvm-authoring');
                editor.open=true; editor.dispatchEvent(new Event('toggle'));"""
            )
            page.wait_for_function(
                "document.querySelector('#vm-editor-state').textContent.includes('valid')"
            )
            operation_select = page.locator("#vm-operation-select")
            assert operation_select.locator("option").count() == 2

            first_operation = tuple_identity(
                "test:synthetic", "trainer", "python_worker", "train",
                "test.synthetic.v1.Train",
            )
            first_component = tuple_identity("activation", "a:b", "c")
            page.select_option("#vm-operation-select", first_operation)
            page.select_option("#vm-operation-node", "train_to_boundary")
            page.locator("#vm-operation-component").fill("synthetic_first")
            page.select_option('[data-vm-operation-slot="activation"]', first_component)
            first_gain = page.locator('[data-vm-operation-field]').first
            assert first_gain.input_value() == ""
            page.click("#vm-operation-bind")
            assert "gain is required" in page.locator("#vm-editor-state").inner_text()
            first_gain.fill("1.25")
            page.click("#vm-operation-bind")
            page.wait_for_function(
                "JSON.parse(document.querySelector('#vm-json-source').value).spec.components.synthetic_first !== undefined"
            )

            second_operation = tuple_identity(
                "test", "synthetic:trainer", "python_worker", "train",
                "test.synthetic.v1.Train",
            )
            second_component = tuple_identity("activation", "a", "b:c")
            assert first_operation != second_operation
            assert first_component != second_component
            page.select_option("#vm-operation-select", second_operation)
            page.select_option("#vm-operation-node", "resume_training")
            page.locator("#vm-operation-component").fill("synthetic_second")
            page.select_option('[data-vm-operation-slot="activation"]', second_component)
            page.locator('[data-vm-operation-field]').fill("2.5")
            page.click("#vm-operation-bind")
            page.wait_for_function(
                "JSON.parse(document.querySelector('#vm-json-source').value).spec.components.synthetic_second !== undefined"
            )

            source = json.loads(page.locator("#vm-json-source").input_value())
            first_invocation = source["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
            second_invocation = source["spec"]["workflow"]["nodes"]["resume_training"]["invoke"]
            assert source["spec"]["components"]["synthetic_first"]["adapter"] == "test:synthetic"
            assert source["spec"]["components"]["synthetic_second"]["adapter"] == "test"
            assert first_invocation["training"]["model_family"] == "synthetic"
            assert first_invocation["training"]["components"]["activation"]["key"] == {
                "category": "activation", "name": "a:b", "version": "c"
            }
            assert first_invocation["training"]["components"]["activation"]["configuration"]["gain"] == 1.25
            assert second_invocation["training"]["components"]["activation"]["key"] == {
                "category": "activation", "name": "a", "version": "b:c"
            }
            assert second_invocation["training"]["components"]["activation"]["configuration"]["gain"] == 2.5

            map_key = page.evaluate(
                """[...document.querySelectorAll('[data-vm-map-key]')]
                .find((node) => JSON.parse(decodeURIComponent(node.dataset.vmMapKey)).at(-1) === 'labels')
                ?.dataset.vmMapKey || ''"""
            )
            assert map_key
            map_input = page.locator(f'[data-vm-map-key="{map_key}"]')
            map_input.evaluate(
                """node => {
                    for (let parent = node.parentElement; parent; parent = parent.parentElement) {
                        if (parent.tagName === 'DETAILS') parent.open = true;
                    }
                }"""
            )
            map_input.fill("browser_test")
            page.locator(f'[data-vm-add-map="{map_key}"]').click()
            page.wait_for_function(
                "JSON.parse(document.querySelector('#vm-json-source').value).metadata.labels.browser_test === ''"
            )

            page.locator("#vm-submit-reason").fill("browser contract")
            page.wait_for_function("!document.querySelector('#vm-submit').disabled")
            page.click("#vm-submit")
            page.wait_for_function(
                "document.querySelector('#vm-editor-state').textContent.includes('queued')"
            )
            assert Handler.submitted is not None
            submitted_source = json.loads(str(Handler.submitted["source_document"]))
            assert submitted_source["metadata"]["labels"]["browser_test"] == ""
            assert "synthetic_first" in submitted_source["spec"]["components"]
            assert "synthetic_second" in submitted_source["spec"]["components"]
            browser.close()
    finally:
        server.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
