#!/usr/bin/env python3
"""Real-browser contract test for the descriptor-driven TrainVM composer."""

from __future__ import annotations

import hashlib
import json
import copy
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError
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


def recipe_profile(name: str, version: str = "1") -> dict[str, object]:
    template = copy.deepcopy(EXAMPLE)
    training = template["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"].setdefault(
        "training", {"components": {}}
    )
    training["components"].update({
        "data": {"configuration": {
            "content_fingerprint": "sha256:" + "0" * 64,
            "manifest_path": "/datasets/train/manifest.jsonl",
        }},
        "model_loader": {"configuration": {
            "checkpoint_fingerprint": "sha256:" + "0" * 64,
            "model_path": "/models/base",
        }},
    })
    overrides = [
        {"domain": "model", "name": "model.fingerprint", "type": "string", "target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/model_loader/configuration/checkpoint_fingerprint", "required": True},
        {"domain": "model", "name": "model.path", "type": "path", "target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/model_loader/configuration/model_path", "required": True},
        {"domain": "data", "name": "data.content_fingerprint", "type": "string", "target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/data/configuration/content_fingerprint", "required": True},
        {"domain": "data", "name": "data.manifest", "type": "path", "target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/data/configuration/manifest_path", "required": True},
        {"domain": "trainability", "name": "trainability.rank", "type": "integer", "target": "/spec/parameters/target_step/value", "required": False, "minimum": 1, "maximum": 4096},
        {"domain": "optimizer", "name": "optimizer.learning_rate", "type": "number", "target": "/spec/controls/catalog/learning_rate/default", "required": False, "minimum": 1e-8, "maximum": 0.1},
        {"domain": "schedule", "name": "schedule.maximum_steps", "type": "integer", "target": "/spec/parameters/final_step/value", "required": False, "minimum": 1, "maximum": 10_000_000},
        {"domain": "precision", "name": "precision.mode", "type": "enumeration", "target": "/spec/controls/catalog/mixed_precision/default", "required": False, "values": ["bf16", "fp16"]},
        {"domain": "evaluation", "name": "evaluation.every", "type": "integer", "target": "/spec/controls/catalog/eval_every/default", "required": False, "minimum": 1, "maximum": 100_000},
        {"domain": "checkpointing", "name": "checkpointing.enabled", "type": "boolean", "target": "/spec/resources/accelerators/exclusive", "required": False},
        {"domain": "resources", "name": "resources.gpu_memory_gib", "type": "number", "target": "/spec/resources/accelerators/minimum_memory_gib", "required": False, "minimum": 24, "maximum": 192},
        {"domain": "profiling", "name": "profiling.enabled", "type": "boolean", "target": "/spec/resources/accelerators/exclusive", "required": False},
        {"domain": "controls", "name": "controls.caption_dropout", "type": "number", "target": "/spec/controls/catalog/caption_dropout/default", "required": False, "minimum": 0, "maximum": 1},
    ]
    return {
        "key": {"name": name, "version": version},
        "description": f"Generic browser fixture for {name}.",
        "template_document": template,
        "overrides": sorted(overrides, key=lambda field: field["name"]),
        "content_bindings": [
            {
                "path_target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/data/configuration/manifest_path",
                "fingerprint_target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/data/configuration/content_fingerprint",
            },
            {
                "path_target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/model_loader/configuration/model_path",
                "fingerprint_target": "/spec/workflow/nodes/train_to_boundary/invoke/training/components/model_loader/configuration/checkpoint_fingerprint",
            },
        ],
        "compatibility": [{
            "fields": ["precision.mode", "resources.gpu_memory_gib"],
            "allowed": [["bf16", 48], ["fp16", 64]],
            "description": "Only measured precision/memory tuples are allowed.",
        }],
    }


RECIPES = {
    "api_version": "trainvm.recipe-profiles/v1",
    "default_registry_path": "/opt/trainvm/recipe-profiles.json",
    "registry_digest": "sha256:" + "a" * 64,
    "registry_path": "/opt/trainvm/recipe-profiles.json",
    "recipes": [
        recipe_profile("hf_multimodal_sft"),
        recipe_profile("mageflow_flow_matching"),
        recipe_profile("rwkv_language_model"),
        recipe_profile("transformer_language_model"),
    ],
}

HTML = """<!doctype html><html><body>
<details id="trainvm-authoring">
  <button id="vm-load-example"></button><button id="vm-compile"></button>
  <button id="vm-diff"></button><input id="vm-submit-reason" />
  <button id="vm-submit" disabled></button><span id="vm-editor-state"></span>
  <span id="vm-operation-registry-state"></span><div id="vm-operation-composer"></div>
  <span id="vm-component-registry-state"></span><div id="vm-component-composer"></div>
  <span id="vm-recipe-registry-state"></span><div id="vm-recipe-composer"></div>
  <div id="vm-schema-form"></div><textarea id="vm-json-source"></textarea>
  <button id="vm-apply-json"></button><div id="vm-diagnostics"></div>
  <div id="vm-plan-preview"></div><div id="vm-plan-diff"></div>
</details><script src="/trainvm-editor.js"></script><script src="/trainvm-recipes.js"></script></body></html>"""


def compact(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tuple_identity(*parts: str) -> str:
    return json.dumps(list(parts), separators=(",", ":"))


class Handler(BaseHTTPRequestHandler):
    submitted: dict[str, object] | None = None
    authored: list[dict[str, object]] = []

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
        elif self.path == "/trainvm-recipes.js":
            body = (WEB_ROOT / "static/trainvm-recipes.js").read_bytes()
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
        elif self.path == "/api/trainvm/recipe-profiles":
            raw = compact(RECIPES)
            self.send_json({"schema": RECIPES, "schema_hash": "sha256:" + hashlib.sha256(raw).hexdigest()})
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
        elif self.path == "/api/trainvm/author-runs":
            request_document = json.loads(payload["request_document"])
            type(self).authored.append(payload)
            overrides = request_document["source"]["recipe"]["instance"]["overrides"]
            plan = {
                "experiment": request_document["source"]["recipe"]["instance"]["recipe"],
                "run_identity": request_document["source"]["recipe"]["instance"]["run_identity"],
                "overrides": overrides,
            }
            provenance = {
                f"/effective/{name}": {"kind": "instance_override", "reference": name}
                for name in overrides
            }
            plan_hash = ("2" * 64
                         if (not payload["dry_run"] and request_document["reason"] == "mismatch launch")
                         else "1" * 64)
            profile = next(
                item for item in RECIPES["recipes"]
                if item["key"] == request_document["source"]["recipe"]["instance"]["recipe"]
            )
            fields_by_target = {field["target"]: field for field in profile["overrides"]}
            derived_content_bindings = [
                {
                    **binding,
                    "path": overrides[fields_by_target[binding["path_target"]]["name"]],
                    "tree_sha256": "sha256:" + str(index + 3) * 64,
                    "provenance": "authority_measured",
                }
                for index, binding in enumerate(profile["content_bindings"])
            ]
            updates = [
                {"stage": "validating", "detail": "closed document accepted", "terminal": False, "dry_run": payload["dry_run"]},
                {"stage": "resolving", "detail": "exact recipe expanded", "terminal": False, "dry_run": payload["dry_run"]},
                {"stage": "locking_inputs", "detail": "content identities locked", "terminal": False, "dry_run": payload["dry_run"]},
                {"stage": "preflight", "detail": "passive checks passed", "terminal": False, "dry_run": payload["dry_run"]},
                *([] if payload["dry_run"] else [
                    {"stage": "provisioning", "detail": "workspace provisioned", "terminal": False, "dry_run": False},
                    {"stage": "submitting", "detail": "journal submission", "terminal": False, "dry_run": False},
                ]),
                {
                    "stage": "failed" if (not payload["dry_run"] and request_document["reason"] == "fail launch") else "complete",
                    "detail": "synthetic launch failure" if (not payload["dry_run"] and request_document["reason"] == "fail launch") else ("canonical dry-run complete" if payload["dry_run"] else "queued"),
                    "plan_hash": plan_hash, "canonical_plan_json": json.dumps(plan),
                    "recipe_expansion_json": json.dumps({
                        "effective_overrides": overrides,
                        "final_plan_hash": plan_hash,
                        "provenance": provenance,
                        "derived_content_bindings": derived_content_bindings,
                    }),
                    "preflight_receipt_json": json.dumps({
                        "passed": True,
                        "plan_hash": plan_hash,
                    }),
                    "diagnostics": ([{"severity": "error", "path": "/submission", "message": "synthetic launch failure"}]
                                    if (not payload["dry_run"] and request_document["reason"] == "fail launch") else []),
                    "run": {"run_id": "browser-recipe-run", "plan_hash": plan_hash},
                    "terminal": True, "dry_run": payload["dry_run"],
                },
            ]
            body = b"".join(json.dumps(update).encode() + b"\n" for update in updates)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
            address = f"http://127.0.0.1:{server.server_port}/"
            for attempt in range(3):
                try:
                    page.goto(address)
                    break
                except PlaywrightError as error:
                    if "ERR_NETWORK_CHANGED" not in str(error) or attempt == 2:
                        raise
            page.evaluate(
                """const editor=document.querySelector('#trainvm-authoring');
                editor.open=true; editor.dispatchEvent(new Event('toggle'));"""
            )
            page.wait_for_function(
                "document.querySelector('#vm-editor-state').textContent.includes('valid')"
            )
            page.wait_for_function(
                "document.querySelectorAll('#vm-recipe-select option').length === 4"
            )

            authored_families = [
                "hf_multimodal_sft", "mageflow_flow_matching",
                "rwkv_language_model", "transformer_language_model",
            ]
            for index, family in enumerate(authored_families):
                page.select_option("#vm-recipe-select", tuple_identity(family, "1"))
                model_path = "/" if index == 0 else f"/models/browser/{family}/checkpoint"
                manifest_path = f"/datasets/browser/{family}/train.jsonl"
                page.locator('[data-vm-recipe-field="model.path"]').fill(model_path)
                page.locator('[data-vm-recipe-field="data.manifest"]').fill(manifest_path)
                page.locator("#vm-recipe-run").fill(f"browser-{family.replace('_', '-')}")
                page.locator("#vm-recipe-author").fill("browser-agent")
                page.locator("#vm-recipe-reason").fill(f"author {family}")
                page.locator("#vm-recipe-reason").blur()
                assert not page.locator("#vm-recipe-preview").is_disabled(), (
                    family, page.locator("#vm-recipe-local-diagnostics").inner_text()
                )
                page.click("#vm-recipe-preview")
                page.wait_for_function(
                    "document.querySelector('#vm-recipe-request-state').textContent.includes('ready')"
                )
                exported = json.loads(page.locator("#vm-recipe-source").input_value())
                assert exported["source"]["recipe"]["instance"]["recipe"]["name"] == family
                assert exported["source"]["recipe"]["registry_path"] == "/opt/trainvm/recipe-profiles.json"
                assert exported["source"]["recipe"]["instance"]["overrides"]["model.path"] == model_path
                assert exported["source"]["recipe"]["instance"]["overrides"]["data.manifest"] == manifest_path
                assert "input_content" not in exported
                assert not any("fingerprint" in key for key in exported["source"]["recipe"]["instance"]["overrides"])
                assert page.locator('[data-vm-recipe-field*="fingerprint"]').count() == 0
                assert page.locator("#vm-recipe-roots").count() == 0
                preview_text = page.locator("#vm-recipe-preview-panel").inner_text()
                assert "authority_measured" in preview_text
                assert "sha256:" + "3" * 64 in preview_text
                assert model_path in preview_text
                assert manifest_path in preview_text
                assert page.locator(".vm-recipe-domain").count() == 11
                assert set(page.locator(".vm-recipe-domain .vm-editor-heading").all_inner_texts()) == {
                    "model", "data", "trainability", "optimizer", "schedule", "precision",
                    "evaluation", "checkpointing", "resources", "profiling", "controls",
                }
                assert page.locator(".vm-recipe-provenance.override").count() >= 2
                assert page.locator(".vm-recipe-provenance.template").count() >= 1
                precision = page.locator('[data-vm-recipe-field="precision.mode"]')
                assert precision.locator("option").all_inner_texts() == ["bf16"]
                if index == 0:
                    page.locator('[data-vm-recipe-present="trainability.rank"]').check()
                    page.locator('[data-vm-recipe-field="trainability.rank"]').fill("256")
                    page.locator('[data-vm-recipe-field="trainability.rank"]').blur()
                    page.click("#vm-recipe-preview")
                    page.wait_for_function(
                        "document.querySelector('#vm-recipe-request-state').textContent.includes('ready')"
                    )
                    diff_summary = page.locator("#vm-recipe-preview-panel summary", has_text="change from previous").inner_text()
                    assert not diff_summary.endswith("· 0")
                    page.locator(".vm-recipe-import").evaluate("node => node.open = true")
                    original = page.locator("#vm-recipe-source").input_value()
                    page.click("#vm-recipe-import")
                    assert json.loads(page.locator("#vm-recipe-source").input_value()) == json.loads(original)
                    page.locator(".vm-recipe-import").evaluate("node => node.open = true")
                    rejected = json.loads(original)
                    rejected["unknown"] = True
                    page.locator("#vm-recipe-source").fill(json.dumps(rejected))
                    page.click("#vm-recipe-import")
                    assert "import rejected" in page.locator("#vm-recipe-request-state").inner_text(), page.locator("#vm-recipe-request-state").inner_text()
                    rejected = json.loads(original)
                    rejected["source"]["recipe"]["instance"]["overrides"]["model.fingerprint"] = "sha256:" + "f" * 64
                    page.locator("#vm-recipe-source").fill(json.dumps(rejected))
                    page.click("#vm-recipe-import")
                    assert "authority-derived" in page.locator("#vm-recipe-request-state").inner_text()
                    page.locator("#vm-recipe-source").fill(original)
                    page.click("#vm-recipe-import")
                    page.click("#vm-recipe-preview")
                    page.wait_for_function(
                        "document.querySelector('#vm-recipe-request-state').textContent.includes('ready')"
                    )

            assert len(Handler.authored) >= 5
            page.click("#vm-recipe-launch")
            page.wait_for_function(
                "document.querySelector('#vm-recipe-request-state').textContent.includes('queued')"
            )
            assert Handler.authored[-1]["dry_run"] is False
            assert Handler.authored[-1]["expected_plan_hash"] == "1" * 64
            assert "submitting" in page.locator("#vm-recipe-preview-panel").inner_text()

            page.locator("#vm-recipe-reason").fill("mismatch launch")
            page.locator("#vm-recipe-reason").blur()
            page.click("#vm-recipe-preview")
            page.wait_for_function(
                "document.querySelector('#vm-recipe-request-state').textContent.includes('ready')"
            )
            page.click("#vm-recipe-launch")
            page.wait_for_function(
                "document.querySelector('#vm-recipe-request-state').textContent.includes('failed')"
            )
            assert "changed the previewed plan identity" in page.locator("#vm-recipe-preview-panel").inner_text()

            page.locator("#vm-recipe-reason").fill("fail launch")
            page.locator("#vm-recipe-reason").blur()
            page.click("#vm-recipe-preview")
            page.wait_for_function(
                "document.querySelector('#vm-recipe-request-state').textContent.includes('ready')"
            )
            page.click("#vm-recipe-launch")
            page.wait_for_function(
                "document.querySelector('#vm-recipe-request-state').textContent.includes('failed')"
            )
            assert "synthetic launch failure" in page.locator("#vm-recipe-preview-panel").inner_text()
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
