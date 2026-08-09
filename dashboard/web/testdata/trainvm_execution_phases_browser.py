#!/usr/bin/env python3
"""Real-browser contract test for the execution-phase diagnostics panel.

A worker execution-phase receipt carries `repeated Diagnostic diagnostics`, and
the authority admits up to 64 of them at severity info, warning or error. The
panel used to render `diagnostics[0]` inside a `vm-phase-error` element under a
`phase failure` caption, which loses every diagnostic after the first and
paints whatever sorts first as the error that failed the phase.

Only one producer exists today (`execution_phases.py`, one diagnostic, always
error), so this harness supplies the snapshot itself: it stands in for the
second producer the wire already permits. That is the whole point of the panel
contract — it must be honest about payloads it does not yet receive.

`app.js` is a single IIFE with no exports, so the render cannot be called
directly. The page is driven the way a browser drives it: a stub authority
serves `/api/trainvm/runs` and one observability snapshot, and the assertions
read the DOM that `renderVMExecutionPhases` produced.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:
    raise SystemExit(77)


WEB_ROOT = Path(__file__).resolve().parents[1]
STATIC = WEB_ROOT / "static"

RUN_ID = "vm-run-phases"
JOURNAL_ID = "journal-phases"
DIGEST = "sha256:" + "a" * 64

# The dashboard never invents a caption, so this phrase must not appear in the
# rendered panel for any diagnostic, with or without a code.
INVENTED_CAPTION = "phase failure"

RUN = {
    "run_id": RUN_ID,
    "plan_hash": "sha256:plan-phases",
    "run_revision": 1,
    "observed_state": "running",
    "current_attempt_id": "attempt-1",
    "current_node_id": "train_to_boundary",
    "last_event_sequence": 40,
}


def receipt(sequence: int, phase: str, disposition: str, diagnostics: list[dict],
            node_id: str = "train_to_boundary") -> dict:
    # The panel keys receipts by (node_id, attempt_id, phase) and keeps the
    # highest sequence per key, so two receipts of the same phase need
    # different nodes or one silently replaces the other.
    return {
        "sequence": sequence,
        "run_id": RUN_ID,
        "node_id": node_id,
        "attempt_id": "attempt-1",
        "worker_sequence": sequence,
        "phase": phase,
        "enabled": True,
        "requested_steps": 2,
        "request_digest": DIGEST,
        "disposition": disposition,
        "steps_executed": 2 if disposition == "completed" else 1,
        "state_fingerprint_before": DIGEST,
        "state_fingerprint_after": DIGEST,
        "started_at_ns": 1_000_000_000,
        "completed_at_ns": 1_000_500_000,
        "diagnostics": diagnostics,
    }


def diagnostic(severity: str, code: str, message: str) -> dict:
    return {
        "severity": severity,
        "code": code,
        "document_path": "/spec/execution/warmup",
        "message": message,
        "help": "inspect the immutable phase receipt before retrying",
    }


# The warning is deliberately ordered ahead of the error, which is exactly the
# case `diagnostics[0]` reports wrongly: it names a warning as the phase error
# and hides the error that follows it.
MIXED = receipt(
    30,
    "warmup",
    "failed",
    [
        diagnostic("warning", "execution.phase_slow", "warmup ran slower than the declared bound"),
        diagnostic("error", "execution.phase_failed", "warmup did not restore the training trajectory"),
        diagnostic("info", "execution.phase_note", "state fingerprint recomputed from the snapshot"),
    ],
)
# A diagnostic with no code. The authority rejects this today (service.cpp
# requires code and message, telemetry.go re-checks it), so it can only arrive
# from a looser future producer — and if it does, the message must stand alone
# rather than be captioned with text the dashboard wrote.
NO_CODE = receipt(
    20, "compile", "failed", [diagnostic("error", "", "compile rejected the composition")]
)
# Neither field, and no severity. The dashboard has to say something, and what
# it says must be marked as its own words and must not be styled as an error.
EMPTY = receipt(10, "compile", "failed", [{"severity": "", "code": "", "message": ""}],
                node_id="release_gpu")

SNAPSHOT = {
    "journal_id": JOURNAL_ID,
    "run": RUN,
    "observability": {},
    "after_sequence": 0,
    "target_sequence": 40,
    "next_sequence": 40,
    "replay_pending": False,
    "caught_up": True,
    "heartbeats": [],
    "metrics": [],
    "artifacts": [],
    "execution_phases": [EMPTY, NO_CODE, MIXED],
}


def index_html() -> bytes:
    """The shipped page, opened on the TrainVM panel and made hermetic.

    Serving the real `index.html` rather than a hand-built fragment keeps the
    element ids the render path touches in step with the page it ships with; a
    reconstructed fragment silently drifts.
    """
    source = (STATIC / "index.html").read_text()
    source = re.sub(r'<link[^>]*fonts\.g[^>]*>', "", source)
    source = source.replace('<details class="panel" id="trainvm-panel">',
                            '<details class="panel" id="trainvm-panel" open>')
    if 'id="trainvm-panel" open' not in source:
        raise SystemExit("index.html no longer declares the TrainVM panel as expected")
    return source.encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path, _, raw_query = self.path.partition("?")
        query = parse_qs(raw_query)
        if path == "/":
            self.send_bytes(index_html(), "text/html")
        elif path in ("/static/app.js", "/static/app.css"):
            name = path.rsplit("/", 1)[1]
            kind = "text/javascript" if name.endswith(".js") else "text/css"
            self.send_bytes((STATIC / name).read_bytes(), kind)
        elif path.startswith("/static/"):
            # Datastar, Pixi, the editor and the Pixi glue play no part in this
            # render. Stub them so the page is hermetic and no unrelated script
            # can fail the test.
            self.send_bytes(b"", "text/javascript")
        elif path == "/api/trainvm/runs":
            self.send_bytes(
                json.dumps({
                    "enabled": True,
                    "commands_enabled": False,
                    "journal_id": JOURNAL_ID,
                    "runs": [RUN],
                }).encode(), "application/json")
        elif path == f"/api/trainvm/runs/{RUN_ID}/observability":
            # The panel polls once a second and requires the snapshot's
            # after_sequence to equal the cursor it asked with, so every poll
            # after the first is served empty rather than replaying.
            after = int(query.get("after", ["0"])[0])
            snapshot = dict(SNAPSHOT, after_sequence=after)
            if after:
                snapshot["execution_phases"] = []
            self.send_bytes(json.dumps(snapshot).encode(), "application/json")
        else:
            # Every other panel (host authority, plan, controls, timeline,
            # galleries, profiles, checkpoints) handles an unavailable
            # authority on its own and does not block the telemetry path.
            self.send_error(404)


def text_of(page, selector: str) -> str:
    return page.eval_on_selector_all(
        selector, "nodes => nodes.map((node) => node.textContent).join('\\n')")


# Checks are named and reported individually rather than aborting on the first
# assert. One process renders the panel once, but a change that breaks only the
# severity colouring should be distinguishable from one that drops the second
# diagnostic — a single pass/fail cannot say which mechanism went, and a suite
# where every mutation reddens the same line is one assertion in several hats.
FAILURES: list[str] = []


def rgb_of(page, colour: str) -> str:
    """Resolve a CSS colour token the way the browser resolves it, for comparison."""
    return page.evaluate(
        """(colour) => {
            const probe = document.createElement('span');
            probe.style.color = colour;
            document.body.appendChild(probe);
            const resolved = getComputedStyle(probe).color;
            probe.remove();
            return resolved;
        }""", colour)


def check(name: str, condition: object, detail: object = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" :: {detail}" if not condition else ""))
    if not condition:
        FAILURES.append(name)


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
            page.wait_for_function(
                "document.querySelectorAll('#vm-execution-phases article').length > 0",
                timeout=15000)
            check("receipts_render",
                  page.eval_on_selector_all(
                      "#vm-execution-phases article", "nodes => nodes.length") == 3,
                  "the three receipts must each get a card")

            phases = page.eval_on_selector_all(
                "#vm-execution-phases article",
                """nodes => nodes.map((node) => ({
                    header: node.querySelector('header').textContent,
                    diagnostics: [...node.querySelectorAll('.vm-phase-diagnostic')].map((line) => ({
                        className: line.className,
                        text: line.textContent,
                        severity: (line.querySelector('.vm-phase-severity') || {}).textContent || '',
                        stated: [...line.querySelectorAll('.vm-phase-stated')]
                            .map((own) => own.textContent),
                        title: line.getAttribute('title') || '',
                        unstated: [...line.querySelectorAll('.vm-phase-unstated')]
                            .map((own) => own.textContent),
                    })),
                }))""")

            # Receipts sort newest first, so the mixed-severity receipt leads.
            mixed = phases[0]["diagnostics"] if phases else []
            no_code = phases[1]["diagnostics"] if len(phases) > 1 else []
            empty = phases[2]["diagnostics"] if len(phases) > 2 else []

            # Nothing anywhere ranks diagnostics, so the panel shows all of them
            # in the order the authority sent them. Rendering only the first
            # hides the error behind the warning that precedes it.
            check("every_diagnostic_renders", len(mixed) == 3, len(mixed))
            marks = ["warmup ran slower", "did not restore", "fingerprint recomputed"]
            check("diagnostic_order_preserved",
                  len(mixed) == 3 and
                  all(mark in mixed[index]["text"] for index, mark in enumerate(marks)),
                  [m["text"] for m in mixed])

            # Each line is coloured by its own severity: a warning ordered ahead
            # of the error must not carry the error class, and vice versa.
            check("severity_class_per_diagnostic",
                  len(mixed) == 3 and
                  "warning" in mixed[0]["className"] and "error" not in mixed[0]["className"] and
                  "error" in mixed[1]["className"] and
                  "info" in mixed[2]["className"] and "error" not in mixed[2]["className"],
                  [m["className"] for m in mixed])
            # The class only means something if the stylesheet paints it. A
            # warning that resolves to the error colour reads as the error that
            # failed the phase however it is named in the markup.
            colours = page.eval_on_selector_all(
                "#vm-execution-phases article:first-of-type .vm-phase-diagnostic",
                "nodes => nodes.map((node) => getComputedStyle(node).color)")
            error_colour = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--err').trim()")
            check("severity_colours_differ",
                  len(colours) == 3 and len(set(colours)) == 3, colours)
            check("only_the_error_line_takes_the_error_colour",
                  len(colours) == 3 and bool(error_colour) and
                  colours[1] == rgb_of(page, error_colour) and
                  colours[0] != colours[1] and colours[2] != colours[1],
                  (colours, error_colour))
            check("severity_named_in_line",
                  [m["severity"] for m in mixed] == ["warning", "error", "info"],
                  [m["severity"] for m in mixed])
            check("authority_words_marked_as_theirs",
                  bool(mixed) and all(m["stated"] and not m["unstated"] for m in mixed),
                  mixed)

            # A diagnostic with no code renders its message alone.
            check("missing_code_shows_message_alone",
                  len(no_code) == 1 and
                  "compile rejected the composition" in no_code[0]["text"] and
                  not no_code[0]["unstated"],
                  no_code)

            # Neither code nor message nor severity: the dashboard says so in
            # its own marked words, and does not style it as an error.
            check("unusable_diagnostic_reported_not_dropped",
                  len(empty) == 1 and bool(empty[0]["unstated"]), empty)
            check("unusable_diagnostic_not_styled_as_error",
                  len(empty) == 1 and "error" not in empty[0]["className"] and
                  "unstated" in empty[0]["className"],
                  empty)

            panel = text_of(page, "#vm-execution-phases")
            titles = page.eval_on_selector_all(
                "#vm-execution-phases [title]",
                "nodes => nodes.map((node) => node.getAttribute('title')).join('\\n')")
            check("no_invented_caption",
                  INVENTED_CAPTION not in panel and INVENTED_CAPTION not in titles, panel)
            # The old element name must be gone: it is what made a warning read
            # as the phase error.
            check("no_blanket_error_element",
                  page.eval_on_selector_all(
                      "#vm-execution-phases .vm-phase-error", "nodes => nodes.length") == 0)

            browser.close()
    finally:
        server.shutdown()
        thread.join()
    if FAILURES:
        raise SystemExit("execution-phase diagnostics contract failed: " + ", ".join(FAILURES))


if __name__ == "__main__":
    main()
