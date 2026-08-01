(function () {
  "use strict";

  let schema = null;
  let draft = null;
  let loaded = false;
  let compileTimer = 0;
  let compileGeneration = 0;
  let validatedDraft = null;
  let submissionIntent = null;
  let submissionBusy = false;
  let submissionStorageAvailable = true;
  let submittedDraftSource = null;
  let authorityJournalID = "";
  const submissionStorageKey = "trainvm.submission.intent.v1";

  const byID = (id) => document.getElementById(id);
  const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);

  function resolveSchema(node, value) {
    if (!node || node === true) return {};
    if (node.$ref && node.$ref.startsWith("#/$defs/") && schema) {
      const name = node.$ref.slice("#/$defs/".length);
      return { ...(schema.$defs?.[name] || {}), ...Object.fromEntries(
        Object.entries(node).filter(([key]) => key !== "$ref")) };
    }
    if (Array.isArray(node.oneOf)) {
      const branch = node.oneOf.find((candidate) => {
        const resolved = resolveSchema(candidate, value);
        return resolved.required?.every((key) => value && typeof value === "object" && key in value);
      }) || node.oneOf[0];
      return { ...node, ...resolveSchema(branch, value), oneOf: undefined };
    }
    return node;
  }

  function childSchema(parent, key, value) {
    const resolved = resolveSchema(parent, value);
    if (resolved.properties && key in resolved.properties) {
      return resolveSchema(resolved.properties[key], value);
    }
    if (resolved.additionalProperties && typeof resolved.additionalProperties === "object") {
      return resolveSchema(resolved.additionalProperties, value);
    }
    if (resolved.items) return resolveSchema(resolved.items, value);
    return {};
  }

  function typeOf(node, value) {
    const resolved = resolveSchema(node, value);
    if (resolved.type) return Array.isArray(resolved.type) ? resolved.type[0] : resolved.type;
    if ("const" in resolved) return typeof resolved.const;
    if (resolved.enum?.length) return typeof resolved.enum[0];
    if (Array.isArray(value)) return "array";
    if (value !== null && typeof value === "object") return "object";
    return typeof value;
  }

  function encodedPath(path) { return encodeURIComponent(JSON.stringify(path)); }

  function renderPrimitive(label, value, node, path, required) {
    const resolved = resolveSchema(node, value);
    const kind = typeOf(resolved, value);
    const pathValue = encodedPath(path);
    const requiredClass = required ? " vm-required" : "";
    const title = escapeHTML(resolved.description || resolved.pattern || "");
    let control;
    if (Array.isArray(resolved.enum)) {
      control = `<select data-vm-edit="${pathValue}" data-vm-kind="${escapeHTML(kind)}">` +
        resolved.enum.map((option) => `<option${option === value ? " selected" : ""}>${escapeHTML(option)}</option>`).join("") +
        "</select>";
    } else if ("const" in resolved) {
      control = `<input value="${escapeHTML(resolved.const)}" disabled />`;
    } else if (kind === "boolean") {
      control = `<input type="checkbox" data-vm-edit="${pathValue}" data-vm-kind="boolean"${value ? " checked" : ""} />`;
    } else if (kind === "integer" || kind === "number") {
      const step = kind === "integer" ? "1" : "any";
      const minimum = resolved.minimum === undefined ? "" : ` min="${escapeHTML(resolved.minimum)}"`;
      const maximum = resolved.maximum === undefined ? "" : ` max="${escapeHTML(resolved.maximum)}"`;
      control = `<input type="number" step="${step}"${minimum}${maximum} value="${escapeHTML(value)}" data-vm-edit="${pathValue}" data-vm-kind="${kind}" />`;
    } else {
      control = `<input type="text" value="${escapeHTML(value)}" data-vm-edit="${pathValue}" data-vm-kind="string" />`;
    }
    return `<div class="vm-schema-field" title="${title}"><label class="${requiredClass}">${escapeHTML(label)}</label>${control}</div>`;
  }

  function renderNode(label, value, node, path, depth, required = false) {
    const resolved = resolveSchema(node, value);
    const kind = typeOf(resolved, value);
    if (kind !== "object" && kind !== "array") {
      return renderPrimitive(label, value, resolved, path, required);
    }
    const entries = kind === "array"
      ? (Array.isArray(value) ? value.map((item, index) => [String(index), item]) : [])
      : Object.entries(value && typeof value === "object" ? value : {});
    const requiredFields = new Set(resolved.required || []);
    const children = entries.map(([key, child]) => renderNode(
      kind === "array" ? `[${key}]` : key,
      child,
      kind === "array" ? resolveSchema(resolved.items, child) : childSchema(resolved, key, child),
      [...path, kind === "array" ? Number(key) : key], depth + 1, requiredFields.has(key))).join("");
    const open = depth < 2 ? " open" : "";
    const count = entries.length;
    return `<details class="vm-schema-group"${open}><summary>${escapeHTML(label)}<span class="vm-kind">${kind} · ${count}</span></summary>${children || '<div class="empty">empty</div>'}</details>`;
  }

  function renderForm() {
    const target = byID("vm-schema-form");
    if (!target || !schema || !draft) return;
    target.innerHTML = renderNode("experiment", draft, schema, [], 0, true);
    const source = byID("vm-json-source");
    if (source) source.value = JSON.stringify(draft, null, 2);
  }

  function setAtPath(path, value) {
    if (!path.length) { draft = value; return; }
    let current = draft;
    for (let index = 0; index < path.length - 1; index++) current = current[path[index]];
    current[path[path.length - 1]] = value;
  }

  function scheduleCompile() {
    compileGeneration += 1;
    validatedDraft = null;
    submittedDraftSource = null;
    updateSubmitState();
    clearTimeout(compileTimer);
    compileTimer = setTimeout(compileDraft, 450);
    const state = byID("vm-editor-state");
    if (state) state.textContent = "draft changed · preview pending";
  }

  function restoreSubmissionIntent() {
    try {
      const stored = sessionStorage.getItem(submissionStorageKey);
      submissionIntent = stored ? JSON.parse(stored) : null;
      if (!submissionIntent || typeof submissionIntent.body !== "string") {
        submissionIntent = null;
        return;
      }
      const payload = JSON.parse(submissionIntent.body);
      if (!payload.expected_journal_id || !payload.expected_plan_hash ||
          !payload.expected_adapter_lock_digest ||
          typeof payload.source_document !== "string") {
        submissionIntent = null;
        return;
      }
      submissionIntent.source ||= payload.source_document;
      submissionIntent.journalID ||= payload.expected_journal_id;
      submissionIntent.planHash ||= payload.expected_plan_hash;
      submissionIntent.adapterLockDigest ||= payload.expected_adapter_lock_digest;
      submissionIntent.trainingComponentLockDigest ||=
        payload.expected_training_component_lock_digest || "";
      const reason = byID("vm-submit-reason");
      if (reason) reason.value = payload.reason || "";
    } catch (_) {
      submissionIntent = null;
      submissionStorageAvailable = false;
    }
  }

  function persistSubmissionIntent() {
    try {
      if (submissionIntent) sessionStorage.setItem(submissionStorageKey, JSON.stringify(submissionIntent));
      else sessionStorage.removeItem(submissionStorageKey);
      submissionStorageAvailable = true;
      return true;
    } catch (_) {
      submissionStorageAvailable = false;
      return false;
    }
  }

  function lockEditor(locked) {
    document.querySelectorAll("#vm-schema-form input, #vm-schema-form select, #vm-json-source, #vm-apply-json, #vm-load-example, #vm-compile")
      .forEach((control) => {
        if (locked && !control.disabled) {
          control.dataset.vmIntentDisabled = "true";
          control.disabled = true;
        } else if (!locked && control.dataset.vmIntentDisabled === "true") {
          delete control.dataset.vmIntentDisabled;
          control.disabled = false;
        }
      });
  }

  function updateSubmitState(message = "") {
    const button = byID("vm-submit");
    const reason = byID("vm-submit-reason");
    if (!button) return;
    const retry = Boolean(submissionIntent);
    const alreadySubmitted = Boolean(validatedDraft && submittedDraftSource === validatedDraft.source);
    button.disabled = submissionBusy || (!retry &&
      (!validatedDraft || !reason?.value.trim() || !authorityJournalID || alreadySubmitted));
    button.textContent = submissionBusy ? "submitting…" :
      (retry ? "retry exact submission" : "create queued run");
    if (reason) reason.disabled = retry || submissionBusy;
    lockEditor(retry || submissionBusy);
    if (message) {
      const state = byID("vm-editor-state");
      if (state) state.textContent = message;
    }
  }

  function renderDiagnostics(result) {
    const target = byID("vm-diagnostics");
    if (!target) return;
    const diagnostics = Array.isArray(result.diagnostics) ? result.diagnostics : [];
    if (!diagnostics.length) {
      target.innerHTML = `<div class="vm-diag ${result.valid ? "" : "error"}">${result.valid ? "native compiler accepted the draft" : "draft rejected without diagnostics"}</div>`;
      return;
    }
    target.innerHTML = diagnostics.map((item) =>
      `<div class="vm-diag ${escapeHTML(item.severity || "error")}"><code>${escapeHTML(item.path || "/")}</code><strong>${escapeHTML(item.code || "diagnostic")}</strong> · ${escapeHTML(item.message || "")}</div>`).join("");
  }

  function renderPlan(result) {
    const target = byID("vm-plan-preview");
    if (!target) return;
    if (!result.valid) { target.textContent = ""; return; }
    const fact = (label, value, title = value) => `<div class="vm-fact"><span>${escapeHTML(label)}</span><strong title="${escapeHTML(title)}">${escapeHTML(value)}</strong></div>`;
    target.innerHTML = `<div class="vm-plan-summary">${fact("plan hash", String(result.plan_hash || "").slice(0, 12), result.plan_hash)}${fact("entrypoint", result.entrypoint)}${fact("nodes", result.node_count)}${fact("artifacts", result.artifact_count)}${fact("controls", result.control_count)}${fact("experiment", result.experiment)}</div>` +
      `<details class="vm-canonical"><summary>canonical reflected plan</summary><pre>${escapeHTML(JSON.stringify(result.canonical_plan, null, 2))}</pre></details>`;
  }

  async function compileDraft() {
    if (!draft) return;
    clearTimeout(compileTimer);
    validatedDraft = null;
    updateSubmitState();
    const generation = ++compileGeneration;
    const source = JSON.stringify(draft);
    const state = byID("vm-editor-state");
    if (state) state.textContent = "native compile…";
    try {
      const response = await fetch("/api/trainvm/compile", {
        method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
        body: source,
      });
      const text = await response.text();
      if (generation !== compileGeneration) return;
      if (!response.ok) throw new Error(text.trim() || `HTTP ${response.status}`);
      const result = JSON.parse(text);
      renderDiagnostics(result);
      renderPlan(result);
      validatedDraft = result.valid && result.plan_hash && result.adapter_lock_digest ? {
        generation, source, planHash: String(result.plan_hash),
        adapterLockDigest: String(result.adapter_lock_digest),
        trainingComponentLockDigest: String(result.training_component_lock_digest || ""),
      } : null;
      updateSubmitState();
      if (state) state.textContent = result.valid ? `valid · ${String(result.plan_hash || "").slice(0, 12)}` : "native compiler rejected draft";
    } catch (error) {
      if (generation !== compileGeneration) return;
      if (state) state.textContent = "preview unavailable";
      const diagnostics = byID("vm-diagnostics");
      if (diagnostics) diagnostics.innerHTML = `<div class="vm-diag error">${escapeHTML(error.message)}</div>`;
    }
  }

  async function submitDraft() {
    if (submissionBusy) return;
    const reason = byID("vm-submit-reason");
    if (!submissionIntent) {
      if (!validatedDraft || !reason?.value.trim() || !authorityJournalID ||
          submittedDraftSource === validatedDraft.source) return;
      submissionIntent = {
        body: JSON.stringify({
          source_document: validatedDraft.source,
          source_format: "json",
          create_run: true,
          idempotency_key: crypto.randomUUID(),
          expected_journal_id: authorityJournalID,
          expected_plan_hash: validatedDraft.planHash,
          expected_adapter_lock_digest: validatedDraft.adapterLockDigest,
          expected_training_component_lock_digest:
            validatedDraft.trainingComponentLockDigest,
          reason: reason.value.trim(),
        }),
        source: validatedDraft.source,
        journalID: authorityJournalID,
        planHash: validatedDraft.planHash,
        adapterLockDigest: validatedDraft.adapterLockDigest,
        trainingComponentLockDigest:
          validatedDraft.trainingComponentLockDigest,
      };
      persistSubmissionIntent();
    }
    const intent = submissionIntent;
    submissionBusy = true;
    updateSubmitState("submitting frozen draft to native authority…");
    let finalMessage = "";
    try {
      const response = await fetch("/api/trainvm/experiments", {
        method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
        body: intent.body,
      });
      const text = await response.text();
      let result = null;
      try { result = JSON.parse(text); } catch (_) { /* HTTP text is shown below. */ }
      if (!response.ok) {
        if (response.status === 408 || response.status >= 500) {
          finalMessage = submissionStorageAvailable ?
            "outcome unknown · retry exact submission" :
            "outcome unknown · retry exact submission · keep this tab open";
          return;
        }
        submissionIntent = null;
        persistSubmissionIntent();
        validatedDraft = null;
        submittedDraftSource = null;
        if (result?.diagnostics) {
          renderDiagnostics({ valid: false, diagnostics: result.diagnostics });
          renderPlan({ valid: false });
        }
        finalMessage = `submission rejected · ${result?.diagnostics?.[0]?.message ||
          text.trim() || `HTTP ${response.status}`}`;
        return;
      }
      const runID = result?.run?.run_id || "";
      if (!runID || result.plan_hash !== intent.planHash ||
          result.run?.plan_hash !== intent.planHash ||
          result.adapter_lock_digest !== intent.adapterLockDigest ||
          String(result.training_component_lock_digest || "") !==
            intent.trainingComponentLockDigest) {
        finalMessage = "authority returned an inconsistent result · retry exact submission";
        return;
      }
      submissionIntent = null;
      persistSubmissionIntent();
      submittedDraftSource = intent.source;
      if (reason) reason.value = "";
      finalMessage = `queued · ${runID}`;
      window.dispatchEvent(new CustomEvent("trainvm-refresh", { detail: { runID } }));
    } catch (_) {
      finalMessage = submissionStorageAvailable ?
        "outcome unknown · retry exact submission" :
        "outcome unknown · retry exact submission · keep this tab open";
    } finally {
      submissionBusy = false;
      updateSubmitState(finalMessage);
    }
  }

  async function loadAuthoring(force = false) {
    if (loaded && !force) return;
    clearTimeout(compileTimer);
    compileGeneration += 1;
    validatedDraft = null;
    submittedDraftSource = null;
    updateSubmitState();
    const state = byID("vm-editor-state");
    if (state) state.textContent = "loading schema and reference…";
    try {
      const [schemaResponse, exampleResponse, runsResponse] = await Promise.all([
        fetch("/api/trainvm/schema", { cache: "no-store" }),
        fetch("/api/trainvm/example", { cache: "no-store" }),
        fetch("/api/trainvm/runs", { cache: "no-store" }),
      ]);
      if (!schemaResponse.ok || !exampleResponse.ok || !runsResponse.ok) {
        throw new Error("TrainVM authoring endpoints unavailable");
      }
      schema = await schemaResponse.json();
      draft = await exampleResponse.json();
      const runs = await runsResponse.json();
      authorityJournalID = String(runs.journal_id || "");
      if (!authorityJournalID) throw new Error("TrainVM journal identity unavailable");
      loaded = true;
      renderForm();
      updateSubmitState();
      await compileDraft();
    } catch (error) {
      if (state) state.textContent = error.message;
    }
  }

  const authoring = byID("trainvm-authoring");
  restoreSubmissionIntent();
  updateSubmitState();
  if (authoring) authoring.addEventListener("toggle", () => { if (authoring.open) loadAuthoring(); });
  byID("vm-load-example")?.addEventListener("click", () => loadAuthoring(true));
  byID("vm-compile")?.addEventListener("click", compileDraft);
  byID("vm-submit")?.addEventListener("click", submitDraft);
  byID("vm-submit-reason")?.addEventListener("input", () => updateSubmitState());
  byID("vm-json-source")?.addEventListener("input", () => {
    clearTimeout(compileTimer);
    compileGeneration += 1;
    validatedDraft = null;
    submittedDraftSource = null;
    updateSubmitState("raw source changed · apply JSON before compiling");
  });
  byID("vm-apply-json")?.addEventListener("click", () => {
    const source = byID("vm-json-source");
    try {
      draft = JSON.parse(source.value);
      renderForm();
      scheduleCompile();
    } catch (error) {
      clearTimeout(compileTimer);
      compileGeneration += 1;
      validatedDraft = null;
      submittedDraftSource = null;
      updateSubmitState();
      byID("vm-editor-state").textContent = `invalid JSON · ${error.message}`;
    }
  });
  byID("vm-schema-form")?.addEventListener("change", (event) => {
    const input = event.target.closest("[data-vm-edit]");
    if (!input || !draft) return;
    const path = JSON.parse(decodeURIComponent(input.dataset.vmEdit));
    let value = input.value;
    if (input.dataset.vmKind === "boolean") value = input.checked;
    else if (input.dataset.vmKind === "integer") value = Number.parseInt(value, 10);
    else if (input.dataset.vmKind === "number") value = Number(value);
    setAtPath(path, value);
    const source = byID("vm-json-source");
    if (source) source.value = JSON.stringify(draft, null, 2);
    scheduleCompile();
  });
})();
