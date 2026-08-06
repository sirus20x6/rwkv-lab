(function () {
  "use strict";

  const API_VERSION = "trainvm.author-run/v1";
  const INSTANCE_VERSION = "trainvm.recipe-instance/v1";
  const PROFILE_ENDPOINT = "/api/trainvm/recipe-profiles";
  const AUTHOR_ENDPOINT = "/api/trainvm/author-runs";
  const RUN_IDENTITY = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
  const byID = (id) => document.getElementById(id);
  const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);

  let registry = null;
  let registryHash = "";
  let registryPath = "";
  let selectedIdentity = "";
  let instance = null;
  let author = "";
  let reason = "";
  let inputRoots = [];
  let preview = null;
  let requestView = null;
  let previousPlan = null;
  let requestBusy = false;
  let loaded = false;
  let requestMessage = "preview required";

  function identityOf(profile) {
    return JSON.stringify([profile?.key?.name || "", profile?.key?.version || ""]);
  }

  function labelOf(profile) {
    return `${profile?.key?.name || "unknown"}@${profile?.key?.version || "unknown"}`;
  }

  function selectedProfile() {
    const profiles = registry?.recipes || [];
    return profiles.find((profile) => identityOf(profile) === selectedIdentity) || profiles[0] || null;
  }

  function decodePointerPart(value) {
    return value.replace(/~1/g, "/").replace(/~0/g, "~");
  }

  function valueAtPointer(documentValue, pointer) {
    if (pointer === "") return documentValue;
    if (!pointer?.startsWith("/")) return undefined;
    let value = documentValue;
    for (const raw of pointer.slice(1).split("/")) {
      const part = decodePointerPart(raw);
      if (value === null || typeof value !== "object" || !(part in value)) return undefined;
      value = value[part];
    }
    return value;
  }

  function clone(value) {
    return value === undefined ? undefined : structuredClone(value);
  }

  function equal(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
  }

  function effectiveValue(profile, field, overrides = instance?.overrides || {}) {
    if (Object.prototype.hasOwnProperty.call(overrides, field.name)) return overrides[field.name];
    return clone(valueAtPointer(profile?.template_document, field.target));
  }

  function newInstance(profile) {
    const overrides = {};
    for (const field of profile?.overrides || []) {
      if (field.required) overrides[field.name] = effectiveValue(profile, field, {});
    }
    return {
      api_version: INSTANCE_VERSION,
      recipe: { name: profile.key.name, version: profile.key.version },
      run_identity: "",
      overrides,
    };
  }

  function ensureSelection() {
    const profiles = registry?.recipes || [];
    if (!profiles.some((profile) => identityOf(profile) === selectedIdentity)) {
      selectedIdentity = identityOf(profiles[0]);
    }
    const profile = selectedProfile();
    if (!profile) {
      instance = null;
      return;
    }
    if (instance?.recipe?.name !== profile.key.name ||
        instance?.recipe?.version !== profile.key.version) {
      instance = newInstance(profile);
      invalidatePreview();
    }
  }

  function compatibilityDiagnostics(profile, overrides = instance?.overrides || {}) {
    const fields = new Map((profile?.overrides || []).map((field) => [field.name, field]));
    const diagnostics = [];
    for (const rule of profile?.compatibility || []) {
      const tuple = (rule.fields || []).map((name) => effectiveValue(profile, fields.get(name), overrides));
      if (!(rule.allowed || []).some((allowed) => equal(allowed, tuple))) {
        diagnostics.push({
          path: `/source/recipe/instance/overrides/${rule.fields?.join("+") || "compatibility"}`,
          message: rule.description || `unsupported combination: ${(rule.fields || []).join(", ")}`,
          help: `Allowed tuples: ${JSON.stringify(rule.allowed || [])}`,
        });
      }
    }
    return diagnostics;
  }

  function compatibleEnumerationValues(profile, field) {
    const rules = (profile?.compatibility || []).filter((rule) => (rule.fields || []).includes(field.name));
    if (!rules.length) return field.values || [];
    const fields = new Map((profile.overrides || []).map((candidate) => [candidate.name, candidate]));
    return (field.values || []).filter((candidate) => rules.every((rule) => {
      const index = rule.fields.indexOf(field.name);
      return (rule.allowed || []).some((tuple) => (rule.fields || []).every((name, position) => {
        if (position === index) return equal(tuple[position], candidate);
        return equal(tuple[position], effectiveValue(profile, fields.get(name)));
      }));
    }));
  }

  function localDiagnostics() {
    const profile = selectedProfile();
    const diagnostics = [];
    if (!profile || !instance) return [{ path: "/source/recipe", message: "Select an exact recipe profile." }];
    if (!registryPath.startsWith("/")) {
      diagnostics.push({ path: "/source/recipe/registry_path", message: "A canonical absolute recipe registry path is required." });
    }
    if (!RUN_IDENTITY.test(instance.run_identity || "") || instance.run_identity.length > 128) {
      diagnostics.push({ path: "/source/recipe/instance/run_identity", message: "Run identity must be a bounded lowercase symbolic identity." });
    }
    if (!author.trim()) diagnostics.push({ path: "/author", message: "Author is required." });
    if (!reason.trim()) diagnostics.push({ path: "/reason", message: "Reason is required." });
    for (const field of profile.overrides || []) {
      const present = Object.prototype.hasOwnProperty.call(instance.overrides || {}, field.name);
      const value = instance.overrides?.[field.name];
      if (field.required && !present) {
        diagnostics.push({ path: `/overrides/${field.name}`, message: `${field.name} is required.` });
        continue;
      }
      if (!present) continue;
      if (field.type === "boolean" && typeof value !== "boolean" ||
          field.type === "integer" && !Number.isInteger(value) ||
          field.type === "number" && (typeof value !== "number" || !Number.isFinite(value)) ||
          (field.type === "string" || field.type === "path" || field.type === "enumeration") && typeof value !== "string") {
        diagnostics.push({ path: `/overrides/${field.name}`, message: `${field.name} has the wrong type.` });
      } else if (field.minimum !== undefined && value < Number(field.minimum) ||
                 field.maximum !== undefined && value > Number(field.maximum)) {
        diagnostics.push({ path: `/overrides/${field.name}`, message: `${field.name} is outside its declared bounds.` });
      } else if (field.type === "enumeration" && !(field.values || []).some((candidate) => equal(candidate, value))) {
        diagnostics.push({ path: `/overrides/${field.name}`, message: `${field.name} is not an allowed value.` });
      }
    }
    diagnostics.push(...compatibilityDiagnostics(profile));
    return diagnostics;
  }

  function authorDocument() {
    const documentValue = {
      api_version: API_VERSION,
      source: {
        recipe: {
          registry_path: registryPath,
          instance: clone(instance),
        },
      },
      author: author.trim(),
      reason: reason.trim(),
    };
    if (inputRoots.length) {
      documentValue.input_content = {
        api_version: "trainvm.input-content-root-set/v1",
        paths: [...inputRoots],
      };
    }
    return documentValue;
  }

  function compactSource() {
    return JSON.stringify(authorDocument());
  }

  function invalidatePreview() {
    preview = null;
    requestView = null;
    requestMessage = "preview required";
    const launch = byID("vm-recipe-launch");
    if (launch) launch.disabled = true;
    renderPreview();
  }

  function fieldControl(profile, field) {
    const present = Object.prototype.hasOwnProperty.call(instance.overrides || {}, field.name);
    const value = effectiveValue(profile, field);
    const required = field.required ? " vm-required" : "";
    const disabled = !present ? " disabled" : "";
    const token = escapeHTML(field.name);
    let control;
    if (field.type === "enumeration") {
      const compatible = compatibleEnumerationValues(profile, field);
      control = `<select data-vm-recipe-field="${token}" data-vm-kind="string"${disabled}>` +
        compatible.map((option) => `<option value="${escapeHTML(option)}"${equal(option, value) ? " selected" : ""}>${escapeHTML(option)}</option>`).join("") + "</select>";
    } else if (field.type === "boolean") {
      control = `<select data-vm-recipe-field="${token}" data-vm-kind="boolean"${disabled}>` +
        `<option value="true"${value === true ? " selected" : ""}>true</option>` +
        `<option value="false"${value === false ? " selected" : ""}>false</option></select>`;
    } else if (field.type === "integer" || field.type === "number") {
      const minimum = field.minimum === undefined ? "" : ` min="${escapeHTML(field.minimum)}"`;
      const maximum = field.maximum === undefined ? "" : ` max="${escapeHTML(field.maximum)}"`;
      control = `<input type="number" step="${field.type === "integer" ? "1" : "any"}"${minimum}${maximum} value="${escapeHTML(value)}" data-vm-recipe-field="${token}" data-vm-kind="${field.type}"${disabled} />`;
    } else {
      control = `<input type="text" maxlength="4096" value="${escapeHTML(value)}" data-vm-recipe-field="${token}" data-vm-kind="string"${disabled} />`;
    }
    const include = field.required ? "" : `<input type="checkbox" data-vm-recipe-present="${token}"${present ? " checked" : ""} aria-label="Override ${token}" />`;
    const source = present ? "instance override" : "recipe template";
    return `<div class="vm-recipe-field" title="${escapeHTML(field.description || field.target || "")}">` +
      `<label class="${required}">${escapeHTML(field.name)}</label><div class="vm-component-value">${include}${control}</div>` +
      `<span class="vm-recipe-provenance ${present ? "override" : "template"}">${source}</span></div>`;
  }

  function renderDiagnostics(diagnostics) {
    if (!diagnostics.length) return '<div class="vm-diag ok">locally valid · canonical dry-run required</div>';
    return diagnostics.map((item) => {
      const severity = String(item.severity || "error").toLowerCase();
      const kind = severity.includes("error") ? "error" : severity.includes("warn") ? "warning" : "ok";
      return `<div class="vm-diag ${kind}"><code>${escapeHTML(item.path || "/")}</code> ${escapeHTML(item.message || "invalid")}${item.help ? `<small>${escapeHTML(item.help)}</small>` : ""}</div>`;
    }).join("");
  }

  function refreshLocalState() {
    const diagnostics = localDiagnostics();
    const target = byID("vm-recipe-local-diagnostics");
    if (target) target.innerHTML = renderDiagnostics(diagnostics);
    const previewButton = byID("vm-recipe-preview");
    if (previewButton) previewButton.disabled = requestBusy || diagnostics.length > 0;
  }

  function renderComposer() {
    const target = byID("vm-recipe-composer");
    const state = byID("vm-recipe-registry-state");
    if (!target) return;
    const profiles = registry?.recipes || [];
    if (state) state.textContent = registry ? `${profiles.length} exact recipes · ${registryHash.slice(0, 18)}…` : "authority registry unavailable";
    if (!profiles.length) {
      target.innerHTML = '<div class="empty">no exact recipe profiles are authorized by this daemon</div>';
      return;
    }
    ensureSelection();
    const profile = selectedProfile();
    const options = profiles.map((candidate) => `<option value="${escapeHTML(identityOf(candidate))}"${identityOf(candidate) === selectedIdentity ? " selected" : ""}>${escapeHTML(labelOf(candidate))}</option>`).join("");
    const domains = new Map();
    for (const field of profile.overrides || []) {
      if (!domains.has(field.domain)) domains.set(field.domain, []);
      domains.get(field.domain).push(field);
    }
    const groups = [...domains].map(([domain, fields]) => `<section class="vm-recipe-domain"><div class="vm-editor-heading">${escapeHTML(domain)}</div><div class="vm-recipe-fields">${fields.map((field) => fieldControl(profile, field)).join("")}</div></section>`).join("");
    const local = localDiagnostics();
    target.innerHTML = `<div class="vm-recipe-head">` +
      `<div class="vm-component-input"><label>exact recipe</label><select id="vm-recipe-select">${options}</select></div>` +
      `<div class="vm-component-input"><label>run identity</label><input id="vm-recipe-run" maxlength="128" value="${escapeHTML(instance.run_identity)}" /></div>` +
      `<div class="vm-component-input"><label>author</label><input id="vm-recipe-author" maxlength="192" value="${escapeHTML(author)}" /></div>` +
      `<div class="vm-component-input"><label>reason</label><input id="vm-recipe-reason" maxlength="2048" value="${escapeHTML(reason)}" /></div></div>` +
      `<div class="vm-recipe-authority"><div class="vm-component-input"><label>authority recipe registry</label><input id="vm-recipe-path" maxlength="4096" value="${escapeHTML(registryPath)}" readonly /></div>` +
      `<div class="vm-component-input"><label>content-lock roots · one absolute path per line</label><textarea id="vm-recipe-roots" rows="2">${escapeHTML(inputRoots.join("\n"))}</textarea></div></div>` +
      `<div class="vm-component-meta vm-recipe-description"><strong>${escapeHTML(labelOf(profile))}</strong>${escapeHTML(profile.description || "Exact-versioned authority-owned training recipe.")}</div>` +
      `<div class="vm-recipe-domains">${groups || '<div class="empty">this recipe exposes no overrides</div>'}</div>` +
      `<div id="vm-recipe-local-diagnostics" class="vm-diagnostics">${renderDiagnostics(local)}</div>` +
      `<div class="vm-recipe-actions"><button class="btn sm" id="vm-recipe-preview" type="button"${local.length || requestBusy ? " disabled" : ""}>canonical dry-run</button>` +
      `<button class="btn sm" id="vm-recipe-launch" type="button"${!preview || preview.source !== compactSource() || requestBusy ? " disabled" : ""}>launch exact preview</button>` +
      `<button class="btn sm" id="vm-recipe-copy" type="button">copy compact document</button>` +
      `<span id="vm-recipe-request-state" class="sub">${escapeHTML(requestMessage)}</span></div>` +
      `<details class="vm-raw-editor vm-recipe-import"><summary>compact import / export</summary><textarea id="vm-recipe-source" spellcheck="false" aria-label="Compact AuthorRun source">${escapeHTML(JSON.stringify(authorDocument(), null, 2))}</textarea><button class="btn sm" id="vm-recipe-import" type="button">import exact document</button></details>` +
      `<div id="vm-recipe-preview-panel" class="vm-recipe-preview"></div>`;
    renderPreview();
  }

  function flatten(documentValue, path = "", output = new Map()) {
    if (documentValue !== null && typeof documentValue === "object") {
      const entries = Array.isArray(documentValue) ? documentValue.map((value, index) => [String(index), value]) : Object.entries(documentValue);
      for (const [key, value] of entries) flatten(value, `${path}/${String(key).replace(/~/g, "~0").replace(/\//g, "~1")}`, output);
    } else output.set(path || "/", documentValue);
    return output;
  }

  function planDifference(before, after) {
    if (!before) return [];
    const left = flatten(before), right = flatten(after);
    const paths = [...new Set([...left.keys(), ...right.keys()])].sort();
    return paths.filter((path) => !left.has(path) || !right.has(path) || !equal(left.get(path), right.get(path))).map((path) => ({
      path, before: left.has(path) ? left.get(path) : undefined, after: right.has(path) ? right.get(path) : undefined,
    }));
  }

  function renderPreview() {
    const target = byID("vm-recipe-preview-panel");
    if (!target) return;
    const shown = requestView || preview;
    if (!shown) {
      target.innerHTML = '<div class="empty">canonical dry-run will show the expanded plan, provenance, diagnostics, preflight, and changes from the previous preview</div>';
      return;
    }
    const expansion = shown.recipeExpansion || {};
    const provenance = expansion.provenance || shown.provenance || {};
    const provenanceRows = Object.entries(provenance).map(([path, source]) => `<div class="vm-recipe-provenance-row"><code>${escapeHTML(path)}</code><span>${escapeHTML(source?.kind || "authority")}</span><code>${escapeHTML(source?.reference || "")}</code></div>`).join("");
    const diffs = shown.diff || [];
    const diffRows = diffs.map((change) => `<div class="vm-diff-op"><code>change</code><code>${escapeHTML(change.path)}</code><code>${escapeHTML(JSON.stringify(change.after))}</code></div>`).join("");
    const stages = (shown.updates || []).map((update) => `<div class="vm-recipe-stage ${update.terminal ? "terminal" : ""}"><code>${escapeHTML(update.stage || "update")}</code><span>${escapeHTML(update.detail || "")}</span></div>`).join("");
    target.innerHTML = `<div class="vm-recipe-preview-grid"><section><div class="vm-editor-heading">authority progress</div>${stages || '<div class="empty">waiting for authority</div>'}</section>` +
      `<section><div class="vm-editor-heading">authority diagnostics</div>${renderDiagnostics(shown.diagnostics || [])}</section></div>` +
      `<details class="vm-canonical" open><summary>canonical expanded plan · ${escapeHTML(shown.planHash || "pending")}</summary><pre>${escapeHTML(JSON.stringify(shown.plan, null, 2))}</pre></details>` +
      `<details class="vm-canonical"><summary>effective-value provenance · ${Object.keys(provenance).length}</summary>${provenanceRows || '<div class="empty">field-level template/override provenance is shown above; authority expansion provenance was not emitted</div>'}</details>` +
      `<details class="vm-canonical"><summary>change from previous canonical preview · ${diffs.length}</summary>${diffRows || '<div class="empty">first preview or no canonical changes</div>'}</details>` +
      `${shown.preflight ? `<details class="vm-canonical"><summary>passive preflight receipt</summary><pre>${escapeHTML(JSON.stringify(shown.preflight, null, 2))}</pre></details>` : ""}`;
  }

  function setRequestState(message) {
    requestMessage = message;
    const state = byID("vm-recipe-request-state");
    if (state) state.textContent = message;
  }

  function normalizeUpdate(value) {
    const parseJSON = (candidate) => {
      if (!candidate) return null;
      if (typeof candidate === "object") return candidate;
      try { return JSON.parse(candidate); } catch (_) { return null; }
    };
    return {
      ...value,
      stage: value.stage || value.stage_name || value.stageName || "update",
      planHash: value.plan_hash || value.planHash || "",
      plan: parseJSON(value.canonical_plan_json || value.canonicalPlanJson || value.canonical_plan),
      preflight: parseJSON(value.preflight_receipt_json || value.preflightReceiptJson || value.preflight_receipt),
      recipeExpansion: parseJSON(value.recipe_expansion_json || value.recipeExpansionJson || value.recipe_expansion),
      diagnostics: Array.isArray(value.diagnostics) ? value.diagnostics : [],
      terminal: Boolean(value.terminal),
      dryRun: Boolean(value.dry_run ?? value.dryRun),
    };
  }

  async function readUpdates(response, onUpdate) {
    if (!response.body) throw new Error("authority stream has no body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) if (line.trim()) onUpdate(normalizeUpdate(JSON.parse(line)));
      if (done) break;
    }
    if (buffer.trim()) onUpdate(normalizeUpdate(JSON.parse(buffer)));
  }

  async function authorRun(dryRun) {
    if (requestBusy || localDiagnostics().length) return;
    const source = compactSource();
    if (!dryRun && (!preview || preview.source !== source)) return;
    requestBusy = true;
    const current = { source, planHash: "", plan: null, preflight: null, recipeExpansion: null, diagnostics: [], updates: [], diff: [] };
    requestView = current;
    if (dryRun) preview = current;
    renderComposer();
    setRequestState(dryRun ? "canonical dry-run…" : "launching frozen preview…");
    let terminal = null;
    try {
      const response = await fetch(AUTHOR_ENDPOINT, {
        method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_document: source, source_format: "json", dry_run: dryRun }),
      });
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      if (!String(response.headers.get("Content-Type") || "").toLowerCase().startsWith("application/x-ndjson")) {
        throw new Error("authority response is not an NDJSON update stream");
      }
      await readUpdates(response, (update) => {
        terminal = update.terminal ? update : terminal;
        current.updates.push(update);
        current.diagnostics.push(...update.diagnostics);
        if (update.planHash) current.planHash = update.planHash;
        if (update.plan) current.plan = update.plan;
        if (update.preflight) current.preflight = update.preflight;
        if (update.recipeExpansion) current.recipeExpansion = update.recipeExpansion;
        if (dryRun) {
          current.diff = planDifference(previousPlan, current.plan);
          preview = current;
        }
        requestView = current;
        renderPreview();
        setRequestState(`${update.stage}${update.detail ? ` · ${update.detail}` : ""}`);
      });
      if (!terminal) throw new Error("authority stream ended without a terminal update");
      const failed = terminal.stage === "failed" || terminal.stage === "AUTHOR_RUN_STAGE_FAILED" || current.diagnostics.some((item) => String(item.severity || "").toLowerCase().includes("error"));
      if (failed) throw new Error(terminal.detail || "authority rejected the run");
      if (dryRun) {
        if (!current.plan || !current.planHash) throw new Error("dry-run completed without a canonical plan identity");
        previousPlan = clone(current.plan);
        preview = current;
        setRequestState(`canonical preview ready · ${current.planHash}`);
      } else {
        const runID = terminal.run?.run_id || terminal.run?.runId || "";
        setRequestState(runID ? `queued · ${runID}` : "queued");
        window.dispatchEvent(new CustomEvent("trainvm-refresh", { detail: { runID } }));
      }
    } catch (error) {
      current.diagnostics.push({ path: "/author-run", message: error.message });
      if (dryRun) preview = current;
      requestView = current;
      renderPreview();
      setRequestState(`failed · ${error.message}`);
    } finally {
      requestBusy = false;
      renderComposer();
    }
  }

  function exactImportedDocument(value) {
    const exactKeys = (object, required, optional = []) => object && typeof object === "object" && !Array.isArray(object) &&
      required.every((key) => Object.prototype.hasOwnProperty.call(object, key)) &&
      Object.keys(object).every((key) => required.includes(key) || optional.includes(key));
    if (!exactKeys(value, ["api_version", "author", "reason", "source"], ["input_content"]) || value.api_version !== API_VERSION ||
        !exactKeys(value.source, ["recipe"]) || !exactKeys(value.source.recipe, ["instance", "registry_path"]) ||
        !exactKeys(value.source.recipe.instance, ["api_version", "overrides", "recipe", "run_identity"]) ||
        value.source.recipe.instance.api_version !== INSTANCE_VERSION ||
        !exactKeys(value.source.recipe.instance.recipe, ["name", "version"]) ||
        !value.source.recipe.instance.overrides || typeof value.source.recipe.instance.overrides !== "object" || Array.isArray(value.source.recipe.instance.overrides)) {
      throw new Error("compact document does not match the closed AuthorRun recipe shape");
    }
    if (value.source.recipe.registry_path !== registryPath) {
      throw new Error("compact document must use the loaded authority recipe registry");
    }
    if (value.input_content && (!exactKeys(value.input_content, ["api_version", "paths"]) ||
        value.input_content.api_version !== "trainvm.input-content-root-set/v1" || !Array.isArray(value.input_content.paths))) {
      throw new Error("input_content does not match its closed root-set shape");
    }
    const profileIdentity = JSON.stringify([value.source.recipe.instance.recipe.name, value.source.recipe.instance.recipe.version]);
    if (!(registry?.recipes || []).some((profile) => identityOf(profile) === profileIdentity)) {
      throw new Error("compact document selects an unavailable exact recipe");
    }
    const profile = registry.recipes.find((candidate) => identityOf(candidate) === profileIdentity);
    const fields = new Set((profile.overrides || []).map((field) => field.name));
    if (Object.keys(value.source.recipe.instance.overrides).some((name) => !fields.has(name))) {
      throw new Error("compact document contains an unknown recipe override");
    }
    return clone(value);
  }

  function importSource() {
    const source = byID("vm-recipe-source");
    try {
      const documentValue = exactImportedDocument(JSON.parse(source.value));
      selectedIdentity = JSON.stringify([documentValue.source.recipe.instance.recipe.name, documentValue.source.recipe.instance.recipe.version]);
      instance = documentValue.source.recipe.instance;
      registryPath = documentValue.source.recipe.registry_path;
      author = documentValue.author;
      reason = documentValue.reason;
      inputRoots = documentValue.input_content?.paths || [];
      invalidatePreview();
      renderComposer();
      setRequestState("compact document imported exactly · preview required");
    } catch (error) {
      setRequestState(`import rejected · ${error.message}`);
    }
  }

  async function copySource() {
    const source = JSON.stringify(authorDocument(), null, 2);
    try {
      await navigator.clipboard.writeText(source);
      setRequestState("compact document copied");
    } catch (_) {
      const textarea = byID("vm-recipe-source");
      textarea?.focus();
      textarea?.select();
      setRequestState("compact document selected for copy");
    }
  }

  function captureIdentityInputs() {
    instance.run_identity = byID("vm-recipe-run")?.value.trim() || "";
    author = byID("vm-recipe-author")?.value || "";
    reason = byID("vm-recipe-reason")?.value || "";
    inputRoots = (byID("vm-recipe-roots")?.value || "").split("\n").map((value) => value.trim()).filter(Boolean);
  }

  function captureField(input) {
    const name = input.dataset.vmRecipeField;
    if (!name) return;
    let value = input.value;
    if (input.dataset.vmKind === "boolean") value = input.value === "true";
    else if (input.dataset.vmKind === "integer") value = Number(input.value);
    else if (input.dataset.vmKind === "number") value = Number(input.value);
    instance.overrides[name] = value;
  }

  async function loadProfiles(force = false) {
    if (loaded && !force) return;
    const state = byID("vm-recipe-registry-state");
    if (state) state.textContent = "loading authority registry…";
    try {
      const response = await fetch(PROFILE_ENDPOINT, { cache: "no-store" });
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      const catalog = await response.json();
      registry = catalog.schema || null;
      registryHash = String(catalog.schema_hash || "");
      registryPath = String(registry?.default_registry_path || registry?.registry_path ||
        catalog.default_registry_path || catalog.registry_path || registryPath || "");
      loaded = true;
      ensureSelection();
      renderComposer();
    } catch (error) {
      registry = null;
      if (state) state.textContent = `recipe authority unavailable · ${error.message}`;
      renderComposer();
    }
  }

  const authoring = byID("trainvm-authoring");
  if (authoring) authoring.addEventListener("toggle", () => { if (authoring.open) loadProfiles(); });
  byID("vm-recipe-composer")?.addEventListener("change", (event) => {
    if (!instance) return;
    if (event.target.id === "vm-recipe-select") {
      selectedIdentity = event.target.value;
      instance = null;
      ensureSelection();
      renderComposer();
      return;
    }
    if (event.target.dataset.vmRecipePresent) {
      const name = event.target.dataset.vmRecipePresent;
      const field = (selectedProfile()?.overrides || []).find((candidate) => candidate.name === name);
      if (event.target.checked) instance.overrides[name] = effectiveValue(selectedProfile(), field);
      else delete instance.overrides[name];
      invalidatePreview();
      renderComposer();
      return;
    }
    captureIdentityInputs();
    captureField(event.target);
    invalidatePreview();
    renderComposer();
  });
  byID("vm-recipe-composer")?.addEventListener("input", (event) => {
    if (!instance || event.target.id === "vm-recipe-source") return;
    captureIdentityInputs();
    captureField(event.target);
    invalidatePreview();
    const source = byID("vm-recipe-source");
    if (source) source.value = JSON.stringify(authorDocument(), null, 2);
    refreshLocalState();
  });
  byID("vm-recipe-composer")?.addEventListener("click", (event) => {
    if (event.target.closest("#vm-recipe-preview")) authorRun(true);
    else if (event.target.closest("#vm-recipe-launch")) authorRun(false);
    else if (event.target.closest("#vm-recipe-import")) importSource();
    else if (event.target.closest("#vm-recipe-copy")) copySource();
  });
})();
