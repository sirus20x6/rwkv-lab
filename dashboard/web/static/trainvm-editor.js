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
  let selectedRunIdentity = null;
  let planDiff = null;
  let planDiffBusy = false;
  let planDiffGeneration = 0;
  let trainingRegistry = null;
  let operationRegistry = null;
  let operationNode = "";
  let operationComponentName = "";
  let composerOperation = "";
  const operationSlotDrafts = new Map();
  let composerNode = "";
  let composerFamily = "";
  let composerSlot = "";
  let composerComponent = "";
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

  function schemaAtPath(path) {
    let node = schema;
    let value = draft;
    for (const key of path) {
      const next = value?.[key];
      node = childSchema(node, key, next);
      value = next;
    }
    return resolveSchema(node, value);
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

  function valueAtPath(path) {
    let current = draft;
    for (const part of path) current = current?.[part];
    return current;
  }

  function defaultForSchema(node) {
    const resolved = resolveSchema(node, undefined);
    if ("const" in resolved) return structuredClone(resolved.const);
    if ("default" in resolved) return structuredClone(resolved.default);
    if (Array.isArray(resolved.enum) && resolved.enum.length) {
      return structuredClone(resolved.enum[0]);
    }
    const kind = typeOf(resolved, undefined);
    if (kind === "object") {
      const result = {};
      for (const key of resolved.required || []) {
        result[key] = defaultForSchema(resolved.properties?.[key] ||
          resolved.additionalProperties || {});
      }
      return result;
    }
    if (kind === "array") {
      return Array.from({ length: Number(resolved.minItems || 0) }, () =>
        defaultForSchema(resolved.items || {}));
    }
    if (kind === "boolean") return false;
    if (kind === "integer" || kind === "number") {
      return resolved.minimum === undefined ? 0 : resolved.minimum;
    }
    return "";
  }

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
      const minimum = resolved.minLength === undefined ? "" : ` minlength="${escapeHTML(resolved.minLength)}"`;
      const maximum = resolved.maxLength === undefined ? "" : ` maxlength="${escapeHTML(resolved.maxLength)}"`;
      const pattern = resolved.pattern === undefined ? "" : ` pattern="${escapeHTML(resolved.pattern)}"`;
      control = `<input type="text"${minimum}${maximum}${pattern} value="${escapeHTML(value)}" data-vm-edit="${pathValue}" data-vm-kind="string" />`;
    }
    return `<div class="vm-schema-field" title="${title}"><label class="${requiredClass}">${escapeHTML(label)}</label>${control}</div>`;
  }

  function renderNode(label, value, node, path, depth, required = false, removable = false) {
    const resolved = resolveSchema(node, value);
    const kind = typeOf(resolved, value);
    if (kind !== "object" && kind !== "array") {
      const remove = removable ? `<button class="btn sm vm-schema-remove" type="button" data-vm-remove="${encodedPath(path)}">remove</button>` : "";
      return `<div class="vm-schema-leaf">${renderPrimitive(label, value, resolved, path, required)}${remove}</div>`;
    }
    const entries = kind === "array"
      ? (Array.isArray(value) ? value.map((item, index) => [String(index), item]) : [])
      : Object.entries(value && typeof value === "object" ? value : {});
    const requiredFields = new Set(resolved.required || []);
    const minimumItems = Number(resolved.minItems || 0);
    const children = entries.map(([key, child]) => renderNode(
      kind === "array" ? `[${key}]` : key,
      child,
      kind === "array" ? resolveSchema(resolved.items, child) : childSchema(resolved, key, child),
      [...path, kind === "array" ? Number(key) : key], depth + 1,
      requiredFields.has(key), kind === "array" ? entries.length > minimumItems : !requiredFields.has(key))).join("");
    const missingProperties = kind === "object" ? Object.entries(resolved.properties || {})
      .filter(([key]) => !(key in (value || {})) && !requiredFields.has(key)) : [];
    const optionalAdds = missingProperties.length ? `<div class="vm-schema-adds">${missingProperties.map(([key]) =>
      `<button class="btn sm" type="button" data-vm-add-property="${encodedPath(path)}" data-vm-property="${escapeHTML(key)}">+ ${escapeHTML(key)}</button>`).join("")}</div>` : "";
    const mapAdd = kind === "object" && resolved.additionalProperties && typeof resolved.additionalProperties === "object" ?
      `<div class="vm-schema-map-add"><input type="text" maxlength="192" data-vm-map-key="${encodedPath(path)}" placeholder="new key" aria-label="New key in ${escapeHTML(label)}" /><button class="btn sm" type="button" data-vm-add-map="${encodedPath(path)}">add entry</button></div>` : "";
    const arrayAdd = kind === "array" ? `<div class="vm-schema-adds"><button class="btn sm" type="button" data-vm-add-array="${encodedPath(path)}">+ item</button></div>` : "";
    const remove = removable ? `<button class="btn sm vm-schema-remove" type="button" data-vm-remove="${encodedPath(path)}">remove</button>` : "";
    const open = depth < 2 ? " open" : "";
    const count = entries.length;
    return `<details class="vm-schema-group"${open}><summary>${escapeHTML(label)}<span class="vm-kind">${kind} · ${count}</span>${remove}</summary>${children || '<div class="empty">empty</div>'}${optionalAdds}${mapAdd}${arrayAdd}</details>`;
  }

  function renderForm() {
    const target = byID("vm-schema-form");
    if (!target || !schema || !draft) return;
    target.innerHTML = renderNode("experiment", draft, schema, [], 0, true);
    const source = byID("vm-json-source");
    if (source) source.value = JSON.stringify(draft, null, 2);
    renderOperationComposer();
    renderTrainingComposer();
  }

  function componentIdentity(descriptor) {
    const key = descriptor?.key || {};
    return JSON.stringify([key.category || "", key.name || "", key.version || ""]);
  }

  function componentLabel(descriptor) {
    const key = descriptor?.key || {};
    return `${key.category || "unknown"} · ${key.name || "unknown"} · ${key.version || "unknown"}`;
  }

  function operationIdentity(descriptor) {
    const key = descriptor?.key || {};
    return JSON.stringify([
      key.adapter || "", key.version || "", key.runtime || "",
      key.operation || "", key.contract || "",
    ]);
  }

  function operationLabel(descriptor) {
    const key = descriptor?.key || {};
    return `${key.adapter || "unknown"} · ${key.operation || "unknown"} · ${key.version || "unknown"}`;
  }

  function selectedOperationDescriptor() {
    const operations = operationRegistry?.operations || [];
    return operations.find((item) => operationIdentity(item) === composerOperation) || operations[0] || null;
  }

  function compatibleComponents(category, family) {
    return (trainingRegistry?.components || []).filter((descriptor) => {
      const families = descriptor.model_families || [];
      return descriptor.key?.category === category &&
        (families.includes("*") || families.includes(family));
    });
  }

  function initialComponentConfiguration(descriptor, previous) {
    const configuration = {};
    for (const field of descriptor.configuration || []) {
      if (previous && componentIdentity({ key: previous.key }) === componentIdentity(descriptor) &&
          Object.prototype.hasOwnProperty.call(previous.configuration || {}, field.name)) {
        configuration[field.name] = structuredClone(previous.configuration[field.name]);
      } else if (field.default !== undefined) {
        configuration[field.name] = structuredClone(field.default);
      }
    }
    return configuration;
  }

  function operationSlotScope() {
    return JSON.stringify([operationNode, composerOperation]);
  }

  function operationSlotState() {
    const scope = operationSlotScope();
    if (!operationSlotDrafts.has(scope)) operationSlotDrafts.set(scope, { slots: {} });
    return operationSlotDrafts.get(scope);
  }

  function operationFieldToken(slot, component, field) {
    return JSON.stringify([slot, componentIdentity(component), field.name]);
  }

  function operationComponentFieldControl(slot, descriptor, field, configuration) {
    const kind = String(field.type || "string");
    const name = escapeHTML(field.name || "field");
    const token = escapeHTML(operationFieldToken(slot, descriptor, field));
    const required = field.required ? " vm-required" : "";
    const hasConfigured = Object.prototype.hasOwnProperty.call(configuration || {}, field.name);
    const present = field.required || hasConfigured || field.default !== undefined;
    const value = hasConfigured ? configuration[field.name] : field.default;
    const disabled = present ? "" : " disabled";
    const requiredInput = field.required && value === undefined ? " required" : "";
    let control;
    if (kind === "enumeration") {
      const placeholder = value === undefined ? '<option value="" selected>select…</option>' : "";
      control = `<select data-vm-operation-field="${token}" data-vm-kind="string"${requiredInput}${disabled}>${placeholder}` +
        (field.values || []).map((option) => `<option${option === value ? " selected" : ""}>${escapeHTML(option)}</option>`).join("") + "</select>";
    } else if (kind === "boolean" && field.required && value === undefined) {
      control = `<select data-vm-operation-field="${token}" data-vm-kind="boolean" required>` +
        '<option value="" selected>select…</option><option value="true">true</option><option value="false">false</option></select>';
    } else if (kind === "boolean") {
      control = `<input type="checkbox" data-vm-operation-field="${token}" data-vm-kind="boolean"${value ? " checked" : ""}${disabled} />`;
    } else if (kind === "integer" || kind === "number") {
      const minimum = field.minimum === undefined ? "" : ` min="${escapeHTML(field.minimum)}"`;
      const maximum = field.maximum === undefined ? "" : ` max="${escapeHTML(field.maximum)}"`;
      control = `<input type="number" step="${kind === "integer" ? "1" : "any"}"${minimum}${maximum}${requiredInput} value="${value === undefined ? "" : escapeHTML(value)}" data-vm-operation-field="${token}" data-vm-kind="${kind}"${disabled} />`;
    } else {
      control = `<input type="text" maxlength="4096"${requiredInput} value="${value === undefined ? "" : escapeHTML(value)}" data-vm-operation-field="${token}" data-vm-kind="string"${disabled} />`;
    }
    const presence = field.required ? "" : `<input type="checkbox" data-vm-operation-present="${token}"${present ? " checked" : ""} aria-label="Include ${name}" />`;
    return `<div class="vm-schema-field vm-component-field" title="${escapeHTML(field.description || "")}"><label class="${required}">${name}${field.unit ? ` · ${escapeHTML(field.unit)}` : ""}</label><div class="vm-component-value">${presence}${control}</div></div>`;
  }

  function captureOperationSlotDraft() {
    const state = operationSlotState();
    for (const input of document.querySelectorAll("[data-vm-operation-field]")) {
      let token;
      try { token = JSON.parse(input.dataset.vmOperationField); } catch (_) { continue; }
      if (!Array.isArray(token) || token.length !== 3) continue;
      const [slot, identity, name] = token;
      const slotState = state.slots[slot] ||= { selected: identity, configurations: {} };
      const configuration = slotState.configurations[identity] ||= {};
      const presence = [...document.querySelectorAll("[data-vm-operation-present]")]
        .find((candidate) => candidate.dataset.vmOperationPresent === input.dataset.vmOperationField);
      if (presence && !presence.checked) {
        delete configuration[name];
        continue;
      }
      if (input.value === "" && input.type !== "checkbox") {
        delete configuration[name];
        continue;
      }
      let value = input.type === "checkbox" ? input.checked : input.value;
      if (input.dataset.vmKind === "boolean" && input.tagName === "SELECT") value = input.value === "true";
      else if (input.dataset.vmKind === "integer" || input.dataset.vmKind === "number") value = Number(input.value);
      configuration[name] = value;
    }
    for (const select of document.querySelectorAll("[data-vm-operation-slot]")) {
      const slot = select.dataset.vmOperationSlot;
      const slotState = state.slots[slot] ||= { configurations: {} };
      slotState.selected = select.value;
    }
  }

  function validateComponentConfiguration(descriptor, configuration) {
    for (const field of descriptor.configuration || []) {
      if (!Object.prototype.hasOwnProperty.call(configuration, field.name)) {
        if (field.required) return `${field.name} is required`;
        continue;
      }
      const value = configuration[field.name];
      if (field.type === "boolean" && typeof value !== "boolean") return `${field.name} must be boolean`;
      if (field.type === "integer" && !Number.isInteger(value)) return `${field.name} must be an integer`;
      if (field.type === "number" && (typeof value !== "number" || !Number.isFinite(value))) return `${field.name} must be finite`;
      if ((field.type === "string" || field.type === "enumeration") && typeof value !== "string") return `${field.name} must be text`;
      if (field.minimum !== undefined && value < Number(field.minimum)) return `${field.name} is below its minimum`;
      if (field.maximum !== undefined && value > Number(field.maximum)) return `${field.name} exceeds its maximum`;
      if (field.type === "enumeration" && !(field.values || []).includes(value)) return `${field.name} is not an allowed value`;
    }
    return "";
  }

  function operationPortRows(ports, direction) {
    const entries = Object.entries(ports || {});
    if (!entries.length) return `<div class="empty">no declared ${direction} ports</div>`;
    return entries.map(([name, port]) =>
      `<div class="vm-operation-port"><code>${escapeHTML(name)}</code><span>${escapeHTML(port.type || "unknown")}${port.artifact_type ? ` · ${escapeHTML(port.artifact_type)}` : ""}${port.artifact_schema ? ` · ${escapeHTML(port.artifact_schema)}` : ""}</span><strong>${port.required ? "required" : "optional"}</strong></div>`).join("");
  }

  function defaultInputBinding(name, port) {
    if (port.type === "artifact") {
      const match = Object.entries(draft?.spec?.artifacts || {}).find(([, artifact]) =>
        (!port.artifact_type || artifact.type === port.artifact_type) &&
        (!port.artifact_schema || artifact.schema === port.artifact_schema));
      return match ? { artifact: match[0] } : null;
    }
    if (port.type === "object") return { literal: {} };
    if (port.type === "boolean") return { literal: false };
    if (port.type === "integer" || port.type === "number") return { literal: 0 };
    return { literal: name === "concurrency_key" ? draft?.spec?.workspace?.concurrency_key || "resource" : "" };
  }

  function renderOperationComposer() {
    const target = byID("vm-operation-composer");
    const state = byID("vm-operation-registry-state");
    if (!target || !draft) return;
    const operations = operationRegistry?.operations || [];
    if (state) state.textContent = operationRegistry ? `${operations.length} exact operations · ${String(operationRegistry.schemaHash || "").slice(0, 18)}…` : "authority registry unavailable";
    if (!operations.length) {
      target.innerHTML = '<div class="empty">no operations are authorized by this daemon</div>';
      return;
    }
    const nodes = Object.entries(draft?.spec?.workflow?.nodes || {});
    if (!nodes.length) {
      target.innerHTML = '<div class="empty">add a workflow node before binding an operation</div>';
      return;
    }
    if (!nodes.some(([name]) => name === operationNode)) operationNode = nodes[0][0];
    if (!operations.some((item) => operationIdentity(item) === composerOperation)) {
      composerOperation = operationIdentity(operations[0]);
    }
    const descriptor = selectedOperationDescriptor();
    const key = descriptor.key || {};
    if (!operationComponentName) {
      operationComponentName = String(key.adapter || "component")
        .replace(/[^A-Za-z0-9_.:-]/g, "_").replace(/^[^A-Za-z]/, "component_");
    }
    const nodeOptions = nodes.map(([name]) => `<option${name === operationNode ? " selected" : ""}>${escapeHTML(name)}</option>`).join("");
    const operationOptions = operations.map((item) => {
      const identity = operationIdentity(item);
      return `<option value="${escapeHTML(identity)}"${identity === composerOperation ? " selected" : ""}>${escapeHTML(operationLabel(item))}</option>`;
    }).join("");
    const composition = descriptor.training_composition;
    const selectedNode = draft.spec.workflow.nodes[operationNode];
    const slotDraft = operationSlotState();
    const slots = composition ? Object.entries(composition.slots || {}).map(([slot, category]) => {
      const compatible = compatibleComponents(category, composition.model_family);
      const existing = selectedNode?.invoke?.training?.components?.[slot];
      const existingIdentity = componentIdentity({ key: existing?.key });
      const state = slotDraft.slots[slot] ||= { selected: "", configurations: {} };
      if (!compatible.some((item) => componentIdentity(item) === state.selected)) {
        state.selected = compatible.some((item) => componentIdentity(item) === existingIdentity) ?
          existingIdentity : componentIdentity(compatible[0]);
      }
      const selected = compatible.find((item) => componentIdentity(item) === state.selected) || null;
      if (selected && !state.configurations[state.selected]) {
        const previous = existingIdentity === state.selected ? existing : null;
        state.configurations[state.selected] = initialComponentConfiguration(selected, previous);
      }
      const options = compatible.map((item) => {
        const identity = componentIdentity(item);
        return `<option value="${escapeHTML(identity)}"${identity === state.selected ? " selected" : ""}>${escapeHTML(componentLabel(item))}</option>`;
      }).join("");
      const fields = selected ? (selected.configuration || []).map((field) =>
        operationComponentFieldControl(slot, selected, field, state.configurations[state.selected] || {})).join("") || '<div class="empty">no configuration fields</div>' : "";
      return `<div class="vm-operation-slot"><div class="vm-component-input"><label>${escapeHTML(slot)} · ${escapeHTML(category)}</label>${compatible.length ? `<select data-vm-operation-slot="${escapeHTML(slot)}">${options}</select>` : '<span class="vm-operation-missing">no compatible component</span>'}</div>${selected ? `<div class="vm-operation-slot-fields">${fields}</div>` : ""}</div>`;
    }).join("") : '<div class="empty">operation has no training-component contract</div>';
    const lifecycle = Object.entries(descriptor.lifecycle || {}).map(([name, value]) =>
      `<span>${escapeHTML(name)}=${escapeHTML(value)}</span>`).join(" · ");
    target.innerHTML = `<div class="vm-operation-compose">` +
      `<div class="vm-component-input"><label>workflow node</label><select id="vm-operation-node">${nodeOptions}</select></div>` +
      `<div class="vm-component-input"><label>component name</label><input id="vm-operation-component" maxlength="192" value="${escapeHTML(operationComponentName)}" /></div>` +
      `<div class="vm-component-input"><label>registered operation</label><select id="vm-operation-select">${operationOptions}</select></div>` +
      `<button class="btn sm" id="vm-operation-bind" type="button">bind exact operation</button></div>` +
      `<div class="vm-operation-detail"><div class="vm-component-meta"><strong>${escapeHTML(operationLabel(descriptor))}</strong>` +
      `${escapeHTML(descriptor.effect || "unknown")} · ${escapeHTML(descriptor.idempotency || "unknown")}<br />${lifecycle || "stateless lifecycle"}<br />capabilities: ${escapeHTML((descriptor.required_capabilities || []).join(", ") || "none")}</div>` +
      `<div><div class="vm-editor-heading">inputs</div>${operationPortRows(descriptor.authoring?.inputs, "input")}</div>` +
      `<div><div class="vm-editor-heading">outputs</div>${operationPortRows(descriptor.authoring?.outputs, "output")}</div></div>` +
      `<div class="vm-operation-slots"><div class="vm-editor-heading">closed training slots${composition ? ` · ${escapeHTML(composition.model_family)}` : ""}</div><div class="vm-component-fields">${slots}</div></div>`;
  }

  function bindOperation() {
    operationNode = byID("vm-operation-node")?.value || operationNode;
    operationComponentName = byID("vm-operation-component")?.value.trim() || "";
    composerOperation = byID("vm-operation-select")?.value || composerOperation;
    captureOperationSlotDraft();
    const descriptor = selectedOperationDescriptor();
    const node = draft?.spec?.workflow?.nodes?.[operationNode];
    const key = descriptor?.key || {};
    const symbolic = /^[A-Za-z][A-Za-z0-9_.:-]*$/;
    if (!descriptor || !node || !symbolic.test(operationComponentName)) {
      byID("vm-editor-state").textContent = "operation binding requires a node, symbolic component name, and exact descriptor";
      return;
    }
    const composition = descriptor.training_composition;
    let selections = null;
    if (composition) {
      selections = {};
      const slotDraft = operationSlotState();
      for (const [slot, category] of Object.entries(composition.slots || {})) {
        const state = slotDraft.slots[slot];
        const chosen = compatibleComponents(category, composition.model_family)
          .find((item) => componentIdentity(item) === state?.selected);
        if (!chosen) {
          byID("vm-editor-state").textContent = `operation slot ${slot} has no compatible registered component`;
          return;
        }
        const configuration = structuredClone(state?.configurations?.[state.selected] || {});
        const invalid = validateComponentConfiguration(chosen, configuration);
        if (invalid) {
          byID("vm-editor-state").textContent = `operation slot ${slot}: ${invalid}`;
          return;
        }
        selections[slot] = { key: { ...chosen.key }, configuration };
      }
    }
    const components = draft.spec.components ||= {};
    const existing = components[operationComponentName];
    if (existing && (existing.adapter !== key.adapter || existing.version !== key.version ||
        existing.runtime !== key.runtime)) {
      byID("vm-editor-state").textContent = "component name is already bound to a different exact adapter";
      return;
    }
    const component = existing || {
      adapter: key.adapter, version: key.version, runtime: key.runtime, operations: {},
    };
    component.operations ||= {};
    component.operations[key.operation] = { contract: key.contract };
    components[operationComponentName] = component;
    node.invoke ||= { inputs: {} };
    node.invoke.component = operationComponentName;
    node.invoke.operation = key.operation;
    const previousInputs = node.invoke.inputs || {};
    const nextInputs = {};
    for (const [name, port] of Object.entries(descriptor.authoring?.inputs || {})) {
      if (Object.prototype.hasOwnProperty.call(previousInputs, name)) {
        nextInputs[name] = previousInputs[name];
      } else if (port.required) {
        const binding = defaultInputBinding(name, port);
        if (binding) nextInputs[name] = binding;
      }
    }
    node.invoke.inputs = nextInputs;
    node.effect = descriptor.effect;
    node.idempotency = descriptor.idempotency;
    const declaredOutputs = descriptor.authoring?.outputs || {};
    const previousPublishes = node.publishes || {};
    const nextPublishes = {};
    for (const [name, port] of Object.entries(declaredOutputs)) {
      if (Object.prototype.hasOwnProperty.call(previousPublishes, name)) {
        nextPublishes[name] = previousPublishes[name];
      } else if (port.required) {
        const artifactName = `${operationComponentName}.${name}`;
        draft.spec.artifacts ||= {};
        draft.spec.artifacts[artifactName] ||= {
          type: port.artifact_type || "opaque",
          ...(port.artifact_schema ? { schema: port.artifact_schema } : {}),
          immutability: "mutable_until_publish",
          fingerprint: "manifest_sha256",
        };
        nextPublishes[name] = artifactName;
      }
    }
    if (Object.keys(nextPublishes).length) node.publishes = nextPublishes;
    else delete node.publishes;
    if (composition) {
      node.invoke.training = {
        model_family: composition.model_family,
        components: selections,
      };
    } else {
      delete node.invoke.training;
    }
    renderForm();
    scheduleCompile();
  }

  function componentFieldControl(field, configuration) {
    const kind = String(field.type || "string");
    const name = escapeHTML(field.name || "field");
    const required = field.required ? " vm-required" : "";
    const description = escapeHTML(field.description || "");
    const hasConfigured = Object.prototype.hasOwnProperty.call(configuration || {}, field.name);
    const present = field.required || hasConfigured || field.default !== undefined;
    const value = hasConfigured ? configuration[field.name] : field.default;
    const disabled = present ? "" : " disabled";
    let control;
    if (kind === "enumeration") {
      control = `<select data-vm-component-field="${name}" data-vm-kind="string"${disabled}>` +
        (field.values || []).map((option) => `<option${option === value ? " selected" : ""}>${escapeHTML(option)}</option>`).join("") + "</select>";
    } else if (kind === "boolean") {
      control = `<input type="checkbox" data-vm-component-field="${name}" data-vm-kind="boolean"${value ? " checked" : ""}${disabled} />`;
    } else if (kind === "integer" || kind === "number") {
      const minimum = field.minimum === undefined ? "" : ` min="${escapeHTML(field.minimum)}"`;
      const maximum = field.maximum === undefined ? "" : ` max="${escapeHTML(field.maximum)}"`;
      control = `<input type="number" step="${kind === "integer" ? "1" : "any"}"${minimum}${maximum} value="${value === undefined ? "" : escapeHTML(value)}" data-vm-component-field="${name}" data-vm-kind="${kind}"${disabled} />`;
    } else {
      control = `<input type="text" maxlength="4096" value="${value === undefined ? "" : escapeHTML(value)}" data-vm-component-field="${name}" data-vm-kind="string"${disabled} />`;
    }
    const presence = field.required ? "" : `<input type="checkbox" data-vm-component-present="${name}"${present ? " checked" : ""} aria-label="Include ${name}" />`;
    return `<div class="vm-schema-field vm-component-field" title="${description}"><label class="${required}">${name}${field.unit ? ` · ${escapeHTML(field.unit)}` : ""}</label><div class="vm-component-value">${presence}${control}</div></div>`;
  }

  function operationDescriptorForNode(node) {
    const component = draft?.spec?.components?.[node?.invoke?.component];
    if (!component) return null;
    return (operationRegistry?.operations || []).find((descriptor) =>
      descriptor.key?.adapter === component.adapter &&
      descriptor.key?.version === component.version &&
      descriptor.key?.runtime === component.runtime &&
      descriptor.key?.operation === node.invoke.operation &&
      descriptor.key?.contract === component.operations?.[node.invoke.operation]?.contract) || null;
  }

  function composerComponentDescriptors() {
    const node = draft?.spec?.workflow?.nodes?.[composerNode];
    const operation = operationDescriptorForNode(node);
    const composition = operation?.training_composition;
    const category = composition?.slots?.[composerSlot];
    return category ? compatibleComponents(category, composition.model_family) : [];
  }

  function selectedComponentDescriptor() {
    const components = composerComponentDescriptors();
    return components.find((item) => componentIdentity(item) === composerComponent) || components[0] || null;
  }

  function renderTrainingComposer() {
    const target = byID("vm-component-composer");
    const state = byID("vm-component-registry-state");
    if (!target || !draft) return;
    const components = trainingRegistry?.components || [];
    if (state) state.textContent = trainingRegistry ? `${components.length} exact descriptors · ${String(trainingRegistry.schemaHash || "").slice(0, 18)}…` : "authority registry unavailable";
    if (!components.length) {
      target.innerHTML = '<div class="empty">no training components are authorized by this daemon</div>';
      return;
    }
    const nodes = Object.entries(draft?.spec?.workflow?.nodes || {})
      .filter(([, node]) => operationDescriptorForNode(node)?.training_composition);
    if (!nodes.length) {
      target.innerHTML = '<div class="empty">the draft has no process node that can receive training components</div>';
      return;
    }
    if (!nodes.some(([name]) => name === composerNode)) composerNode = nodes[0][0];
    const selectedNode = draft.spec.workflow.nodes[composerNode];
    const operation = operationDescriptorForNode(selectedNode);
    const composition = operation.training_composition;
    composerFamily = composition.model_family;
    const slotEntries = Object.entries(composition.slots || {});
    if (!slotEntries.some(([slot]) => slot === composerSlot)) composerSlot = slotEntries[0]?.[0] || "";
    const compatible = composerComponentDescriptors();
    const existingSelection = selectedNode.invoke?.training?.components?.[composerSlot];
    const existingIdentity = componentIdentity({ key: existingSelection?.key });
    if (compatible.some((item) => componentIdentity(item) === existingIdentity)) {
      composerComponent = existingIdentity;
    } else if (!compatible.some((item) => componentIdentity(item) === composerComponent)) {
      composerComponent = componentIdentity(compatible[0]);
    }
    const descriptor = selectedComponentDescriptor();
    const nodeOptions = nodes.map(([name]) => `<option${name === composerNode ? " selected" : ""}>${escapeHTML(name)}</option>`).join("");
    const componentOptions = compatible.map((item) => {
      const identity = componentIdentity(item);
      return `<option value="${escapeHTML(identity)}"${identity === composerComponent ? " selected" : ""}>${escapeHTML(componentLabel(item))}</option>`;
    }).join("");
    const existingConfiguration = existingIdentity === componentIdentity(descriptor) ? existingSelection?.configuration || {} : {};
    const fields = descriptor ? (descriptor.configuration || []).map((field) => componentFieldControl(field, existingConfiguration)).join("") || '<div class="empty">no configuration fields</div>' : '<div class="empty">no compatible component is registered</div>';
    const families = (descriptor?.model_families || []).join(", ");
    const capabilities = (descriptor?.required_capabilities || []).join(", ") || "none";
    const bound = [];
    for (const [nodeName, node] of Object.entries(draft.spec.workflow.nodes || {})) {
      for (const [slot, selection] of Object.entries(node?.invoke?.training?.components || {})) {
        bound.push(`<div class="vm-component-existing-row"><span>${escapeHTML(nodeName)} · ${escapeHTML(slot)}</span><span>${escapeHTML(componentLabel({ key: selection.key }))}</span><span>declared slot</span></div>`);
      }
    }
    target.innerHTML = `<div class="vm-component-compose">` +
      `<div class="vm-component-input"><label>process node</label><select id="vm-component-node">${nodeOptions}</select></div>` +
      `<div class="vm-component-input"><label>model family</label><span>${escapeHTML(composerFamily)}</span></div>` +
      `<div class="vm-component-input"><label>declared slot</label><select id="vm-component-slot">${slotEntries.map(([slot, category]) => `<option value="${escapeHTML(slot)}"${slot === composerSlot ? " selected" : ""}>${escapeHTML(slot)} · ${escapeHTML(category)}</option>`).join("")}</select></div>` +
      `<div class="vm-component-input"><label>exact component</label><select id="vm-component-select">${componentOptions}</select></div>` +
      `<button class="btn sm" id="vm-component-add" type="button">bind component</button></div>` +
      `<div class="vm-component-detail"><div class="vm-component-meta"><strong>${escapeHTML(componentLabel(descriptor))}</strong>` +
      `${escapeHTML(descriptor?.backend || "unknown backend")} · ${escapeHTML(descriptor?.implementation || "")}` +
      `<br />families: ${escapeHTML(families)}<br />worker capabilities: ${escapeHTML(capabilities)}<br />state: ${escapeHTML(descriptor?.state_grade || "unknown")}${descriptor?.step_domain ? ` · ${escapeHTML(descriptor.step_domain)}` : ""}</div>` +
      `<div class="vm-component-fields">${fields}<div class="vm-component-bound">Fields are generated from the authority descriptor; binding updates only this declarative draft.</div></div></div>` +
      (bound.length ? `<div class="vm-component-existing"><div class="vm-editor-heading">bound components</div>${bound.join("")}</div>` : "");
  }

  function captureComposerIdentity() {
    composerNode = byID("vm-component-node")?.value || composerNode;
    composerSlot = byID("vm-component-slot")?.value || composerSlot;
    composerComponent = byID("vm-component-select")?.value || composerComponent;
  }

  function bindTrainingComponent() {
    captureComposerIdentity();
    const descriptor = selectedComponentDescriptor();
    const node = draft?.spec?.workflow?.nodes?.[composerNode];
    const state = byID("vm-editor-state");
    const operation = operationDescriptorForNode(node);
    const expectedCategory = operation?.training_composition?.slots?.[composerSlot];
    if (!descriptor || !node || !expectedCategory ||
        descriptor.key?.category !== expectedCategory) {
      if (state) state.textContent = "component binding requires a declared operation slot and compatible exact descriptor";
      return;
    }
    const configuration = {};
    for (const field of descriptor.configuration || []) {
      const input = [...document.querySelectorAll("[data-vm-component-field]")]
        .find((candidate) => candidate.dataset.vmComponentField === field.name);
      if (!input) continue;
      if (input.type !== "checkbox" && input.value === "") {
        if (field.required && field.default === undefined) {
          if (state) state.textContent = `component field ${field.name} is required`;
          return;
        }
        continue;
      }
      const presence = [...document.querySelectorAll("[data-vm-component-present]")]
        .find((candidate) => candidate.dataset.vmComponentPresent === field.name);
      if (presence && !presence.checked) continue;
      let value = input.type === "checkbox" ? input.checked : input.value;
      if (field.type === "integer") value = Number(value);
      else if (field.type === "number") value = Number(value);
      if ((field.type === "integer" && !Number.isInteger(value)) ||
          (field.type === "number" && !Number.isFinite(value)) ||
          (typeof input.checkValidity === "function" && !input.checkValidity())) {
        if (state) state.textContent = `component field ${field.name} must be finite`;
        return;
      }
      configuration[field.name] = value;
    }
    node.invoke.training ||= { model_family: operation.training_composition.model_family, components: {} };
    node.invoke.training.model_family = operation.training_composition.model_family;
    node.invoke.training.components ||= {};
    node.invoke.training.components[composerSlot] = {
      key: { ...descriptor.key }, configuration,
    };
    renderForm();
    scheduleCompile();
  }

  function setAtPath(path, value) {
    if (!path.length) { draft = value; return; }
    let current = draft;
    for (let index = 0; index < path.length - 1; index++) current = current[path[index]];
    current[path[path.length - 1]] = value;
  }

  function removeAtPath(path) {
    if (!path.length) return;
    const parent = valueAtPath(path.slice(0, -1));
    const key = path[path.length - 1];
    if (Array.isArray(parent)) parent.splice(Number(key), 1);
    else if (parent && typeof parent === "object") delete parent[key];
  }

  function commitStructuralEdit(message) {
    renderForm();
    scheduleCompile();
    const state = byID("vm-editor-state");
    if (state) state.textContent = message;
  }

  function scheduleCompile() {
    compileGeneration += 1;
    validatedDraft = null;
    resetPlanDiff("draft changed · compare again after validation");
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
    document.querySelectorAll("#vm-schema-form input, #vm-schema-form select, #vm-schema-form button, #vm-operation-composer input, #vm-operation-composer select, #vm-operation-composer button, #vm-component-composer input, #vm-component-composer select, #vm-component-composer button, #vm-json-source, #vm-apply-json, #vm-load-example, #vm-compile, #vm-diff")
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
    const diffButton = byID("vm-diff");
    const reason = byID("vm-submit-reason");
    if (!button) return;
    const retry = Boolean(submissionIntent);
    const retryPayload = retry ? JSON.parse(submissionIntent.body) : null;
    const fork = currentForkIdentity();
    const alreadySubmitted = Boolean(validatedDraft && submittedDraftSource === validatedDraft.source);
    button.disabled = submissionBusy || (!retry &&
      (!validatedDraft || !reason?.value.trim() || !authorityJournalID || alreadySubmitted));
    button.textContent = submissionBusy ? "submitting…" :
      (retry ? (retryPayload?.forked_from_run_id ? "retry exact fork" : "retry exact submission") :
        (fork ? "create revision fork" : "create queued run"));
    if (diffButton) {
      diffButton.disabled = submissionBusy || planDiffBusy || !validatedDraft || !selectedRunIdentity;
      diffButton.textContent = planDiffBusy ? "comparing…" : "compare selected run";
    }
    if (reason) reason.disabled = retry || submissionBusy;
    lockEditor(retry || submissionBusy);
    if (message) {
      const state = byID("vm-editor-state");
      if (state) state.textContent = message;
    }
  }

  function currentForkIdentity() {
    if (!planDiff || planDiff.result.adoptable_in_place || !validatedDraft || !selectedRunIdentity) return null;
    if (planDiff.source !== validatedDraft.source || planDiff.proposedPlanHash !== validatedDraft.planHash ||
        planDiff.runID !== selectedRunIdentity.runID || planDiff.runRevision !== selectedRunIdentity.runRevision ||
        planDiff.currentPlanHash !== selectedRunIdentity.planHash) return null;
    return {
      runID: planDiff.runID,
      runRevision: planDiff.runRevision,
      planHash: planDiff.currentPlanHash,
    };
  }

  function resetPlanDiff(message = "select a run and compare a valid draft") {
    planDiffGeneration += 1;
    planDiff = null;
    planDiffBusy = false;
    const target = byID("vm-plan-diff");
    if (target) target.innerHTML = `<div class="empty">${escapeHTML(message)}</div>`;
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

  function renderPlanDiff(result, identity) {
    const target = byID("vm-plan-diff");
    if (!target) return;
    const operations = Array.isArray(result.semantic_diff) ? result.semantic_diff : [];
    const shortRun = identity.runID.length > 28 ? `${identity.runID.slice(0, 25)}…` : identity.runID;
    if (!operations.length) {
      target.innerHTML = `<div class="vm-diff-head"><strong>plans are identical</strong><span>${escapeHTML(shortRun)} · r${identity.runRevision}</span></div>` +
        `<div class="empty">No semantic changes; a lineage fork is not needed.</div>`;
      return;
    }
    const rows = operations.map((operation) => {
      const value = Object.prototype.hasOwnProperty.call(operation, "value") ?
        JSON.stringify(operation.value) : (operation.from || "—");
      return `<div class="vm-diff-op"><code>${escapeHTML(operation.op || "change")}</code>` +
        `<code>${escapeHTML(operation.path || "/")}</code><code>${escapeHTML(value)}</code></div>`;
    }).join("");
    target.innerHTML = `<div class="vm-diff-head"><strong>${operations.length} semantic change${operations.length === 1 ? "" : "s"}</strong>` +
      `<span>fork from ${escapeHTML(shortRun)} · r${identity.runRevision}</span></div>${rows}`;
  }

  async function compareDraft() {
    if (planDiffBusy || !validatedDraft || !selectedRunIdentity || !authorityJournalID) return;
    const identity = { ...selectedRunIdentity };
    const preview = { ...validatedDraft };
    const generation = ++planDiffGeneration;
    planDiffBusy = true;
    updateSubmitState("requesting authority semantic diff…");
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(identity.runID)}/diff`, {
        method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_run_revision: identity.runRevision,
          proposed_source_document: preview.source,
          source_format: "json",
          expected_journal_id: authorityJournalID,
          expected_current_plan_hash: identity.planHash,
          expected_proposed_plan_hash: preview.planHash,
        }),
      });
      const text = await response.text();
      if (generation !== planDiffGeneration) return;
      let result = null;
      try { result = JSON.parse(text); } catch (_) { /* HTTP text is shown below. */ }
      if (!response.ok) throw new Error(result?.diagnostics?.[0]?.message || text.trim() || `HTTP ${response.status}`);
      if (result?.proposed_plan_hash !== preview.planHash || !Array.isArray(result?.semantic_diff)) {
        throw new Error("authority returned an inconsistent semantic diff");
      }
      planDiff = {
        runID: identity.runID, runRevision: identity.runRevision,
        currentPlanHash: identity.planHash, proposedPlanHash: preview.planHash,
        source: preview.source, result,
      };
      renderPlanDiff(result, identity);
      updateSubmitState(result.adoptable_in_place ?
        "identical to selected run · no fork lineage needed" :
        `validated fork from ${identity.runID} r${identity.runRevision}`);
    } catch (error) {
      if (generation !== planDiffGeneration) return;
      planDiff = null;
      const target = byID("vm-plan-diff");
      if (target) target.innerHTML = `<div class="vm-diag error">${escapeHTML(error.message)}</div>`;
      updateSubmitState("semantic diff unavailable");
    } finally {
      if (generation === planDiffGeneration) {
        planDiffBusy = false;
        updateSubmitState();
      }
    }
  }

  async function compileDraft() {
    if (!draft) return;
    clearTimeout(compileTimer);
    validatedDraft = null;
    resetPlanDiff("compile a valid draft before comparing");
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
      const fork = currentForkIdentity();
      const payload = {
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
      };
      if (fork) {
        payload.forked_from_run_id = fork.runID;
        payload.expected_parent_run_revision = fork.runRevision;
        payload.expected_parent_plan_hash = fork.planHash;
      }
      submissionIntent = {
        body: JSON.stringify(payload),
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
      const submittedPayload = JSON.parse(intent.body);
      finalMessage = submittedPayload.forked_from_run_id ?
        `fork queued · ${runID} · from ${submittedPayload.forked_from_run_id} r${submittedPayload.expected_parent_run_revision}` :
        `queued · ${runID}`;
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
    resetPlanDiff("loading selected-run comparison context…");
    submittedDraftSource = null;
    updateSubmitState();
    const state = byID("vm-editor-state");
    if (state) state.textContent = "loading schema and reference…";
    try {
      const [schemaResponse, exampleResponse, runsResponse, componentResponse, operationResponse] = await Promise.all([
        fetch("/api/trainvm/schema", { cache: "no-store" }),
        fetch("/api/trainvm/example", { cache: "no-store" }),
        fetch("/api/trainvm/runs", { cache: "no-store" }),
        fetch("/api/trainvm/training-components", { cache: "no-store" }),
        fetch("/api/trainvm/operations", { cache: "no-store" }),
      ]);
      if (!schemaResponse.ok || !exampleResponse.ok || !runsResponse.ok ||
          !componentResponse.ok || !operationResponse.ok) {
        throw new Error("TrainVM authoring endpoints unavailable");
      }
      schema = await schemaResponse.json();
      draft = await exampleResponse.json();
      operationSlotDrafts.clear();
      const runs = await runsResponse.json();
      const componentCatalog = await componentResponse.json();
      trainingRegistry = { ...(componentCatalog.schema || {}), schemaHash: componentCatalog.schema_hash || "" };
      const operationCatalog = await operationResponse.json();
      operationRegistry = { ...(operationCatalog.schema || {}), schemaHash: operationCatalog.schema_hash || "" };
      authorityJournalID = String(runs.journal_id || "");
      if (!authorityJournalID) throw new Error("TrainVM journal identity unavailable");
      applySelectedRun(window.__trainVMSelectedRun || null);
      loaded = true;
      renderForm();
      updateSubmitState();
      await compileDraft();
    } catch (error) {
      if (state) state.textContent = error.message;
    }
  }

  function applySelectedRun(detail) {
    const candidate = detail && typeof detail === "object" ? {
      runID: String(detail.runID || ""),
      runRevision: Number(detail.runRevision || 0),
      planHash: String(detail.planHash || ""),
    } : null;
    const next = candidate?.runID && Number.isInteger(candidate.runRevision) && candidate.runRevision > 0 && candidate.planHash ? candidate : null;
    const changed = selectedRunIdentity?.runID !== next?.runID ||
      selectedRunIdentity?.runRevision !== next?.runRevision ||
      selectedRunIdentity?.planHash !== next?.planHash;
    selectedRunIdentity = next;
    if (changed) {
      resetPlanDiff(next ? `selected ${next.runID} r${next.runRevision} · compare when ready` :
        "select a run and compare a valid draft");
      updateSubmitState();
    }
  }

  const authoring = byID("trainvm-authoring");
  restoreSubmissionIntent();
  updateSubmitState();
  if (authoring) authoring.addEventListener("toggle", () => { if (authoring.open) loadAuthoring(); });
  byID("vm-load-example")?.addEventListener("click", () => loadAuthoring(true));
  byID("vm-compile")?.addEventListener("click", compileDraft);
  byID("vm-diff")?.addEventListener("click", compareDraft);
  byID("vm-submit")?.addEventListener("click", submitDraft);
  byID("vm-submit-reason")?.addEventListener("input", () => updateSubmitState());
  byID("vm-json-source")?.addEventListener("input", () => {
    clearTimeout(compileTimer);
    compileGeneration += 1;
    validatedDraft = null;
    resetPlanDiff("raw source changed · apply and compile before comparing");
    submittedDraftSource = null;
    updateSubmitState("raw source changed · apply JSON before compiling");
  });
  window.addEventListener("trainvm-run-selected", (event) => applySelectedRun(event.detail));
  applySelectedRun(window.__trainVMSelectedRun || null);
  byID("vm-apply-json")?.addEventListener("click", () => {
    const source = byID("vm-json-source");
    try {
      draft = JSON.parse(source.value);
      operationSlotDrafts.clear();
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
    else if (input.dataset.vmKind === "integer") {
      value = Number(value);
      if (!Number.isInteger(value)) {
        byID("vm-editor-state").textContent = "integer field requires an exact integer";
        return;
      }
    } else if (input.dataset.vmKind === "number") {
      value = Number(value);
      if (!Number.isFinite(value)) {
        byID("vm-editor-state").textContent = "numeric field requires a finite value";
        return;
      }
    }
    if (typeof input.checkValidity === "function" && !input.checkValidity()) {
      byID("vm-editor-state").textContent = "field violates its declared schema constraint";
      return;
    }
    setAtPath(path, value);
    const source = byID("vm-json-source");
    if (source) source.value = JSON.stringify(draft, null, 2);
    scheduleCompile();
  });
  byID("vm-schema-form")?.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-vm-remove]");
    if (remove) {
      removeAtPath(JSON.parse(decodeURIComponent(remove.dataset.vmRemove)));
      commitStructuralEdit("field removed · native validation pending");
      return;
    }
    const optional = event.target.closest("[data-vm-add-property]");
    if (optional) {
      const path = JSON.parse(decodeURIComponent(optional.dataset.vmAddProperty));
      const key = optional.dataset.vmProperty;
      const container = valueAtPath(path);
      const node = schemaAtPath(path);
      if (!container || typeof container !== "object" || Array.isArray(container) ||
          !key || key in container || !node.properties?.[key]) return;
      container[key] = defaultForSchema(node.properties[key]);
      commitStructuralEdit(`${key} added · native validation pending`);
      return;
    }
    const mapButton = event.target.closest("[data-vm-add-map]");
    if (mapButton) {
      const path = JSON.parse(decodeURIComponent(mapButton.dataset.vmAddMap));
      const container = valueAtPath(path);
      const node = schemaAtPath(path);
      const keyInput = [...document.querySelectorAll("[data-vm-map-key]")]
        .find((candidate) => candidate.dataset.vmMapKey === mapButton.dataset.vmAddMap);
      const key = keyInput?.value.trim() || "";
      const propertySchema = resolveSchema(node.propertyNames || {}, key);
      let validPattern = true;
      try { validPattern = !propertySchema.pattern || new RegExp(propertySchema.pattern).test(key); }
      catch (_) { validPattern = false; }
      if (!container || typeof container !== "object" || Array.isArray(container) ||
          !key || key in container || key.length < Number(propertySchema.minLength || 0) ||
          key.length > Number(propertySchema.maxLength || 192) || !validPattern) {
        byID("vm-editor-state").textContent = "new map key is empty, duplicate, or violates its schema";
        return;
      }
      container[key] = defaultForSchema(node.additionalProperties || {});
      commitStructuralEdit(`${key} added · native validation pending`);
      return;
    }
    const arrayButton = event.target.closest("[data-vm-add-array]");
    if (arrayButton) {
      const path = JSON.parse(decodeURIComponent(arrayButton.dataset.vmAddArray));
      const container = valueAtPath(path);
      const node = schemaAtPath(path);
      if (!Array.isArray(container) ||
          (node.maxItems !== undefined && container.length >= Number(node.maxItems))) return;
      container.push(defaultForSchema(node.items || {}));
      commitStructuralEdit("array item added · native validation pending");
    }
  });
  byID("vm-operation-composer")?.addEventListener("change", (event) => {
    if (event.target.dataset.vmOperationPresent) {
      const token = event.target.dataset.vmOperationPresent;
      const field = [...document.querySelectorAll("[data-vm-operation-field]")]
        .find((candidate) => candidate.dataset.vmOperationField === token);
      if (field) field.disabled = !event.target.checked;
      return;
    }
    if (event.target.dataset.vmOperationSlot) {
      captureOperationSlotDraft();
      renderOperationComposer();
      return;
    }
    captureOperationSlotDraft();
    operationComponentName = byID("vm-operation-component")?.value.trim() || operationComponentName;
    if (event.target.id === "vm-operation-select") {
      composerOperation = event.target.value;
      const adapter = selectedOperationDescriptor()?.key?.adapter || "component";
      operationComponentName = adapter.replace(/[^A-Za-z0-9_.:-]/g, "_")
        .replace(/^[^A-Za-z]/, "component_");
      renderOperationComposer();
    } else if (event.target.id === "vm-operation-node") {
      operationNode = event.target.value;
      renderOperationComposer();
    }
  });
  byID("vm-operation-composer")?.addEventListener("input", (event) => {
    if (event.target.id === "vm-operation-component") {
      operationComponentName = event.target.value.trim();
    }
  });
  byID("vm-operation-composer")?.addEventListener("click", (event) => {
    if (event.target.closest("#vm-operation-bind")) bindOperation();
  });
  byID("vm-component-composer")?.addEventListener("change", (event) => {
    if (event.target.dataset.vmComponentPresent) {
      const name = event.target.dataset.vmComponentPresent;
      const field = [...document.querySelectorAll("[data-vm-component-field]")]
        .find((candidate) => candidate.dataset.vmComponentField === name);
      if (field) field.disabled = !event.target.checked;
      return;
    }
    if (event.target.id === "vm-component-select" ||
        event.target.id === "vm-component-node" ||
        event.target.id === "vm-component-slot") {
      captureComposerIdentity();
      renderTrainingComposer();
    } else {
      captureComposerIdentity();
    }
  });
  byID("vm-component-composer")?.addEventListener("click", (event) => {
    if (event.target.closest("#vm-component-add")) {
      bindTrainingComponent();
    }
  });
})();
