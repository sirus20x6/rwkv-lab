// moe-mla dashboard client
// - fetches /api/runs on load
// - SSE /api/stream pushes system/processes/runs every 2s
// - click a run → fetches full train.jsonl and renders Chart.js panels
// - cross-run compare: multi-select runs, overlay selected metric

const STATE = {
  runs: [],            // latest summaries (from SSE)
  currentRun: null,    // name of run shown in main panel
  currentRunData: null,
  compareSet: new Set(),
  charts: {},
  cachedRunData: new Map(),  // name → {train, eval}
  cachedRunVersions: new Map(),
  liveSystem: null,
  liveProcesses: [],
  runFilter: "",
};

const COMPARE_COLORS = [
  "#6fa8ff", "#ff9a4a", "#c97cff", "#3fd07a",
  "#e0c341", "#ff7a7a", "#7bcbd4", "#d4a17b",
  "#a97be0", "#b4f05e",
];

// ---------- utils ----------

function $(id) { return document.getElementById(id); }

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function fmtTime(age_s) {
  if (age_s == null || isNaN(age_s)) return "—";
  if (age_s < 1) return "now";
  if (age_s < 60) return `${Math.round(age_s)}s`;
  if (age_s < 3600) return `${Math.round(age_s / 60)}m`;
  if (age_s < 86400) return `${(age_s / 3600).toFixed(1)}h`;
  return `${(age_s / 86400).toFixed(1)}d`;
}
function fmtDuration(s) {
  if (s == null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `${h}h${m.toString().padStart(2,"0")}m`;
  if (m > 0) return `${m}m${sec.toString().padStart(2,"0")}s`;
  return `${sec}s`;
}
function fmtGB(g) { return g == null ? "—" : `${g.toFixed(1)} GB`; }
function fmtNum(x, digits=3) {
  if (x == null || isNaN(x)) return "—";
  if (Math.abs(x) < 1e-4 && x !== 0) return x.toExponential(2);
  return Number(x).toFixed(digits);
}
function fmtPct(x) { return x == null ? "—" : `${(100*x).toFixed(1)}%`; }
function fmtSigned(x, digits=3) {
  if (x == null || isNaN(x)) return "—";
  const sign = x > 0 ? "+" : "";
  return `${sign}${fmtNum(x, digits)}`;
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

function runVersion(name) {
  const r = findRunSummary(name);
  if (!r) return null;
  return [
    r.last_update ?? "",
    r.num_train_events ?? 0,
    r.num_eval_events ?? 0,
    r.num_checkpoint_events ?? 0,
  ].join(":");
}

async function loadRunData(name, force = false) {
  const version = runVersion(name);
  if (!force && STATE.cachedRunData.has(name) && STATE.cachedRunVersions.get(name) === version) {
    return STATE.cachedRunData.get(name);
  }
  const data = await fetchJson(`/api/runs/${encodeURIComponent(name)}`);
  if (!data.checkpoints) data.checkpoints = [];
  STATE.cachedRunData.set(name, data);
  STATE.cachedRunVersions.set(name, version);
  return data;
}

// ---------- system header ----------

function meterColor(pct) {
  if (pct >= 90) return "err";
  if (pct >= 75) return "warn";
  return "";
}
function renderMeter(label, used, total, unit="", pctVal=null) {
  const pct = pctVal != null ? pctVal : (100 * used / total);
  const cls = meterColor(pct);
  const usedStr = unit === "%" ? `${used.toFixed(0)}%` : `${used.toFixed(1)}${unit}`;
  const totalStr = total != null ? ` / ${total.toFixed(1)}${unit}` : "";
  return `<span><span class="val">${label}</span>: ${usedStr}${totalStr}
    <span class="meter"><span class="${cls}" style="width:${Math.min(100, pct).toFixed(1)}%"></span></span></span>`;
}
function renderSystem(sys) {
  if (!sys) return;
  // GPUs
  const gpus = (sys.gpus || []).map((g, i) => {
    if (g.error) return `<span>GPU${i}: ${esc(g.error)}</span>`;
    const pwr = g.power_w != null
      ? ` · ${g.power_w.toFixed(0)}W${g.power_cap_w ? "/" + g.power_cap_w.toFixed(0) : ""}`
      : "";
    const tempCls = g.temp_c >= 85 ? "err" : g.temp_c >= 75 ? "warn" : "";
    return `<span>GPU${i} <span class="val">${g.util_pct}%</span>
      <span class="meter"><span class="${meterColor(g.util_pct)}" style="width:${g.util_pct}%"></span></span>
      · ${g.mem_used_gb.toFixed(1)}/${g.mem_total_gb.toFixed(1)}GB
      <span class="meter"><span class="${meterColor(g.mem_pct)}" style="width:${g.mem_pct}%"></span></span>
      · <span class="${tempCls ? 'val' : ''}">${g.temp_c}°C</span>${pwr}
      · ${esc(g.name || "")}</span>`;
  }).join("");
  $("sys-gpus").innerHTML = gpus || `<span class="val">no GPU</span>`;

  // Host: cpu, ram, disk
  $("sys-host").innerHTML =
    renderMeter("CPU", sys.cpu_pct, null, "%", sys.cpu_pct) +
    renderMeter("RAM", sys.ram_used_gb, sys.ram_total_gb, "GB", sys.ram_pct) +
    renderMeter("disk", sys.disk_used_gb, sys.disk_total_gb, "GB", sys.disk_pct);
}

function renderProcesses(procs) {
  if (!procs || procs.length === 0) {
    $("sys-proc").innerHTML = `<span class="val">no training processes</span>`;
    return;
  }
  const items = procs.map(p => {
    const logAge = p.log_age_s != null ? fmtTime(p.log_age_s) : "—";
    return `<span><span class="dot ${p.state}"></span>
      PID <span class="val">${p.pid}</span>
      · ${esc(p.run_name || "?")}
      · ${fmtDuration(p.runtime_s)} · ${p.rss_gb.toFixed(1)}GB · log ${logAge}</span>`;
  });
  $("sys-proc").innerHTML = items.join("");
}

function renderTs(ts) {
  if (!ts) return;
  const d = new Date(ts * 1000);
  $("sys-ts").textContent = d.toLocaleTimeString();
}

// ---------- sidebar ----------

function effectiveRunState(run) {
  // If a process is attached to this run, promote the state from that
  // — a live process on a run overrides a "cold" log-age state.
  const proc = STATE.liveProcesses.find(p => p.run_name === run.name);
  if (proc && proc.state === "healthy") return "healthy";
  if (proc && proc.state === "stalling") return "stalling";
  return run.alive_state;
}

function renderRunList() {
  const list = $("run-list");
  const compare = $("compare-list");
  list.innerHTML = "";
  compare.innerHTML = "";
  const loggedRuns = STATE.runs.filter(r => r.has_log);
  const filter = STATE.runFilter.trim().toLowerCase();
  const shownRuns = filter
    ? loggedRuns.filter(r => r.name.toLowerCase().includes(filter))
    : loggedRuns;

  const runCount = $("run-count");
  if (runCount) runCount.textContent = filter ? `${shownRuns.length}/${loggedRuns.length}` : String(loggedRuns.length);
  const compareCount = $("compare-count");
  if (compareCount) compareCount.textContent = `${STATE.compareSet.size} selected`;

  if (!shownRuns.length) {
    list.innerHTML = `<div class="empty">${filter ? "no matching runs" : "no logged runs"}</div>`;
  }

  for (const r of shownRuns) {
    const state = effectiveRunState(r);
    const item = document.createElement("div");
    item.className = "run-item" + (r.name === STATE.currentRun ? " active" : "");
    item.innerHTML = `
      <span class="dot ${state}"></span>
      <div class="run-info">
        <div class="run-name">${esc(r.name)}</div>
        <div class="run-meta">step ${r.latest_step ?? "—"} · ${fmtTime(r.last_update_age_s)} ago${r.has_horizons ? " · h=1…4" : ""}</div>
      </div>
    `;
    item.onclick = () => selectRun(r.name);
    list.appendChild(item);

    // Compare row
    const idx = [...STATE.compareSet].indexOf(r.name);
    const color = idx >= 0 ? COMPARE_COLORS[idx % COMPARE_COLORS.length] : "transparent";
    const row = document.createElement("label");
    row.className = "compare-item";
    row.innerHTML = `
      <input type="checkbox" ${STATE.compareSet.has(r.name) ? "checked" : ""} />
      <span class="swatch" style="background:${color}"></span>
      <span>${esc(r.name)}</span>
    `;
    const cb = row.querySelector("input");
    cb.onchange = () => {
      if (cb.checked) STATE.compareSet.add(r.name);
      else STATE.compareSet.delete(r.name);
      renderRunList();
      renderCompare();
    };
    compare.appendChild(row);
  }
}

// ---------- run detail ----------

async function selectRun(name) {
  STATE.currentRun = name;
  $("run-body").classList.remove("hidden");
  $("run-header").innerHTML = `<h1>${esc(name)}</h1><div class="sub">loading…</div>`;
  renderRunList();

  let data;
  try {
    data = await loadRunData(name, true);
    STATE.currentRunData = data;
  } catch (e) {
    $("run-header").innerHTML = `<h1>${esc(name)}</h1><div class="sub">error: ${esc(e.message)}</div>`;
    return;
  }

  // isolate each render: one throwing must not skip the panels after it
  const _safe = (fn, arg) => { try { fn(arg); } catch (e) { console.error("render error in", fn.name, e); } };
  _safe(renderRunHeader, name);
  _safe(renderKPIs, data);
  _safe(renderLossChart, data);
  _safe(renderLoopRw, data);
  _safe(renderHorizonsChart, data);
  _safe(renderTrainingChart, data);
  _safe(renderEvents, data);

  const ckpts = await fetchJson(`/api/runs/${encodeURIComponent(name)}/checkpoints`);
  renderCheckpoints(ckpts);

  // Architecture is fetched lazily — see arch-toggle wiring below. Reset the
  // panel state for the new run so the previous run's data isn't reused.
  STATE.currentArch = null;
  $("arch-sub").textContent = "(click 'show' to load)";
  $("arch-summary").innerHTML = "";
  $("arch-layer-list").innerHTML = "";
  $("arch-body").classList.add("hidden");
  const tBtn = $("arch-toggle");
  if (tBtn) tBtn.textContent = "show";
}

async function loadArchitecture(name) {
  const arch = await fetchJson(`/api/runs/${encodeURIComponent(name)}/architecture`);
  STATE.currentArch = arch;
  renderArchitecture(arch);
}

function fmtParams(n) {
  if (n == null) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " K";
  return String(n);
}

function trainableBadge(state) {
  // state: true | false | "trainable" | "frozen" | "partial"
  let label, cls;
  if (state === true || state === "trainable") { label = "trainable"; cls = "tr-on"; }
  else if (state === "partial") { label = "partial"; cls = "tr-partial"; }
  else { label = "frozen"; cls = "tr-off"; }
  return `<span class="tr-badge ${cls}">${label}</span>`;
}

function renderArchitecture(arch) {
  const sub = $("arch-sub");
  const body = $("arch-body");
  const summary = $("arch-summary");
  const list = $("arch-layer-list");

  if (arch.error) {
    sub.textContent = `(${arch.error})`;
    body.classList.add("hidden");
    return;
  }

  const t = arch.totals;
  const cfg = arch.config;
  const m = arch.modifications;
  sub.textContent = `${arch.model_name} · ${fmtParams(t.total_params)} total · ${fmtParams(t.trainable_params)} trainable (${t.trainable_pct.toFixed(1)}%)`;

  // Summary
  const moeBadge = arch.is_moe ? `<span class="arch-tag moe">MoE×${cfg.n_experts} (top${cfg.n_experts_per_tok})</span>` : "";
  const mlaTag = m.mla_layer_indices.length
    ? `<span class="arch-tag mla">${m.mla_layer_indices.length} MLA layers · L=${JSON.stringify(m.mla_layer_indices)}</span>`
    : "";
  const mtpTag = m.mtp_installed ? `<span class="arch-tag mtp">MTP installed</span>` : "";
  const engTag = (m.engram_layer_indices || []).length
    ? `<span class="arch-tag engram">Engram L=${JSON.stringify(m.engram_layer_indices)}</span>` : "";
  const freezeTag = `<span class="arch-tag freeze-${m.freeze_mode}">freeze=${m.freeze_mode}</span>`;

  summary.innerHTML = `
    <div class="arch-cfg">
      <span><b>${arch.model_name}</b> <i>(${arch.model_type})</i></span>
      <span>hidden=${cfg.hidden_size}</span>
      <span>layers=${cfg.num_hidden_layers}</span>
      <span>heads=${cfg.num_attention_heads}/${cfg.num_key_value_heads} (q/kv)</span>
      <span>head_dim=${cfg.head_dim}</span>
      <span>intermediate=${cfg.intermediate_size}</span>
      <span>vocab=${cfg.vocab_size.toLocaleString()}</span>
      ${cfg.attn_output_gate ? '<span class="arch-tag minor">output_gate</span>' : ''}
      ${cfg.tie_word_embeddings ? '<span class="arch-tag minor">tied embed</span>' : ''}
    </div>
    <div class="arch-mods">
      ${freezeTag}${mlaTag}${mtpTag}${engTag}${moeBadge}
    </div>
    <div class="arch-totals">
      <span class="arch-bar">
        <span class="arch-bar-fill" style="width:${t.trainable_pct.toFixed(1)}%"></span>
      </span>
      <span><b>${fmtParams(t.total_params)}</b> total</span>
      <span class="ok"><b>${fmtParams(t.trainable_params)}</b> trainable</span>
      <span class="muted"><b>${fmtParams(t.frozen_params)}</b> frozen</span>
    </div>
  `;

  // Layer list
  const rows = [];
  for (const L of arch.layers) {
    if (L.kind === "decoder_layer") {
      const a = L.attention, mlp = L.mlp;
      const isMla = !!L.is_mla;
      const isRwkv8 = !!L.is_rwkv8;
      const cls = isRwkv8 ? "row-rwkv8"
        : isMla ? "row-mla"
        : (L.layer_type === "linear_attention" ? "row-linear" : "row-fullattn");
      const headLine = `<div class="lr-head ${cls}">
        <span class="lr-idx">L${String(L.index).padStart(2, '0')}</span>
        <span class="lr-name">${L.name}</span>
        <span class="lr-type ${a.kind.toLowerCase().replace(/[^a-z]/g,'')}">${a.kind}</span>
        <span class="lr-params">${fmtParams(L.params)}</span>
        ${trainableBadge(L.trainable_state)}
        <button class="lr-expand" data-idx="${L.index}">▸</button>
      </div>`;
      const detail = `<div class="lr-detail hidden" id="lr-detail-${L.index}">
        <div class="lr-d-row"><span class="lbl">attn (${a.kind})</span><span>${fmtParams(a.params)}</span>${trainableBadge(a.trainable)}</div>
        ${a.kind === "MLA"
          ? `<div class="lr-d-sub">q_lora=${a.q_lora_rank ?? "?"} kv_lora=${a.kv_lora_rank ?? "?"} head_dim=${a.head_dim}</div>`
          : `<div class="lr-d-sub">heads=${a.n_q_heads}/${a.n_kv_heads ?? "—"} head_dim=${a.head_dim}</div>`}
        <div class="lr-d-row"><span class="lbl">mlp (${mlp.kind})</span><span>${fmtParams(mlp.params)}</span>${trainableBadge(mlp.trainable)}</div>
        <div class="lr-d-sub">intermediate=${mlp.intermediate_size}${mlp.n_experts ? ` experts=${mlp.n_experts} top${mlp.n_experts_per_tok}` : ''}</div>
        <div class="lr-d-row"><span class="lbl">norms</span><span>${fmtParams(L.norm_params)}</span></div>
        ${L.other_params ? `<div class="lr-d-row"><span class="lbl">other</span><span>${fmtParams(L.other_params)}</span></div>` : ''}
      </div>`;
      rows.push(headLine + detail);
    } else {
      const cls = `row-${L.kind}`;
      const tag = L.kind === "embedding" ? "EMBED"
        : L.kind === "lm_head" ? "HEAD"
        : L.kind === "norm" ? "NORM"
        : L.kind === "mtp" ? "MTP"
        : L.kind === "engram" ? "ENGRAM"
        : L.kind === "vision" ? "VISION"
        : L.kind === "other" ? "OTHER"
        : L.kind.toUpperCase();
      const approx = L.params_approx ? " ≈" : "";
      const expandable = (L.top_tensors && L.top_tensors.length > 0);
      const detailId = `lr-detail-${L.kind}-${(L.layer_id != null ? L.layer_id : "x")}`;
      const headLine = `<div class="lr-head ${cls}">
        <span class="lr-idx">${L.kind === "engram" ? "L" + L.layer_id : "—"}</span>
        <span class="lr-name">${L.name}${L.shape ? ` <i class="dim">${L.shape}</i>` : ""}</span>
        <span class="lr-type ${L.kind}">${tag}</span>
        <span class="lr-params">${approx}${fmtParams(L.params)}</span>
        ${trainableBadge(L.trainable)}
        ${expandable ? `<button class="lr-expand" data-target="${detailId}">▸</button>` : ""}
      </div>`;
      let detail = "";
      if (expandable) {
        const noteRow = L.note ? `<div class="lr-d-note">${L.note}</div>` : "";
        const countRow = `<div class="lr-d-sub">${L.tensors_count} tensors total — top ${L.top_tensors.length} by size:</div>`;
        const tensorRows = L.top_tensors.map(t =>
          `<div class="lr-d-trow"><code>${t.name}</code><span class="dim">${t.shape}</span><span class="num">${fmtParams(t.params)}</span></div>`
        ).join("");
        detail = `<div class="lr-detail hidden" id="${detailId}">${noteRow}${countRow}${tensorRows}</div>`;
      }
      rows.push(headLine + detail);
    }
  }
  list.innerHTML = rows.join("");

  // Wire expand buttons. Decoder rows store the layer index in data-idx,
  // non-decoder rows (vision, other, …) store the full element id in data-target.
  list.querySelectorAll(".lr-expand").forEach(btn => {
    btn.onclick = () => {
      const det = btn.dataset.target
        ? document.getElementById(btn.dataset.target)
        : document.getElementById(`lr-detail-${btn.dataset.idx}`);
      if (!det) return;
      const open = !det.classList.contains("hidden");
      det.classList.toggle("hidden", open);
      btn.textContent = open ? "▸" : "▾";
    };
  });
}

// Toggle the architecture panel body. The script tag is at the end of <body>,
// so the DOM is already parsed by the time we hit this — wire the handler
// directly. (The earlier DOMContentLoaded wrapper was a no-op because the
// event had already fired.)
{
  const tBtn = $("arch-toggle");
  if (tBtn) {
    tBtn.onclick = async (e) => {
      e.stopPropagation();
      const body = $("arch-body");
      const open = !body.classList.contains("hidden");
      body.classList.toggle("hidden", open);
      tBtn.textContent = open ? "show" : "hide";
      // Lazy fetch: only hit the API the first time the user expands the panel
      // for this run. Old runs without sidecar config.json take seconds to
      // resolve (torch.load fallback), so we don't pay the cost unless the
      // user actually wants the data.
      if (!open && !STATE.currentArch && STATE.currentRun) {
        $("arch-sub").textContent = "loading…";
        try {
          await loadArchitecture(STATE.currentRun);
        } catch (err) {
          $("arch-sub").textContent = "(unavailable)";
          console.warn("architecture fetch failed", err);
        }
      }
    };
  }
}

function findRunSummary(name) {
  return STATE.runs.find(r => r.name === name);
}

function renderRunHeader(name) {
  const r = findRunSummary(name);
  const proc = STATE.liveProcesses.find(p => p.run_name === name);
  let alive_state = r?.alive_state || "cold";
  if (proc) alive_state = proc.state;
  const stateLabel = {
    healthy: "ACTIVE", stalling: "STALLING", dead: "DEAD",
    cold: "idle", no_log: "no log", error: "error",
  }[alive_state] || alive_state;
  const pidStr = proc ? `<span class="pid">PID ${proc.pid} · ${fmtDuration(proc.runtime_s)}</span>` : "";

  let progressHtml = "";
  const step = r?.latest_step;
  const maxSteps = proc?.max_steps;
  if (step != null && maxSteps != null && maxSteps > 0) {
    const pct = Math.max(0, Math.min(100, 100 * step / maxSteps));
    const cls = pct >= 100 ? "done" : "";
    let eta = "";
    if (proc?.runtime_s && step > 0 && pct < 100) {
      const spt = proc.runtime_s / step;               // sec/step (incl. prior steps if resumed; rough)
      const etaSec = spt * (maxSteps - step);
      eta = ` · eta ${fmtDuration(etaSec)}`;
    }
    progressHtml = `
      <div class="progress-row">
        <div class="progress-bar"><span class="${cls}" style="width:${pct.toFixed(1)}%"></span></div>
        <div class="progress-label">step ${Number(step).toLocaleString()} / ${Number(maxSteps).toLocaleString()} (${pct.toFixed(1)}%)${eta}</div>
      </div>
    `;
  }

  $("run-header").innerHTML = `
    <div class="run-title-row">
      <span class="dot ${alive_state}"></span>
      <span class="run-title-main">${esc(name)}</span>
      <span class="status-pill ${alive_state}">${stateLabel}</span>
      ${pidStr}
    </div>
    <div class="sub">${r?.num_train_events ?? 0} train events · ${r?.num_eval_events ?? 0} eval events · ${r?.num_checkpoint_events ?? 0} checkpoint events · last update ${fmtTime(r?.last_update_age_s)} ago</div>
    ${progressHtml}
  `;
}

function bestEvent(events, getValue, better) {
  let best = null;
  for (const e of events) {
    const value = getValue(e);
    if (value == null || isNaN(value)) continue;
    if (!best || better(value, best.value)) best = { event: e, value };
  }
  return best;
}

function lastNAvg(events, key, n = 5) {
  const vals = events
    .map(e => e[key])
    .filter(v => v != null && !isNaN(v))
    .slice(-n);
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function setKpiSub(id, text, cls = "") {
  const el = $(id);
  el.textContent = text || "—";
  el.className = `kpi-sub${cls ? " " + cls : ""}`;
}

function top1Value(e) {
  return e?.top1_acc ?? e?.h1_top1;
}

function renderKPIs(data) {
  const t = data.train.length ? data.train[data.train.length - 1] : null;
  const e = data.eval.length ? data.eval[data.eval.length - 1] : null;
  const steps = [t?.step, e?.step].filter(v => v != null && !isNaN(v));
  const step = steps.length ? Math.max(...steps) : null;
  const bestTrainLoss = bestEvent(data.train, x => x.loss, (a, b) => a < b);
  const bestPpl = bestEvent(data.eval, x => x.ppl, (a, b) => a < b);
  const bestTop1 = bestEvent(data.eval, top1Value, (a, b) => a > b);
  const top1 = top1Value(e);
  const latestCheckpoint = data.checkpoints?.length ? data.checkpoints[data.checkpoints.length - 1] : null;
  const avgToks = lastNAvg(data.train, "tok_per_sec");

  $("kpi-step").textContent = step != null ? Number(step).toLocaleString() : "—";
  setKpiSub("kpi-step-sub", latestCheckpoint ? `last ckpt step ${latestCheckpoint.step}` : "no log ckpt");

  $("kpi-loss").textContent = t?.loss != null ? fmtNum(t.loss, 3) : "—";
  setKpiSub(
    "kpi-loss-sub",
    bestTrainLoss ? `best ${fmtNum(bestTrainLoss.value, 3)} @ ${bestTrainLoss.event.step}` : "no train data",
  );

  $("kpi-ppl").textContent  = e?.ppl != null ? fmtNum(e.ppl, 3) : "—";
  if (e?.ppl != null && bestPpl) {
    const delta = e.ppl - bestPpl.value;
    setKpiSub("kpi-ppl-sub", `${fmtSigned(delta, 3)} vs best`, delta <= 0 ? "good" : "bad");
  } else {
    setKpiSub("kpi-ppl-sub", "no eval data");
  }

  $("kpi-best-ppl").textContent = bestPpl ? fmtNum(bestPpl.value, 3) : "—";
  setKpiSub("kpi-best-ppl-sub", bestPpl ? `step ${bestPpl.event.step}` : "no eval data", bestPpl ? "good" : "");

  $("kpi-top1").textContent = top1 != null ? fmtPct(top1) : "—";
  if (top1 != null && bestTop1) {
    const deltaPp = 100 * (top1 - bestTop1.value);
    setKpiSub("kpi-top1-sub", `${fmtSigned(deltaPp, 2)} pp vs best`, deltaPp >= 0 ? "good" : "bad");
  } else {
    setKpiSub("kpi-top1-sub", "no acc data");
  }

  $("kpi-best-top1").textContent = bestTop1 ? fmtPct(bestTop1.value) : "—";
  setKpiSub("kpi-best-top1-sub", bestTop1 ? `step ${bestTop1.event.step}` : "no acc data", bestTop1 ? "good" : "");

  $("kpi-toks").textContent = t?.tok_per_sec != null ? Math.round(t.tok_per_sec).toLocaleString() : "—";
  setKpiSub("kpi-toks-sub", avgToks != null ? `${Math.round(avgToks).toLocaleString()} avg last 5` : "no train data");

  $("kpi-events").textContent = `${data.train.length}t/${data.eval.length}e`;
  setKpiSub("kpi-events-sub", `${data.checkpoints?.length ?? 0} checkpoint events`);
}

// ---------- charts ----------

const CHART_DEFAULTS = {
  animation: false,
  maintainAspectRatio: false,
  responsive: true,
  parsing: false,
  interaction: { mode: "nearest", axis: "x", intersect: false },
  plugins: {
    legend: { labels: { color: "#cfd5dc", boxWidth: 10, boxHeight: 10, font: { size: 11 } } },
    tooltip: { bodyFont: { family: "JetBrains Mono, monospace" } },
  },
  scales: {
    x: {
      type: "linear",
      title: { display: true, text: "step", color: "#7a8594", font: { size: 11 } },
      ticks: { color: "#aab1b9", font: { size: 10 } },
      grid: { color: "#262f3a" },
    },
    y: {
      ticks: { color: "#aab1b9", font: { size: 10 } },
      grid: { color: "#262f3a" },
    },
  },
};

const CHECKPOINT_LINE_PLUGIN = {
  id: "checkpointLines",
  afterDatasetsDraw(chart, _args, opts) {
    const checkpoints = opts?.checkpoints || [];
    if (!checkpoints.length) return;
    const xScale = chart.scales.x;
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.strokeStyle = "rgba(224, 195, 65, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    for (const c of checkpoints) {
      if (c.step == null) continue;
      const x = xScale.getPixelForValue(c.step);
      if (x < chartArea.left || x > chartArea.right) continue;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
    }
    ctx.restore();
  },
};

function roundedRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

const LEADER_LABEL_PLUGIN = {
  id: "leaderLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    if (!chartArea) return;

    chart.data.datasets.forEach((dataset, datasetIndex) => {
      if (!dataset.leaderLabels) return;
      const meta = chart.getDatasetMeta(datasetIndex);
      if (!meta || meta.hidden) return;

      const points = meta.data
        .map((el, i) => ({ el, raw: dataset.data[i], i }))
        .filter(p => p.raw && Number.isFinite(p.raw.x) && Number.isFinite(p.raw.y));
      if (!points.length) return;

      const maxLabels = dataset.leaderLabelMaxPoints ?? 18;
      const stride = points.length > maxLabels ? Math.ceil(points.length / maxLabels) : 1;
      const selected = points.filter((_, i) => i % stride === 0 || i === points.length - 1);

      ctx.save();
      ctx.font = "600 11px JetBrains Mono, Consolas, monospace";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 1;

      selected.forEach((p, i) => {
        const x = p.el.x;
        const y = p.el.y;
        if (x < chartArea.left || x > chartArea.right || y < chartArea.top || y > chartArea.bottom) return;

        const label = typeof dataset.leaderLabelFormatter === "function"
          ? dataset.leaderLabelFormatter(p.raw.y, p.raw, dataset)
          : fmtNum(p.raw.y, 3);
        const textW = ctx.measureText(label).width;
        const padX = 7;
        const boxW = textW + padX * 2;
        const boxH = 20;
        const rightSide = x < chartArea.right - boxW - 34;
        const dx = rightSide ? 30 : -30;
        const dy = (dataset.leaderLabelDy ?? -28) - ((i % 2) * 8);
        let boxX = rightSide ? x + dx : x + dx - boxW;
        let boxY = y + dy - boxH / 2;
        boxX = Math.max(chartArea.left + 4, Math.min(chartArea.right - boxW - 4, boxX));
        boxY = Math.max(chartArea.top + 4, Math.min(chartArea.bottom - boxH - 4, boxY));
        const anchorX = rightSide ? boxX : boxX + boxW;
        const anchorY = boxY + boxH / 2;
        const color = dataset.borderColor || "#cfd5dc";

        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.72;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(anchorX, anchorY);
        ctx.stroke();

        ctx.globalAlpha = 1;
        ctx.fillStyle = "rgba(15, 17, 19, 0.92)";
        ctx.strokeStyle = color;
        roundedRect(ctx, boxX, boxY, boxW, boxH, 5);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.fillText(label, boxX + padX, boxY + boxH / 2);
      });
      ctx.restore();
    });
  },
};

if (window.Chart) Chart.register(CHECKPOINT_LINE_PLUGIN, LEADER_LABEL_PLUGIN);

function deepMerge(a, b) {
  const out = Array.isArray(a) ? [...a] : { ...a };
  for (const k of Object.keys(b || {})) {
    if (b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a?.[k] && typeof a[k] === "object") {
      out[k] = deepMerge(a[k], b[k]);
    } else {
      out[k] = b[k];
    }
  }
  return out;
}

function setChartEmpty(id, message) {
  const el = $(id);
  const box = el?.parentElement;
  if (!box) return;
  let empty = box.querySelector(".chart-empty");
  if (message) {
    if (!empty) {
      empty = document.createElement("div");
      empty.className = "chart-empty";
      box.appendChild(empty);
    }
    empty.textContent = message;
    el.style.opacity = "0.25";
  } else {
    if (empty) empty.remove();
    el.style.opacity = "1";
  }
}

// Click a legend entry to toggle that metric; the hidden set is remembered per
// chart so the 2s re-render doesn't un-hide it.
function _legendToggle(id) {
  return function (_e, legendItem, legend) {
    const chart = legend.chart;
    const i = legendItem.datasetIndex;
    const label = chart.data.datasets[i].label;
    const vis = chart.isDatasetVisible(i);
    chart.setDatasetVisibility(i, !vis);
    STATE.hidden[id] = STATE.hidden[id] || new Set();
    if (vis) STATE.hidden[id].add(label); else STATE.hidden[id].delete(label);
    chart.update();
  };
}

// Scroll wheel = horizontal (x-axis) zoom centered on the cursor; double-click =
// reset. The listener lives on the persistent canvas DOM node (attached once) and
// looks up the current chart each event, so it survives chart re-creation. The
// zoom range is saved in STATE.zoom[id] and re-applied by mkChart on every render.
function _attachWheelZoom(id, canvas) {
  if (canvas._wheelZoom) return;
  canvas._wheelZoom = true;
  canvas.title = "scroll: zoom x · double-click: reset · click legend: toggle metric";
  canvas.addEventListener("wheel", (e) => {
    const chart = STATE.charts[id];
    if (!chart || !chart.scales.x) return;
    e.preventDefault();
    const xs = chart.scales.x;
    const at = xs.getValueForPixel(e.clientX - canvas.getBoundingClientRect().left);
    const min = xs.min, max = xs.max, range = max - min;
    if (at == null || !isFinite(at) || !isFinite(range) || range <= 0) return;
    const factor = e.deltaY < 0 ? 0.82 : 1 / 0.82;   // up = zoom in, down = zoom out
    const nr = range * factor, frac = (at - min) / range;
    STATE.zoom[id] = { min: at - frac * nr, max: at + (1 - frac) * nr };
    chart.options.scales.x.min = STATE.zoom[id].min;
    chart.options.scales.x.max = STATE.zoom[id].max;
    chart.update("none");
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    delete STATE.zoom[id];
    const chart = STATE.charts[id];
    if (chart && chart.options.scales && chart.options.scales.x) {
      chart.options.scales.x.min = undefined;
      chart.options.scales.x.max = undefined;
      chart.update("none");
    }
  });
}

function mkChart(id, datasets, yAxes = { y: {} }, extraOptions = {}) {
  if (STATE.charts[id]) STATE.charts[id].destroy();
  STATE.zoom = STATE.zoom || {};
  STATE.hidden = STATE.hidden || {};
  const el = $(id);
  if (!el) return null;
  const ctx = el.getContext("2d");
  const scales = { x: CHART_DEFAULTS.scales.x };
  for (const [k, v] of Object.entries(yAxes)) {
    scales[k] = deepMerge(CHART_DEFAULTS.scales.y, v);
  }
  const points = datasets.flatMap(d => d.data || []);
  const xs = [...new Set(points.map(p => p.x).filter(v => v != null && isFinite(v)))];
  if (xs.length === 0) {
    scales.x = deepMerge(scales.x, { min: 0, max: 1 });
    setChartEmpty(id, "no data for this metric yet");
  } else {
    setChartEmpty(id, "");
    if (xs.length === 1) {
      const x = xs[0];
      scales.x = deepMerge(scales.x, { min: Math.max(0, x - 1), max: x + 1 });
    }
  }
  // re-apply a persisted wheel-zoom x-range (cleared on double-click)
  if (STATE.zoom[id] && xs.length > 0) {
    scales.x = deepMerge(scales.x, { min: STATE.zoom[id].min, max: STATE.zoom[id].max });
  }

  const opts = deepMerge(deepMerge(CHART_DEFAULTS, { scales }), extraOptions);
  opts.plugins = Object.assign({}, opts.plugins);
  opts.plugins.legend = Object.assign({}, opts.plugins.legend, { onClick: _legendToggle(id) });
  const chart = new Chart(ctx, { type: "line", data: { datasets }, options: opts });
  STATE.charts[id] = chart;
  // re-apply persisted metric (legend) visibility across the re-render
  if (STATE.hidden[id] && STATE.hidden[id].size) {
    chart.data.datasets.forEach((ds, i) => {
      if (STATE.hidden[id].has(ds.label)) chart.setDatasetVisibility(i, false);
    });
    chart.update("none");
  }
  _attachWheelZoom(id, el);
  return chart;
}

function checkpointChartOptions(data) {
  return {
    plugins: {
      checkpointLines: { checkpoints: data.checkpoints || [] },
    },
  };
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function lineDataset(label, color, points, extra = {}) {
  return {
    label,
    borderColor: color,
    backgroundColor: extra.backgroundColor ?? hexToRgba(color, extra.fill ? 0.16 : 1),
    pointBackgroundColor: extra.pointBackgroundColor ?? color,
    pointBorderColor: extra.pointBorderColor ?? "#0f1113",
    data: points,
    pointRadius: extra.pointRadius ?? 0,
    pointHitRadius: extra.pointHitRadius ?? 16,
    pointHoverRadius: extra.pointHoverRadius ?? 9,
    pointBorderWidth: extra.pointBorderWidth ?? 1,
    pointHoverBorderWidth: extra.pointHoverBorderWidth ?? 3,
    borderWidth: 1.6,
    tension: 0,
    fill: extra.fill ?? false,
    spanGaps: true,
    ...extra,
  };
}

function xy(e, yKey) {
  const y = e[yKey];
  if (y == null || isNaN(y)) return null;
  return { x: e.step, y };
}
function xyList(events, yKey) {
  return events.map(e => xy(e, yKey)).filter(p => p != null);
}

// Drop far-outlier points from a series for display. Real training loss spikes
// (a hard batch, a fresh layer swap) are finite and unflagged, so they pass the
// skipped/non-finite filter; a single 12.0 point then rescales the whole y-axis
// and flattens the real curve. Use a robust median + k*MAD fence so this trims
// ONLY genuine outliers — when the curve is smooth, MAD is small relative to the
// spread and nothing is dropped. spanGaps:true bridges the resulting gaps.
function trimSpikes(points, k = 6, maxFrac = 0.02) {
  const ys = points.map(p => p.y).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (ys.length < 8) return { kept: points, dropped: 0, fence: null };
  const med = ys[Math.floor(ys.length / 2)];
  const devs = ys.map(v => Math.abs(v - med)).sort((a, b) => a - b);
  const mad = devs[Math.floor(devs.length / 2)];
  // robust sigma (1.4826*MAD ≈ std for normal); fall back to a multiplicative
  // fence if MAD collapses to 0 (e.g. many identical values).
  const sigma = mad > 0 ? 1.4826 * mad : med * 0.5;
  const madFence = med + k * sigma;
  // Never hide more than maxFrac of points: on a genuinely noisy run the MAD
  // fence cuts into the distribution's real tail. Back off to the (1-maxFrac)
  // quantile so at most the most extreme ~2% are ever treated as spikes. The
  // higher fence = fewer drops, so this is max(), not min().
  const qFence = ys[Math.min(ys.length - 1, Math.floor(ys.length * (1 - maxFrac)))];
  const fence = Math.max(madFence, qFence);
  const kept = points.filter(p => !(Number.isFinite(p.y) && p.y > fence));
  return { kept, dropped: points.length - kept.length, fence };
}

function renderLossChart(data) {
  // Skipped steps (spike-guard / non-finite) carry junk loss values (up to 1e12)
  // that destroy the y-axis. Drop them so the loss curve reflects real steps.
  const cleanTrain = data.train.filter(t => !t.skipped && Number.isFinite(t.loss));
  const trainTrim = trimSpikes(xyList(cleanTrain, "loss"));
  const train = trainTrim.kept;
  // The base model is the explicit FIRST eval data point (replacing the meaningless
  // step-0 raw-swap reading), so the curve is anchored to the original model and
  // training reads directly against it. (data.baseline injected by app.py.)
  let evalRows = data.eval;
  if (data.baseline && data.baseline.ppl != null && data.eval.length) {
    const firstStep = Math.min(...data.eval.map(e => e.step));
    evalRows = [{ step: firstStep, ppl: data.baseline.ppl, loss: data.baseline.loss,
                  top1_acc: data.baseline.top1_acc, baseline: true },
                ...data.eval.filter(e => e.step > firstStep)];
  }
  const evalLoss = xyList(evalRows, "loss");
  const evalPpl = trimSpikes(xyList(evalRows, "ppl")).kept;
  const hasPpl = evalPpl.length > 0;

  const datasets = [
    lineDataset("train loss", "#6fa8ff", train, { pointRadius: 1.5, pointHitRadius: 4, pointHoverRadius: 4, fill: "origin", backgroundColor: hexToRgba("#6fa8ff", 0.14) }),
    lineDataset("eval loss", "#ff9a4a", evalLoss, {
      pointRadius: 8,
      pointHoverRadius: 11,
      pointHitRadius: 24,
      borderWidth: 2,
      fill: "origin",
      backgroundColor: hexToRgba("#ff9a4a", 0.12),
      leaderLabels: true,
      leaderLabelDy: -24,
      leaderLabelFormatter: v => `loss ${fmtNum(v, 3)}`,
    }),
  ];
  if (hasPpl) {
    datasets.push(lineDataset("eval ppl", "#c97cff", evalPpl, {
      pointRadius: 8,
      pointHoverRadius: 11,
      pointHitRadius: 24,
      borderWidth: 2,
      yAxisID: "y1",
      borderDash: [4, 3],
      leaderLabels: true,
      leaderLabelDy: -52,
      leaderLabelFormatter: v => `ppl ${fmtNum(v, 2)}`,
    }));
    // Original untouched-model ppl as a fixed reference line across the x-range,
    // so every eval is read against the original (data.baseline from app.py).
    if (data.baseline && data.baseline.ppl != null) {
      // min/max WITHOUT spread: data.train can be 50k+ points, and Math.min(...arr)
      // overflows the call stack at that size — it threw here and killed every
      // render after renderLossChart (incl. the loop-utilization panel).
      let xmin = Infinity, xmax = -Infinity;
      for (const arr of [data.train, data.eval]) for (const e of arr) {
        const s = e.step;
        if (s != null && isFinite(s)) { if (s < xmin) xmin = s; if (s > xmax) xmax = s; }
      }
      if (!isFinite(xmin)) { xmin = 0; xmax = 1; }
      datasets.push(lineDataset(`original 9B (ppl ${fmtNum(data.baseline.ppl, 2)})`, "#3ecf8e",
        [{ x: xmin, y: data.baseline.ppl }, { x: xmax, y: data.baseline.ppl }],
        { yAxisID: "y1", borderDash: [8, 4], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 0, fill: false }));
    }
  }

  const yAxes = {
    y: {
      title: { display: true, text: trainTrim.dropped ? `loss (${trainTrim.dropped} spikes hidden)` : "loss", color: "#7a8594", font: { size: 11 } },
      beginAtZero: false,
      grace: "10%",
    },
  };
  if (hasPpl) {
    // PPL = exp(eval_loss). evalPpl is already spike-trimmed above, so include the
    // baseline and use suggested bounds (10% grace fallback) — no hard clamp, so a
    // real point is never clipped and the original-model line stays in range.
    const pplVals = evalPpl.map(p => p.y).filter(v => isFinite(v) && v > 0);
    if (data.baseline && data.baseline.ppl != null) pplVals.push(data.baseline.ppl);
    const pplMin = pplVals.length ? Math.min(...pplVals) : undefined;
    const pplMax = pplVals.length ? Math.max(...pplVals) : undefined;
    yAxes.y1 = {
      position: "right",
      title: { display: true, text: "ppl", color: "#7a8594", font: { size: 11 } },
      grid: { drawOnChartArea: false },
      beginAtZero: false,
      grace: "10%",
      ...(pplMin !== undefined && pplMax !== undefined && pplMax / pplMin < 5
        ? { suggestedMin: pplMin * 0.95, suggestedMax: pplMax * 1.05 }
        : {}),
    };
  }
  mkChart("chart-loss", datasets, yAxes, checkpointChartOptions(data));
}

function renderHorizonsChart(data) {
  const panel = $("horizons-panel");
  const hasHorizons = data.eval.some(e => "h4_top1" in e);
  if (!hasHorizons) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  const colors = ["#6fa8ff", "#3fd07a", "#e0c341", "#ff7a7a"];
  const datasets = [];
  for (let h = 1; h <= 4; h++) {
    const top1 = xyList(data.eval, `h${h}_top1`);
    const top5 = xyList(data.eval, `h${h}_top5`);
    if (top1.length) datasets.push(lineDataset(
      `h=${h} top-1`, colors[h-1], top1,
      { pointRadius: 8, pointHoverRadius: 11, pointHitRadius: 22, borderWidth: 2, fill: h === 1 ? "origin" : false, backgroundColor: hexToRgba(colors[h-1], h === 1 ? 0.12 : 1) }
    ));
    if (top5.length) datasets.push(lineDataset(
      `h=${h} top-5`, colors[h-1], top5,
      { pointRadius: 6, pointHoverRadius: 10, pointHitRadius: 20, borderWidth: 1.5, borderDash: [4, 3] }
    ));
  }

  mkChart("chart-horizons", datasets, {
    y: {
      title: { display: true, text: "accuracy", color: "#7a8594", font: { size: 11 } },
      min: 0, max: 1,
      ticks: { callback: v => `${(v*100).toFixed(0)}%` },
    },
  }, checkpointChartOptions(data));
}

function renderTrainingChart(data) {
  const tps = xyList(data.train, "tok_per_sec");
  const gn = xyList(data.train, "gnorm");
  const lr = xyList(data.train, "lr");

  const datasets = [];
  if (tps.length) datasets.push(lineDataset("tok/sec", "#7bcbd4", tps, {
    yAxisID: "y",
    borderWidth: 1.8,
    pointRadius: 1,
    pointHitRadius: 6,
    pointBorderWidth: 0,
    pointHoverBorderWidth: 2,
    fill: "origin",
    backgroundColor: hexToRgba("#7bcbd4", 0.10),
  }));
  if (gn.length)  datasets.push(lineDataset("gnorm",   "#f0b783", gn,  {
    yAxisID: "y1",
    borderWidth: 2,
    pointRadius: 1,
    pointHitRadius: 7,
    pointBorderWidth: 0,
    pointHoverBorderWidth: 2,
    fill: false,
  }));
  if (lr.length)  datasets.push(lineDataset("lr",      "#c695ff", lr,  {
    yAxisID: "y2",
    borderWidth: 2.2,
    borderDash: [3, 3],
    pointRadius: 1,
    pointHitRadius: 7,
    pointBorderWidth: 0,
    pointHoverBorderWidth: 2,
  }));

  mkChart("chart-training", datasets, {
    y:  { title: { display: true, text: "tok/sec", color: "#7bcbd4", font: { size: 11 } } },
    y1: {
      // gnorm spans ~0.1 to >1000 (rare pre-clip spikes); a linear axis crushes
      // the healthy 0.3-0.6 baseline into a flat line. Log scale shows both the
      // baseline wiggle and the spikes — this IS the gradient-health monitor, so
      // unlike the loss chart we want the spikes visible, not hidden.
      position: "right", type: "logarithmic",
      title: { display: true, text: "gnorm (log)", color: "#d4a17b", font: { size: 11 } },
      grid: { drawOnChartArea: false },
      ticks: { callback: v => (v >= 1000 ? `${v / 1000}k` : v >= 1 ? String(v) : v.toFixed(2)) },
    },
    y2: { position: "right", display: false, title: { display: false } },
  }, checkpointChartOptions(data));
}

function renderLoopRw(data) {
  // LoopedRWKV residual_weight per converted layer. 0 = the loop collapsed to a
  // single pass; non-zero = the loop is actually contributing. Amber bars are
  // pinned on the bf16 grid near 0.25 (the early, most-trained layers).
  const panel = $("looprw-panel");
  const lr = data.loop_rw;
  const chartBox = panel ? panel.querySelector(".chart-box") : null;
  if (!lr || !lr.layers || !lr.layers.length || (lr.loop_count || 1) <= 1) {
    // No loop data: keep the panel VISIBLE with an explanation rather than vanishing.
    // Single-layer isolation runs are bare cores — loops only exist on assembled /
    // consolidated models (loop_conv). Silently hiding it reads as "the card is broken".
    if (panel) {
      panel.classList.remove("hidden");
      const sub = $("looprw-sub");
      if (sub) sub.textContent = "· n/a — single-layer run; loop usage shows on assembled/consolidated models (e.g. loop_conv)";
      if (chartBox) chartBox.style.display = "none";
      if (STATE.charts["chart-looprw"]) { STATE.charts["chart-looprw"].destroy(); delete STATE.charts["chart-looprw"]; }
    }
    return;
  }
  panel.classList.remove("hidden");
  if (chartBox) chartBox.style.display = "";
  const rows = [...lr.layers].sort((a, b) => a.layer - b.layer);
  const labels = rows.map(r => `L${r.layer}`);
  const vals = rows.map(r => r.max_rw);
  const colors = rows.map(r => r.max_rw >= 0.245 ? "#e0c341" : "#6fa8ff");
  const sub = $("looprw-sub");
  if (sub) sub.textContent = `· mean ${(lr.mean_max_rw || 0).toFixed(3)} · ${lr.n_pinned || 0}/${lr.n_layers || rows.length} pinned ~0.25 · ${lr.loop_count} passes`;

  if (STATE.charts["chart-looprw"]) STATE.charts["chart-looprw"].destroy();
  const el = $("chart-looprw");
  if (!el) return;
  STATE.charts["chart-looprw"] = new Chart(el.getContext("2d"), {
    type: "bar",
    data: { labels, datasets: [{ label: "max |residual_weight|", data: vals, backgroundColor: colors, borderWidth: 0 }] },
    options: {
      animation: false, maintainAspectRatio: false, responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => {
          const r = rows[c.dataIndex];
          return `L${r.layer}: max|rw|=${r.max_rw.toFixed(4)}  rw=[${(r.rw || []).map(x => x.toFixed(3)).join(", ")}]`;
        } } },
      },
      scales: {
        x: { ticks: { color: "#aab1b9", font: { size: 10 } }, grid: { display: false } },
        y: { title: { display: true, text: "max |rw|  (0 = single-pass)", color: "#7a8594", font: { size: 11 } },
             beginAtZero: true, ticks: { color: "#aab1b9", font: { size: 10 } }, grid: { color: "#262f3a" } },
      },
    },
  });
}

function renderEvents(data) {
  const list = $("event-list");
  const checkpoints = (data.checkpoints || []).slice().reverse();
  if (!checkpoints.length) {
    list.innerHTML = `<div class="empty">no checkpoint events in train log</div>`;
    return;
  }
  list.innerHTML = checkpoints.slice(0, 20).map(c => `
    <div class="event">
      <span class="step">step ${c.step ?? "—"}</span>
      <span class="detail">checkpoint${c.reason ? " · " + esc(c.reason) : ""}</span>
    </div>
  `).join("");
}

function renderCheckpoints(ckpts) {
  const list = $("ckpt-list");
  if (!ckpts || ckpts.length === 0) {
    list.innerHTML = `<div class="empty">no checkpoints saved yet</div>`;
    return;
  }
  list.innerHTML = ckpts.map(c => `
    <div class="ckpt">
      <span class="name">${esc(c.name)}</span>
      <span class="size">${c.size_gb.toFixed(2)} GB</span>
      <span class="age">${fmtTime(c.age_s)} ago</span>
    </div>
  `).join("");
}

// ---------- compare ----------

function getMetric(events, path) {
  // path is "train.loss" or "eval.h3_top1"
  const [kind, key] = path.split(".");
  return events[kind === "train" ? "train" : "eval"]
    .map(e => xy(e, key))
    .filter(p => p != null);
}

// Simple exponential smoothing for compare chart
function smoothPoints(points, alpha = 0.15) {
  if (!points.length) return points;
  const out = [];
  let ema = points[0].y;
  for (const p of points) {
    ema = alpha * p.y + (1 - alpha) * ema;
    out.push({ x: p.x, y: ema });
  }
  return out;
}

async function renderCompare() {
  const box = $("compare-box");
  const names = [...STATE.compareSet];
  if (names.length === 0) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");

  const metric = $("compare-metric").value;
  const smooth = $("compare-smooth").checked;
  $("compare-metric-label").textContent = metric;

  // Fetch data for each compare run (cached)
  const all = await Promise.all(names.map(async n => {
    try {
      return { name: n, data: await loadRunData(n) };
    } catch {
      return { name: n, data: { train: [], eval: [], checkpoints: [] } };
    }
  }));

  const isAcc = /top\d|acc/.test(metric);
  const datasets = all.map(({ name, data }, i) => {
    let pts = getMetric(data, metric);
    if (smooth) pts = smoothPoints(pts);
    return lineDataset(
      name,
      COMPARE_COLORS[i % COMPARE_COLORS.length],
      pts,
      { pointRadius: /eval/.test(metric) ? 8 : 1.5, pointHoverRadius: /eval/.test(metric) ? 11 : 4, pointHitRadius: /eval/.test(metric) ? 24 : 5, borderWidth: 1.8 },
    );
  });

  const yScale = {
    title: { display: true, text: metric, color: "#7a8594", font: { size: 11 } },
  };
  if (isAcc) {
    yScale.min = 0;
    yScale.max = 1;
    yScale.ticks = { callback: v => `${(v*100).toFixed(0)}%` };
  }
  mkChart("chart-compare", datasets, { y: yScale });
}

// ---------- SSE / live updates ----------

function applySSE(payload) {
  if (payload.system) {
    STATE.liveSystem = payload.system;
    renderSystem(payload.system);
  }
  if (payload.processes) {
    STATE.liveProcesses = payload.processes;
    renderProcesses(payload.processes);
  }
  if (payload.ts) renderTs(payload.ts);

  if (payload.runs) {
    STATE.runs = payload.runs;
    for (const name of STATE.compareSet) {
      const version = runVersion(name);
      if (STATE.cachedRunData.has(name) && STATE.cachedRunVersions.get(name) !== version) {
        STATE.cachedRunData.delete(name);
      }
    }
    renderRunList();
    if (STATE.currentRun) {
      // Update header liveness (but don't refetch full data)
      renderRunHeader(STATE.currentRun);
      const version = runVersion(STATE.currentRun);
      if (STATE.currentRunData && STATE.cachedRunVersions.get(STATE.currentRun) !== version) refreshCurrent();
    }
  }
}

let refreshPending = false;
async function refreshCurrent() {
  if (refreshPending || !STATE.currentRun) return;
  refreshPending = true;
  try {
    const data = await loadRunData(STATE.currentRun, true);
    STATE.currentRunData = data;
    renderKPIs(data);
    renderLossChart(data);
    renderLoopRw(data);
    renderHorizonsChart(data);
    renderTrainingChart(data);
    renderEvents(data);
    // If this run is in compare, refresh compare too
    if (STATE.compareSet.has(STATE.currentRun)) renderCompare();
  } catch (e) {
    console.warn("refreshCurrent failed:", e);
  } finally {
    refreshPending = false;
  }
}

function startSSE() {
  const es = new EventSource("/api/stream");
  es.addEventListener("tick", ev => {
    try { applySSE(JSON.parse(ev.data)); } catch (e) { console.warn(e); }
  });
  es.addEventListener("error", () => {
    // Browser auto-reconnects; no-op
  });
}

// ---------- init ----------

async function init() {
  try {
    STATE.runs = await fetchJson("/api/runs");
  } catch (e) {
    console.warn("failed to load /api/runs:", e);
  }
  renderRunList();
  // Default: auto-select the most recently-updated run
  const active = STATE.runs.find(r => r.has_log);
  if (active) selectRun(active.name);

  $("compare-metric").onchange = renderCompare;
  $("compare-smooth").onchange = renderCompare;
  const search = $("run-search");
  if (search) {
    search.oninput = () => {
      STATE.runFilter = search.value;
      renderRunList();
    };
  }

  startSSE();
}

init();
