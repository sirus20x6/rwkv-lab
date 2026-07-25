// trainboard front-end glue.
//
// Datastar (loaded as a module in index.html) drives the reactive shell: it
// opens /api/stream on load and morphs the system header, run list, and
// selected-run header in, plus patches the $kpi / $runVersions signals.
//
// This file is the imperative Pixi side. Phase 4 fills in pixi-glue with the
// real charts; for now it's a stub that waits for the active-run node so the
// wiring is in place.

(function () {
  "use strict";

  // The hidden #active-run node carries data-run + data-v; Datastar updates it
  // every tick. Phase 4's chart layer observes it to drive incremental append.
  function watchActiveRun(cb) {
    let last = "";
    const fire = () => {
      // Datastar's morph may replace the node rather than mutate the original.
      // Resolve it each time so the observer never remains attached to a
      // detached element (which previously froze charts until a run switch).
      const node = document.getElementById("active-run");
      if (!node) return;
      const key = (node.dataset.run || "") + ":" + (node.dataset.v || "");
      if (key !== last) {
        last = key;
        cb(node.dataset.run || "", parseInt(node.dataset.v || "0", 10));
      }
    };
    new MutationObserver(fire).observe(document.body, {
      attributes: true, childList: true, subtree: true,
      attributeFilter: ["data-run", "data-v"],
    });
    // Mutation delivery can be missed while a tab is suspended or while
    // Datastar replaces a larger ancestor. This cheap fallback makes the
    // hidden run/version marker self-healing without requiring a run switch.
    setInterval(fire, 1000);
    fire();
  }

  // Exposed for pixi-glue.js: subscribe to (run, version) changes that Datastar
  // publishes by morphing the hidden #active-run node every stream tick.
  let evalSampleRequest = 0;
  let evalRenderedKey = "";
  let evalRenderedSignature = "";

  const svgNS = "http://www.w3.org/2000/svg";

  function layoutEvalBoxOverlay(card) {
    const visual = card.querySelector('[data-eval-role="visual"]');
    const image = card.querySelector('[data-eval-role="image"]');
    const overlay = card.querySelector('[data-eval-role="box-overlay"]');
    if (!visual || !image || !overlay || !image.naturalWidth || !image.naturalHeight) return;
    const width = image.clientWidth, height = image.clientHeight;
    if (!width || !height) return;
    const naturalRatio = image.naturalWidth / image.naturalHeight;
    const boxRatio = width / height;
    let drawnWidth, drawnHeight, left, top;
    if (naturalRatio > boxRatio) {
      drawnWidth = width; drawnHeight = width / naturalRatio;
      left = 0; top = (height - drawnHeight) / 2;
    } else {
      drawnHeight = height; drawnWidth = height * naturalRatio;
      top = 0; left = (width - drawnWidth) / 2;
    }
    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${drawnWidth}px`;
    overlay.style.height = `${drawnHeight}px`;
  }

  function renderEvalBoxOverlay(card, item) {
    const overlay = card.querySelector('[data-eval-role="box-overlay"]');
    const legend = card.querySelector('[data-eval-role="box-legend"]');
    const target = Array.isArray(item.target_boxes) ? item.target_boxes : [];
    const predicted = Array.isArray(item.predicted_boxes) ? item.predicted_boxes : [];
    if (!overlay || !legend) return;
    overlay.replaceChildren();
    for (const [kind, boxes] of [["target", target], ["predicted", predicted]]) {
      for (const box of boxes) {
        const rect = document.createElementNS(svgNS, "rect");
        rect.setAttribute("x", String(box.x1)); rect.setAttribute("y", String(box.y1));
        rect.setAttribute("width", String(box.x2 - box.x1));
        rect.setAttribute("height", String(box.y2 - box.y1));
        rect.setAttribute("vector-effect", "non-scaling-stroke");
        rect.classList.add(`eval-box-${kind}`);
        const title = document.createElementNS(svgNS, "title");
        title.textContent = `${kind === "target" ? "target" : "model"}${box.label ? ` · ${box.label}` : ""}`;
        rect.append(title); overlay.append(rect);
      }
    }
    legend.hidden = target.length + predicted.length === 0;
    legend.innerHTML = `<span class="target">target ${target.length}</span><span class="predicted">model ${predicted.length}</span>`;
    const image = card.querySelector('[data-eval-role="image"]');
    if (image && !image.dataset.boxLayoutBound) {
      image.dataset.boxLayoutBound = "1";
      image.addEventListener("load", () => layoutEvalBoxOverlay(card));
    }
    requestAnimationFrame(() => layoutEvalBoxOverlay(card));
  }

  function renderEvalItems(body, items) {
    const existing = Array.from(body.querySelectorAll(":scope > article.eval-sample"));
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      let card = existing[i];
      if (!card) {
        card = document.createElement("article"); card.className = "eval-sample";
        const visual = document.createElement("div");
        visual.className = "eval-sample-visual"; visual.dataset.evalRole = "visual";
        const img = document.createElement("img");
        img.loading = "lazy"; img.alt = "held-out eval image"; img.dataset.evalRole = "image";
        const overlay = document.createElementNS(svgNS, "svg");
        overlay.classList.add("eval-box-overlay"); overlay.dataset.evalRole = "box-overlay";
        overlay.setAttribute("viewBox", "0 0 999 999");
        overlay.setAttribute("preserveAspectRatio", "none");
        overlay.setAttribute("aria-hidden", "true");
        const legend = document.createElement("div");
        legend.className = "eval-box-legend"; legend.dataset.evalRole = "box-legend";
        legend.hidden = true;
        visual.append(img, overlay, legend);
        const copy = document.createElement("div"); copy.className = "eval-sample-copy";
        for (const [tag, role, cls] of [
          ["h3", "prompt-heading", ""], ["p", "prompt", "prompt"],
          ["h3", "generated-heading", ""], ["p", "generated", "generated"],
          ["h3", "reference-heading", ""], ["p", "reference", "reference"],
        ]) {
          const node = document.createElement(tag);
          node.dataset.evalRole = role;
          if (cls) node.className = cls;
          copy.append(node);
        }
        card.append(visual, copy);
        body.append(card);
      }
      card.dataset.evalIndex = String(i);
      const image = card.querySelector('[data-eval-role="image"]');
      if (image && image.getAttribute("src") !== item.image_url) image.src = item.image_url;
      const promptH = card.querySelector('[data-eval-role="prompt-heading"]');
      const prompt = card.querySelector('[data-eval-role="prompt"]');
      if (promptH) { promptH.textContent = "task prompt"; promptH.hidden = !item.prompt; }
      if (prompt) { prompt.textContent = item.prompt || ""; prompt.hidden = !item.prompt; }
      const genH = card.querySelector('[data-eval-role="generated-heading"]');
      const gen = card.querySelector('[data-eval-role="generated"]');
      const refH = card.querySelector('[data-eval-role="reference-heading"]');
      const ref = card.querySelector('[data-eval-role="reference"]');
      if (genH) genH.textContent = `model caption · ${item.tokens} tokens${item.stopped_at_eod ? " · EOD" : (item.complete === false ? " · generating" : " · capped")}`;
      if (gen) gen.textContent = item.caption || "(empty caption)";
      if (refH) refH.textContent = `reference · ${item.source || "unknown"}`;
      if (ref) ref.textContent = item.reference || "";
      renderEvalBoxOverlay(card, item);
    }
    for (let i = items.length; i < existing.length; i++) existing[i].remove();
    const empty = body.querySelector(":scope > .empty");
    if (items.length && empty) empty.remove();
    if (!items.length) {
      if (!empty) body.innerHTML = '<div class="empty">snapshot contains no images</div>';
      else empty.textContent = "snapshot contains no images";
    }
  }

  function resetEvalSamples(run) {
    evalSampleRequest++;
    evalRenderedKey = "";
    evalRenderedSignature = "";
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (title) title.textContent = "eval captions";
    if (meta) meta.textContent = run
      ? `${run} · waiting for a qualitative eval snapshot`
      : "waiting for a qualitative eval snapshot";
    if (body) body.innerHTML = '<div class="empty">waiting for a caption snapshot…</div>';
  }
  async function openEvalSamples(run, step, ppl, attempt = 0) {
    // restart-before-eval can spend well over ten seconds reloading RADIO before
    // the trainer publishes the caption skeleton. Keep checking in the
    // background, but slow down after the first few attempts because many
    // scalar-only eval points will intentionally never get a gallery.
    const exactRetryLimit = 90;
    const exactRetryDelay = attempt < 5 ? 2000 : 5000;
    const request = ++evalSampleRequest;
    const panel = document.getElementById("eval-sample-inline");
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (!panel || !body) return;
    const snapshotKey = `${run}:${Number(step)}:${Number.isFinite(ppl) ? Number(ppl).toPrecision(12) : ""}`;
    const changingSnapshot = snapshotKey !== evalRenderedKey;
    if (changingSnapshot) {
      evalRenderedKey = snapshotKey;
      evalRenderedSignature = "";
      // Keep the previous completed gallery visible until a newer artifact
      // actually exists. Scalar PPL can run more often than qualitative eval,
      // and a scalar-only marker must not blank the most recent captions.
      if (!body.querySelector(":scope > article.eval-sample")) {
        if (title) title.textContent = "eval captions · step " + Number(step).toLocaleString();
        if (meta) meta.textContent = run + (Number.isFinite(ppl) ? " · ppl " + Number(ppl).toFixed(3) : "");
        body.innerHTML = '<div class="empty">loading qualitative snapshot…</div>';
      }
    }
    const url = `/api/runs/${encodeURIComponent(run)}/eval-samples/${encodeURIComponent(step)}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      let res = await fetch(url, { signal: controller.signal, cache: "no-store" });
      let fallback = false;
      if (res.status === 404 && attempt < exactRetryLimit) {
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl, attempt + 1);
        }, exactRetryDelay);
      }
      if (res.status === 404 && !body.querySelector(":scope > article.eval-sample")) {
        // PPL is commonly measured more often than captions are generated.
        // Fill a freshly loaded/reset card immediately with the newest real
        // gallery, while the retries above continue looking for an exact-step
        // artifact that may only just be starting.
        const latestURL = `/api/runs/${encodeURIComponent(run)}/eval-samples/latest?at_step=${encodeURIComponent(step)}`;
        const latest = await fetch(latestURL, { signal: controller.signal, cache: "no-store" });
        if (latest.ok) {
          res = latest;
          fallback = true;
        } else if (attempt < exactRetryLimit) {
          body.innerHTML = '<div class="empty">caption snapshot is starting…</div>';
          return;
        }
      } else if (res.status === 404) {
        return; // scalar-only eval marker; retain the newest qualitative card
      }
      if (!res.ok) throw new Error(res.status === 404
        ? "No caption snapshot was recorded for this older eval. New eval markers will include one."
        : `snapshot request failed (${res.status})`);
      const data = await res.json();
      if (request !== evalSampleRequest) return;
      // Recovery can re-evaluate the same step while the prior generation's
      // complete artifact still exists.  The scalar eval is logged before the
      // new caption skeleton is published, so a blind 200 response would make
      // us display old captions forever (complete=true stops normal polling).
      const artifactStep = Number(data.step), artifactPPL = Number(data.ppl);
      const expectedStep = Number(step), expectedPPL = Number(ppl);
      const pplMismatch = Number.isFinite(expectedPPL) && Number.isFinite(artifactPPL) &&
        Math.abs(artifactPPL - expectedPPL) > 1e-9 * Math.max(1, Math.abs(expectedPPL));
      const stale = !fallback && (artifactStep !== expectedStep || pplMismatch);
      if (stale && attempt < 5) {
        if (meta) meta.textContent = `${run} · ppl ${expectedPPL.toFixed(3)} · replacing stale same-step snapshot`;
        body.innerHTML = '<div class="empty">waiting for this eval generation’s captions…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl, attempt + 1);
        }, 2000);
        return;
      }
      // Retries exhausted on a permanently stale artifact (e.g. dead run whose
      // recovery never republished): show what exists, flagged, and stop polling.
      const pending = data.complete === false && !stale;
      if (title) title.textContent = "eval captions · step " + artifactStep.toLocaleString();
      if (meta) meta.textContent = `${run} · ppl ${Number(data.ppl).toFixed(3)} · ${data.decoding || "greedy"} decoding` +
        (pending ? ` · generating ${Number(data.generation_steps || 0)}/${Number(data.max_new || 0)}` : "") +
        (fallback ? ` · newest gallery before scalar eval ${expectedStep.toLocaleString()}` : "") +
        (stale ? " · stale snapshot from an earlier generation of this step" : "");
      const items = data.items || [];
      const signature = JSON.stringify([
        data.complete, data.generation_steps,
        items.map(item => [item.image_url, item.tokens, item.stopped_at_eod,
                            item.complete, item.caption, item.reference, item.prompt, item.source,
                            item.target_boxes, item.predicted_boxes]),
      ]);
      // Preserve image/card DOM across polls. Replacing it forced the browser
      // to repaint every image, which made the gallery visibly flash.
      if (signature !== evalRenderedSignature) {
        renderEvalItems(body, items);
        evalRenderedSignature = signature;
      }
      // A scalar eval is published before the expensive greedy captions. Poll
      // only while that same card request remains current; switching run/eval
      // invalidates the request and prevents a stale response from taking over.
      if (pending) {
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl);
        }, 2000);
      }
    } catch (err) {
      if (request !== evalSampleRequest) return;
      if (err && err.name === "AbortError") {
        body.innerHTML = '<div class="empty">snapshot request timed out; retrying…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl, attempt);
        }, 2000);
        return;
      }
      body.innerHTML = "";
      const msg = document.createElement("div"); msg.className = "empty"; msg.textContent = err.message || String(err); body.append(msg);
    } finally {
      clearTimeout(timeout);
    }
  }
  window.trainboard = { watchActiveRun, openEvalSamples, resetEvalSamples };
  window.addEventListener("resize", () => {
    document.querySelectorAll("article.eval-sample").forEach(layoutEvalBoxOverlay);
  });

  // SSE remains the efficient primary transport. This tiny selected-run poll
  // is an independent recovery path for the two headline regions: browser tab
  // suspension, a missed nested-signal patch, or an SSE reconnect must never
  // leave eval PPL / best PPL stale until the user switches runs.
  let liveRefreshInFlight = false;
  async function refreshSelectedRun() {
    if (liveRefreshInFlight || document.hidden) return;
    const marker = document.getElementById("active-run");
    const run = marker && marker.dataset.run;
    if (!run) return;
    liveRefreshInFlight = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run)}/live`, {
        cache: "no-store",
        headers: { "Accept": "application/json" },
        signal: controller.signal,
      });
      if (!res.ok) return;
      const data = await res.json();
      // A run switch may have happened while this request was in flight.
      const current = document.getElementById("active-run");
      if (!current || current.dataset.run !== run || data.run !== run) return;
      const header = document.getElementById("run-header");
      const kpis = document.getElementById("run-kpis");
      if (header && data.header_html) header.outerHTML = data.header_html;
      if (kpis && data.kpi_html) kpis.outerHTML = data.kpi_html;
      const version = String(data.version || 0);
      if (current.dataset.v !== version) current.dataset.v = version;
    } catch (_) {
      // The live SSE still owns normal rendering; the next poll retries.
    } finally {
      clearTimeout(timeout);
      liveRefreshInFlight = false;
    }
  }
  setInterval(refreshSelectedRun, 1000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshSelectedRun();
  });

  // ---- keyboard: "/" focuses the run filter, j/k walk the (filtered) run list.
  // Esc (leaderboard close) is handled declaratively on <body> via Datastar.
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
                         t.tagName === "SELECT" || t.isContentEditable);
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") {
      const s = document.getElementById("run-search");
      if (s) { e.preventDefault(); s.focus(); s.select(); }
      return;
    }
    if (e.key === "j" || e.key === "k") {
      const items = Array.from(document.querySelectorAll("#run-list .run-item"))
        .filter((el) => el.style.display !== "none");
      if (!items.length) return;
      let idx = items.findIndex((el) => el.classList.contains("active"));
      if (idx < 0) idx = e.key === "j" ? 0 : items.length - 1;
      else idx = Math.min(items.length - 1, Math.max(0, idx + (e.key === "j" ? 1 : -1)));
      items[idx].click();
      items[idx].scrollIntoView({ block: "nearest" });
      e.preventDefault();
    }
  });

  // ---- heartbeat driver for state pulses (converting cell, stalling dot, …).
  // Deliberately JS-driven rather than @keyframes: OS/browser "disable
  // animations" settings can suppress CSS animations wholesale, and these
  // pulses carry live state. Writes a 0..1 sine to --pulse only while a
  // pulse-carrying element is actually on the page.
  const PULSE_SEL = ".conv-cell.converting, .dot.stalling, .queue-item.running, .alert.critical";
  setInterval(() => {
    const root = document.documentElement;
    if (!document.querySelector(PULSE_SEL)) {
      if (root.style.getPropertyValue("--pulse") !== "") root.style.removeProperty("--pulse");
      return;
    }
    const t = (Date.now() % 2600) / 2600;
    const s = 0.5 - 0.5 * Math.cos(2 * Math.PI * t);
    root.style.setProperty("--pulse", s.toFixed(3));
  }, 90);

  // ---- section jump chips in the sticky toolbar (data-jump="#panel-id").
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-jump]");
    if (!btn) return;
    const el = document.querySelector(btn.dataset.jump);
    if (!el) return;
    // jumping to a collapsed panel opens it (which also triggers its lazy load)
    if (el.tagName === "DETAILS" && !el.open) el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
